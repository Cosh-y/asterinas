use alloc::sync::Weak;
use x86::vmx::vmcs::{guest, host};
use x86_64::registers::control::{Cr0Flags, Cr4Flags};
use core::arch::x86_64::CpuidResult;

use ostd::arch::cpu::context::FpuContext;
use ostd::arch::{read_tsc, tsc_freq};
use ostd::arch::virt::*;
use ostd::mm::kspace::{read_bytes_from_paddr, read_u64_from_paddr};
use ostd::sync::SpinLock;

use crate::context::*;
use crate::emulate::apic::{default_lapic_ldr, lapic_check_pending_vector, Lapic, ApicTimer, TscState};
use crate::emulate::cr::sanitize_guest_cr4;
use crate::emulate::timer::{
    compute_preemption_timer_value, expire_lapic_timer_locked, timer_deactivate_locked, VMX_PREEMPTION_TIMER_MULTIPLIER_FALLBACK
};
use crate::error::*;
use crate::interrupt::{inject_lapic_interrupt, inject_pending_interrupt, ExceptionState, InterruptState, };
use crate::vm::Vm;
use crate::vmcs::{Vmcs, VmcsGuestState};

/// VCPU (Virtual CPU) instance
pub struct Vcpu {
    /// id
    id: u32,
    /// Parent VM reference
    pub(crate) vm: Weak<Vm>,
    /// VCPU state
    pub(crate) state: SpinLock<VcpuState>,
    /// VMCS
    vmcs: Vmcs,
}

/// VCPU state
#[derive(Debug, Default)]
pub struct VcpuState {
    arch: VcpuArchState,

    /// Running state
    pub running: bool,
    pub vmcs_launched: bool,
    pub vmcs_initialized: bool,
    /// Multiprocessor startup state used for INIT/SIPI handling.
    pub mp_state: VcpuMpState,
    
    /// Pending exception injection state.
    pub exception: ExceptionState,
    /// Pending interrupt injection state.
    pub interrupt: InterruptState,
    
    /// Virtual LAPIC state for this vCPU.
    pub lapic: Lapic,
    /// APIC timer state.
    pub apic_timer: ApicTimer,
    /// TSC-tracking state for virtual timer emulation.
    pub tsc: TscState,
}

#[derive(Debug, Default)]
pub struct VcpuArchState {
    /// General purpose registers
    regs: VcpuRegs,
    /// Special registers and descriptor tables provided by userspace.
    sregs: VcpuSregs,
    /// Guest-visible MSRs emulated by the hypervisor.
    msrs: VcpuMsrs,
    /// FPU/SIMD context.
    fpu: FpuContext,
}

#[derive(Debug, Clone, Copy)]
pub struct VcpuMsrs {
    pub apic_base: u64,
    pub efer: u64,
    pub pat: u64,
    pub fs_base: u64,
    pub gs_base: u64,
    pub kernel_gs_base: u64,
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub syscall_mask: u64,
    pub tsc_deadline: u64,
    pub tsc_adjust: u64,
    pub tsc_aux: u64,
    pub sysenter_cs: u64,
    pub sysenter_esp: u64,
    pub sysenter_eip: u64,
}


#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VcpuMpState {
    Runnable,
    WaitForSipi,
}

impl Vcpu {
    /// Gets the VCPU ID
    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn new(id: u32, vm: Weak<Vm>) -> Result<Self> {
        Ok(Self {
            id,
            vm,
            vmcs: Vmcs::new()?,
            state: SpinLock::new(VcpuState::with_vcpu_id(id)),
        })
    }

    /// Runs the VCPU
    pub fn run(&self) -> Result<super::handler::RunStateMessage> {
        if self.state.lock().mp_state == VcpuMpState::WaitForSipi {
            return Ok(self.wait_for_sipi_run_state());
        }

        // VMCS state is per-pCPU while loaded. Keep this run on one pCPU, then
        // clear the VMCS before returning so the next RSH_RUN may migrate safely.
        let _preempt_guard = ostd::task::disable_preempt();
        let _run_guard = self.enter_run()?;

        self.init_vmcs()?;

        loop {
            let irq_guard = ostd::irq::disable_local();

            let host_context = self.prepare_vmentry()?;
            let run_result = self.vmlaunch_or_vmresume();
            self.complete_vmexit(host_context, run_result)?;

            use super::handler::vmexit_handler;
            let exit_info = exit_info().map_err(Error::from)?;
            let run_state = vmexit_handler(self, &exit_info)?;
            drop(irq_guard);

            if let Some(run_state) = run_state {
                return Ok(run_state);
            }
        }
    }

    fn wait_for_sipi_run_state(&self) -> super::handler::RunStateMessage {
        let state = self.state.lock();
        // TODO: Use an enumeration to represent the reasons for exiting.
        const VMX_EXIT_REASON_HLT: u32 = 12;
        super::handler::RunStateMessage {
            exit_reason: VMX_EXIT_REASON_HLT,
            instruction_len: 0,
            guest_rip: state.arch.regs.rip,
            guest_phys_addr: 0,
            exit_qualification: 0,
        }
    }

    fn enter_run(&self) -> Result<VcpuRunGuard<'_>> {
        let mut state = self.state.lock();
        if state.running {
            return Err(Error::with_message(Errno::Busy, "vCPU is already running"));
        }
        state.running = true;
        Ok(VcpuRunGuard { vcpu: self })
    }

    fn init_vmcs(&self) -> Result<()> {
        if self.state.lock().vmcs_initialized {
            return Ok(());
        }
        
        log::error!("Vcpu init vmcs");
        let vmcs_guest_state = self.state.lock().arch.vmcs_guest_state();
        let eptp = self
                    .vm
                    .upgrade()
                    .ok_or_else(|| Error::with_message(Errno::NotFound, "vm not found"))?
                    .get_eptp();
        self.vmcs.init(vmcs_guest_state, eptp)?;

        self.state.lock().vmcs_initialized = true;

        Ok(())
    }

    fn prepare_vmentry(&self) -> Result<HostContext> {
        vmptrld(self.vmcs.vmcs_phys())?;

        self.prepare_pending_events()?;
        self.prepare_guest_timing_before_entry()?;
        
        let host_context = HostContext::save();
        if let Err(err) = self.load_guest_context() {
            host_context.load();
            return Err(err);
        }
        Ok(host_context)
    }

    fn vmlaunch_or_vmresume(&self) -> Result<()> {
        let launched: u64 = if self.state.lock().vmcs_launched { 1 } else { 0 };
        
        let ret = vcpu_run(&mut self.state.lock().arch.regs, launched);
        if ret != 0 {
            self.log_vm_entry_failure(launched);
            return Err(Error::with_message(
                Errno::GuestRunFailed,
                "vcpu_run failed",
            ));
        }
        
        self.state.lock().vmcs_launched = true;
        Ok(())
    }

    fn complete_vmexit(&self, host_context: HostContext, run_result: Result<()>) -> Result<()> {
        let save_guest_context_result = self.save_guest_context();
        host_context.load();
        
        run_result?;
        save_guest_context_result?;

        self.note_vmexit_tsc()?;
        
        Ok(())
    }

    fn load_guest_context(&self) -> Result<()> {
        let cr2 = self.state.lock().arch.sregs.cr2;
        write_cr2_raw(cr2);
        self.load_guest_run_msrs();
        let mut state = self.state.lock();
        state.arch.fpu.load();

        VmcsGuestNW::RIP.write(state.arch.regs.rip as usize).map_err(Error::from)?;
        VmcsGuestNW::RSP.write(state.arch.regs.rsp as usize).map_err(Error::from)?;
        // TODO: why | 0x2 ?
        VmcsGuestNW::RFLAGS.write((state.arch.regs.rflags | 0x2) as usize).map_err(Error::from)?;
        
        let guest_cr0 = state.arch.sregs.cr0;
        VmcsControlNW::CR0_READ_SHADOW
            .write(guest_cr0 as usize)
            .map_err(Error::from)?;
        VmcsGuestNW::CR0
            .write(guest_cr0 as usize)
            .map_err(Error::from)?;

        use x86::vmx::vmcs::control::EntryControls;
        use x86_64::registers::model_specific::EferFlags;
        let guest_efer = state.arch.msrs.efer;
        VmcsGuest64::IA32_EFER
            .write(guest_efer)
            .map_err(Error::from)?;
        let mut entry = VmcsControl32::VMENTRY_CONTROLS
            .read()
            .map_err(Error::from)?;
        if guest_efer & EferFlags::LONG_MODE_ACTIVE.bits() != 0 {
            entry |= EntryControls::IA32E_MODE_GUEST.bits();
        } else {
            entry &= !EntryControls::IA32E_MODE_GUEST.bits();
        }
        VmcsControl32::VMENTRY_CONTROLS
            .write(entry)
            .map_err(Error::from)?;

        let guest_cr3 = state.arch.sregs.cr3;
        VmcsGuestNW::CR3
            .write(guest_cr3 as usize)
            .map_err(Error::from)?;

        let guest_cr4 = state.arch.sregs.cr4;
        VmcsGuestNW::CR4
            .write(guest_cr4 as usize)
            .map_err(Error::from)?;
        VmcsControlNW::CR4_READ_SHADOW
            .write(guest_cr4 as usize)
            .map_err(Error::from)?;

        let msrs = state.arch.msrs;
        VmcsGuest64::IA32_PAT.write(msrs.pat).map_err(Error::from)?;
        VmcsGuestNW::FS_BASE
            .write(msrs.fs_base as usize)
            .map_err(Error::from)?;
        VmcsGuestNW::GS_BASE
            .write(msrs.gs_base as usize)
            .map_err(Error::from)?;
        VmcsGuest32::IA32_SYSENTER_CS
            .write(msrs.sysenter_cs as u32)
            .map_err(Error::from)?;
        VmcsGuestNW::IA32_SYSENTER_ESP
            .write(msrs.sysenter_esp as usize)
            .map_err(Error::from)?;
        VmcsGuestNW::IA32_SYSENTER_EIP
            .write(msrs.sysenter_eip as usize)
            .map_err(Error::from)?;

        Ok(())
    }

    fn save_guest_context(&self) -> Result<()> {
        self.state.lock().arch.fpu.save();
        self.save_guest_run_msrs()?;
        self.state.lock().arch.sregs.cr2 = read_cr2_raw();

        let mut state = self.state.lock();
        state.arch.regs.rsp    = VmcsGuestNW::RSP.read().map_err(Error::from)? as u64;
        state.arch.regs.rip    = VmcsGuestNW::RIP.read().map_err(Error::from)? as u64;
        state.arch.regs.rflags = VmcsGuestNW::RFLAGS.read().map_err(Error::from)? as u64;

        let guest_cr3 = VmcsGuestNW::CR3
            .read()
            .map_err(Error::from)?;
        state.arch.sregs.cr3 = guest_cr3 as u64;

        let guest_cr4 = VmcsGuestNW::CR4
            .read()
            .map_err(Error::from)?;
        state.arch.sregs.cr4 = guest_cr4 as u64;
        
        let guest_cr0 = VmcsGuestNW::CR0
            .read()
            .map_err(Error::from)?;
        state.arch.sregs.cr0 = guest_cr0 as u64;

        let guest_efer = VmcsGuest64::IA32_EFER
            .read()
            .map_err(Error::from)?;
        state.arch.msrs.efer = guest_efer;

        let sregs = &mut state.arch.sregs;
        sregs.gdt = read_dtable_from_vmcs(VmcsGuestNW::GDTR_BASE, VmcsGuest32::GDTR_LIMIT)?;
        sregs.idt = read_dtable_from_vmcs(VmcsGuestNW::IDTR_BASE, VmcsGuest32::IDTR_LIMIT)?;

        sregs.cs = read_segment_from_vmcs(
            VmcsGuest16::CS_SELECTOR,
            VmcsGuestNW::CS_BASE,
            VmcsGuest32::CS_LIMIT,
            VmcsGuest32::CS_ACCESS_RIGHTS,
        )?;
        sregs.ds = read_segment_from_vmcs(
            VmcsGuest16::DS_SELECTOR,
            VmcsGuestNW::DS_BASE,
            VmcsGuest32::DS_LIMIT,
            VmcsGuest32::DS_ACCESS_RIGHTS,
        )?;
        sregs.es = read_segment_from_vmcs(
            VmcsGuest16::ES_SELECTOR,
            VmcsGuestNW::ES_BASE,
            VmcsGuest32::ES_LIMIT,
            VmcsGuest32::ES_ACCESS_RIGHTS,
        )?;
        sregs.fs = read_segment_from_vmcs(
            VmcsGuest16::FS_SELECTOR,
            VmcsGuestNW::FS_BASE,
            VmcsGuest32::FS_LIMIT,
            VmcsGuest32::FS_ACCESS_RIGHTS,
        )?;
        sregs.gs = read_segment_from_vmcs(
            VmcsGuest16::GS_SELECTOR,
            VmcsGuestNW::GS_BASE,
            VmcsGuest32::GS_LIMIT,
            VmcsGuest32::GS_ACCESS_RIGHTS,
        )?;
        sregs.ss = read_segment_from_vmcs(
            VmcsGuest16::SS_SELECTOR,
            VmcsGuestNW::SS_BASE,
            VmcsGuest32::SS_LIMIT,
            VmcsGuest32::SS_ACCESS_RIGHTS,
        )?;
        sregs.tr = read_segment_from_vmcs(
            VmcsGuest16::TR_SELECTOR,
            VmcsGuestNW::TR_BASE,
            VmcsGuest32::TR_LIMIT,
            VmcsGuest32::TR_ACCESS_RIGHTS,
        )?;
        sregs.ldt = read_segment_from_vmcs(
            VmcsGuest16::LDTR_SELECTOR,
            VmcsGuestNW::LDTR_BASE,
            VmcsGuest32::LDTR_LIMIT,
            VmcsGuest32::LDTR_ACCESS_RIGHTS,
        )?;

        Ok(())
    }

    // TODO: understand this two functions.
    fn load_guest_run_msrs(&self) {
        let state = self.state.lock();
        Msr::IA32_STAR.write (state.arch.msrs.star );
        Msr::IA32_LSTAR.write(state.arch.msrs.lstar);
        Msr::IA32_CSTAR.write(state.arch.msrs.cstar);
        Msr::IA32_FMASK.write(state.arch.msrs.syscall_mask);
        Msr::IA32_KERNEL_GSBASE.write(state.arch.msrs.kernel_gs_base);
    }

    fn save_guest_run_msrs(&self) -> Result<()> {
        let star  = Msr::IA32_STAR.read();
        let lstar = Msr::IA32_LSTAR.read();
        let cstar = Msr::IA32_CSTAR.read();
        let syscall_mask   = Msr::IA32_FMASK.read();
        let kernel_gs_base = Msr::IA32_KERNEL_GSBASE.read();
        let fs_base = VmcsGuestNW::FS_BASE.read().map_err(Error::from)? as u64;
        let gs_base = VmcsGuestNW::GS_BASE.read().map_err(Error::from)? as u64;

        let mut state = self.state.lock();
        state.arch.msrs.star  = star;
        state.arch.msrs.lstar = lstar;
        state.arch.msrs.cstar = cstar;
        state.arch.msrs.syscall_mask   = syscall_mask;
        state.arch.msrs.kernel_gs_base = kernel_gs_base;
        state.arch.msrs.fs_base  = fs_base;
        state.arch.msrs.gs_base  = gs_base;
        state.arch.sregs.fs.base = fs_base;
        state.arch.sregs.gs.base = gs_base;
        Ok(())
    }

    /// Gets general purpose registers
    pub fn get_regs(&self) -> Result<VcpuRegs> {
        let state = self.state.lock();
        if state.running {
            return Err(Error::with_message(Errno::Busy, "Can't get regs while vcpu is running."));
        }
        Ok(state.arch.regs)
    }

    /// Sets general purpose registers
    pub fn set_regs(&self, regs: VcpuRegs) -> Result<()> {
        let mut state = self.state.lock();
        if state.running {
            return Err(Error::with_message(Errno::Busy, "Can't set regs while vcpu is running."));
        }
        state.arch.regs = regs;
        Ok(())
    }

    /// Gets special registers.
    pub fn get_sregs(&self) -> Result<VcpuSregs> {
        let mut state = self.state.lock();
        if state.running {
            return Err(Error::with_message(Errno::Busy, "Can't get sregs while vcpu is running."));
        }
        Ok(state.arch.sregs)
    }

    /// Sets special registers.
    pub fn set_sregs(&self, sregs: VcpuSregs) -> Result<()> {
        let mut state = self.state.lock();
        if state.running {
            return Err(Error::with_message(Errno::Busy, "Can't set sregs while vcpu is running."));
        }
        // TODO: clear msrs in sregs
        state.arch.sregs        = sregs;
        state.arch.sregs.cr4    = sanitize_guest_cr4(sregs.cr4);
        state.arch.msrs.efer    = sregs.efer;
        state.arch.msrs.fs_base = sregs.fs.base;
        state.arch.msrs.gs_base = sregs.gs.base;
        Ok(())
    }

    fn log_vm_entry_failure(&self, vmcs_launched: u64) {
        let vm_instruction_error = VmcsReadOnly32::VM_INSTRUCTION_ERROR.read().ok();
        let exit_reason = VmcsReadOnly32::EXIT_REASON.read().ok();
        let guest_rip = VmcsGuestNW::RIP.read().ok();
        let guest_rsp = VmcsGuestNW::RSP.read().ok();
        let guest_rflags = VmcsGuestNW::RFLAGS.read().ok();
        log::error!(
            "rustshyper: {} failed for vcpu id={} vmcs={:#x} vm_instruction_error={:#x?} exit_reason={:#x?} guest_rip={:#x?} guest_rsp={:#x?} guest_rflags={:#x?}",
            if vmcs_launched == 0 {
                "vmlaunch"
            } else {
                "vmresume"
            },
            self.id,
            self.vmcs.vmcs_phys(),
            vm_instruction_error,
            exit_reason,
            guest_rip,
            guest_rsp,
            guest_rflags
        );
    }

    /// TODO: 现在 note 和 refresh 的实现是一样的？
    /// Record Guest TSC value at VM exit.
    pub(crate) fn note_vmexit_tsc(&self) -> Result<()> {
        let mut state = self.state.lock();
        state.tsc.tsc_physical = state.tsc.tsc_offset.wrapping_add(read_tsc());
        Ok(())
    }

    /// Refreshes the guest-visible TSC before VM-entry.
    pub(crate) fn refresh_guest_tsc(&self) {
        let mut state = self.state.lock();
        state.tsc.tsc_physical = state.tsc.tsc_offset.wrapping_add(read_tsc());
    }

    fn prepare_pending_events(&self) -> Result<()> {
        let mut state = self.state.lock();

        // clear_event_injection()?;

        // if has_pending_exception(&state.exception) {
        //     inject_pending_exception(&mut state.exception)?;
        //     return Ok(());
        // }

        {
            let VcpuState {
                lapic, interrupt, ..
            } = &mut *state;
            inject_pending_interrupt(lapic, interrupt)?;
        }
        if state.interrupt.pending {
            return Ok(());
        }

        if let Some(vector) = lapic_check_pending_vector(&state.lapic) {
            inject_lapic_interrupt(&mut state.lapic, u32::from(vector))?;
        }

        Ok(())
    }

    /// 在 VMEntry 前更新 tsc offset 并设置 VMCS 中的 preemption timer 和 tsc offset
    fn prepare_guest_timing_before_entry(&self) -> Result<()> {
        self.refresh_guest_tsc();

        let state = self.state.lock();
        let preemption_timer = compute_preemption_timer_value(&state);
        VmcsGuest32::VMX_PREEMPTION_TIMER_VALUE.write(preemption_timer)?;
        VmcsControl64::TSC_OFFSET.write(state.tsc.tsc_offset)?;
        Ok(())
    }

    fn translate_guest_gpa(&self, gpa: u64) -> Result<u64> {
        self
            .vm
            .upgrade()
            .ok_or_else(|| Error::with_message(Errno::NotFound, "vm not found"))?
            .translate_gpa_to_hpa(gpa)
    }

    fn translate_guest_gva(&self, gva: u64) -> Result<u64> {
        const PTE_PRESENT: u64 = 1 << 0;
        const PTE_HUGE: u64 = 1 << 7;
        const PTE_ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;
        const PAGE_2M_MASK: u64 = (1 << 21) - 1;
        const PAGE_1G_MASK: u64 = (1 << 30) - 1;

        let cr0 = VmcsGuestNW::CR0.read().map_err(Error::from)? as u64;
        if (cr0 & (1 << 31)) == 0 {
            return Ok(gva);
        }

        let cr3 = (VmcsGuestNW::CR3.read().map_err(Error::from)? as u64) & !0xfff;
        let pml4e = self.read_guest_phys_u64(cr3 + (((gva >> 39) & 0x1ff) * 8))?;
        if (pml4e & PTE_PRESENT) == 0 {
            return Err(Error::with_message(
                Errno::Fault,
                "guest PML4 entry is not present",
            ));
        }

        let pdpte =
            self.read_guest_phys_u64((pml4e & PTE_ADDR_MASK) + (((gva >> 30) & 0x1ff) * 8))?;
        if (pdpte & PTE_PRESENT) == 0 {
            return Err(Error::with_message(
                Errno::Fault,
                "guest PDPT entry is not present",
            ));
        }
        if (pdpte & PTE_HUGE) != 0 {
            return Ok((pdpte & PTE_ADDR_MASK) | (gva & PAGE_1G_MASK));
        }

        let pde =
            self.read_guest_phys_u64((pdpte & PTE_ADDR_MASK) + (((gva >> 21) & 0x1ff) * 8))?;
        if (pde & PTE_PRESENT) == 0 {
            return Err(Error::with_message(
                Errno::Fault,
                "guest PD entry is not present",
            ));
        }
        if (pde & PTE_HUGE) != 0 {
            return Ok((pde & PTE_ADDR_MASK) | (gva & PAGE_2M_MASK));
        }

        let pte = self.read_guest_phys_u64((pde & PTE_ADDR_MASK) + (((gva >> 12) & 0x1ff) * 8))?;
        if (pte & PTE_PRESENT) == 0 {
            return Err(Error::with_message(
                Errno::Fault,
                "guest PT entry is not present",
            ));
        }

        Ok((pte & PTE_ADDR_MASK) | (gva & 0xfff))
    }

    pub(crate) fn read_guest_memory(&self, gva: u64, buf: &mut [u8]) -> Result<()> {
        for (index, byte) in buf.iter_mut().enumerate() {
            let gpa = self.translate_guest_gva(gva.wrapping_add(index as u64))?;
            let hpa = self.translate_guest_gpa(gpa)?;
            read_bytes_from_paddr(hpa as usize, core::slice::from_mut(byte));
        }
        Ok(())
    }

    fn read_guest_phys_u64(&self, gpa: u64) -> Result<u64> {
        let hpa = self.translate_guest_gpa(gpa)?;
        Ok(read_u64_from_paddr(hpa as usize))
    }
}

impl VcpuState {
    fn with_vcpu_id(vcpu_id: u32) -> VcpuState {
        let mut state = VcpuState::default();
        state.vmcs_initialized = false;
        state.vmcs_launched = false;
        state.arch.msrs = VcpuMsrs::default();
        if vcpu_id != 0 {
            state.arch.msrs.apic_base &= !(1 << 8);
            state.mp_state = VcpuMpState::WaitForSipi;
        }
        state.lapic.id = vcpu_id;
        state.lapic.ldr = default_lapic_ldr(vcpu_id);
        state.apic_timer.lvt_timer_bits = 1 << 16;
        // Some virtualized environments expose enough VMX state to run the guest
        // but still #GP on RDMSR IA32_VMX_MISC (0x485). Use a marker value and
        // poll active virtual deadlines at a bounded interval.
        state.tsc.multiplier = VMX_PREEMPTION_TIMER_MULTIPLIER_FALLBACK;
        
        state
    }

    pub(crate) fn arch(&self) -> &VcpuArchState {
        &self.arch
    }

    pub(crate) fn arch_mut(&mut self) -> &mut VcpuArchState {
        &mut self.arch
    }
}

impl VcpuArchState {
    pub(crate) fn gpr(&self, index: u8) -> u64 {
        match index {
            // the order comes from 
            // Intel® 64 and IA-32 Architectures Software Developer’s Manual
            // 3.4.1 General-Purpose Registers
            0  => self.regs.rax,
            1  => self.regs.rbx,
            2  => self.regs.rcx,
            3  => self.regs.rdx,
            4  => self.regs.rsi,
            5  => self.regs.rdi,
            6  => self.regs.rbp,
            7  => self.regs.rsp,
            8  => self.regs.r8,
            9  => self.regs.r9,
            10 => self.regs.r10,
            11 => self.regs.r11,
            12 => self.regs.r12,
            13 => self.regs.r13,
            14 => self.regs.r14,
            15 => self.regs.r15,
            _ => 0,
        }
    }

    pub(crate) fn set_gpr(&mut self, index: u8, width_byte: u8, value: u64) {
        let slot = match index {
            0  => &mut self.regs.rax,
            1  => &mut self.regs.rbx,
            2  => &mut self.regs.rcx,
            3  => &mut self.regs.rdx,
            4  => &mut self.regs.rsi,
            5  => &mut self.regs.rdi,
            6  => &mut self.regs.rbp,
            7  => &mut self.regs.rsp,
            8  => &mut self.regs.r8,
            9  => &mut self.regs.r9,
            10 => &mut self.regs.r10,
            11 => &mut self.regs.r11,
            12 => &mut self.regs.r12,
            13 => &mut self.regs.r13,
            14 => &mut self.regs.r14,
            15 => &mut self.regs.r15,
            _ => return,
        };

        *slot = match width_byte {
            1 => (*slot & !0xff) | (value & 0xff),
            2 => (*slot & !0xffff) | (value & 0xffff),
            4 => (*slot & !0xffff_ffff) | (value & 0xffff_ffff),
            _ => value,
        };
    }

    pub(crate) fn advance_rip(&mut self, len: u64) {
        self.regs.rip += len;
    }

    pub(crate) fn msr(&self, index: u32) -> u64 {
        use x86::msr::*;
        match index {
            IA32_TSC_ADJUST    => self.msrs.tsc_adjust,
            IA32_APIC_BASE     => self.msrs.apic_base,
            IA32_SYSENTER_CS   => self.msrs.sysenter_cs,
            IA32_SYSENTER_ESP  => self.msrs.sysenter_esp,
            IA32_SYSENTER_EIP  => self.msrs.sysenter_eip,
            IA32_EFER          => self.msrs.efer,
            IA32_PAT           => self.msrs.pat,
            IA32_FS_BASE       => self.msrs.fs_base,
            IA32_GS_BASE       => self.msrs.gs_base,
            IA32_KERNEL_GSBASE => self.msrs.kernel_gs_base,
            IA32_TSC_AUX       => self.msrs.tsc_aux,
            IA32_STAR          => self.msrs.star,
            IA32_LSTAR         => self.msrs.lstar,
            IA32_CSTAR         => self.msrs.cstar,
            IA32_FMASK         => self.msrs.syscall_mask,
            IA32_TSC_DEADLINE  => self.msrs.tsc_deadline,
            _ => {
                log::error!("get unknown msr {:x}, return 0.", index);
                0
            }
        }
    }

    pub(crate) fn set_msr(&mut self, index: u32, value: u64) {
        use x86::msr::*;
        match index {
            IA32_TSC_ADJUST   => self.msrs.tsc_adjust   = value,
            IA32_APIC_BASE    => self.msrs.apic_base    = value,
            IA32_SYSENTER_CS  => self.msrs.sysenter_cs  = value,
            IA32_SYSENTER_ESP => self.msrs.sysenter_esp = value,
            IA32_SYSENTER_EIP => self.msrs.sysenter_eip = value,
            IA32_EFER    => self.msrs.efer    = value,
            IA32_PAT     => self.msrs.pat     = value,
            IA32_FS_BASE => self.msrs.fs_base = value,
            IA32_GS_BASE => self.msrs.gs_base = value,
            IA32_KERNEL_GSBASE => self.msrs.kernel_gs_base = value,
            IA32_TSC_AUX => self.msrs.tsc_aux = value,
            IA32_STAR    => self.msrs.star  = value,
            IA32_LSTAR   => self.msrs.lstar = value,
            IA32_CSTAR   => self.msrs.cstar = value,
            IA32_FMASK   => self.msrs.syscall_mask = value,
            IA32_TSC_DEADLINE => self.msrs.tsc_deadline = value,
            _ => log::error!("set_msr: msr {:x} not impl.", index),
        }
    }

    pub(crate) fn cr0(&self) -> u64 {
        self.sregs.cr0
    }

    pub(crate) fn cr2(&self) -> u64 {
        self.sregs.cr2
    }

    pub(crate) fn cr3(&self) -> u64 {
        self.sregs.cr3
    }

    pub(crate) fn cr4(&self) -> u64 {
        self.sregs.cr4
    }

    pub(crate) fn set_cr0(&mut self, value: u64) {
        self.sregs.cr0 = value;
    }

    pub(crate) fn set_cr2(&mut self, value: u64) {
        self.sregs.cr2 = value;
    }

    pub(crate) fn set_cr3(&mut self, value: u64) {
        self.sregs.cr3 = value;
    }

    pub(crate) fn set_cr4(&mut self, value: u64) {
        self.sregs.cr4 = value;
    }
    
    fn vmcs_guest_state(&self) -> VmcsGuestState {
        VmcsGuestState {
            regs:  self.regs,
            sregs: self.sregs,
            msrs:  self.msrs,
        }
    }
}

struct VcpuRunGuard<'a> {
    vcpu: &'a Vcpu,
}

impl Drop for VcpuRunGuard<'_> {
    fn drop(&mut self) {
        if let Err(err) = vmclear(self.vcpu.vmcs.vmcs_phys()) {
            log::error!(
                "rustshyper: failed to vmclear vcpu id={} vmcs={:#x}: {:?}",
                self.vcpu.id,
                self.vcpu.vmcs.vmcs_phys(),
                err
            );
        }

        let mut state = self.vcpu.state.lock();
        state.vmcs_launched = false;
        state.running = false;
    }
}

impl Default for VcpuMsrs {
    fn default() -> Self {
        const APIC_BASE_BSP: u64 = 1 << 8;
        const APIC_BASE_ENABLE: u64 = 1 << 11;

        Self {
            apic_base: 0xFEE0_0000_u64 | APIC_BASE_BSP | APIC_BASE_ENABLE,
            efer: Msr::IA32_EFER.read(),
            pat: Msr::IA32_PAT.read(),
            fs_base: 0,
            gs_base: 0,
            kernel_gs_base: 0,
            star: 0,
            lstar: 0,
            cstar: 0,
            syscall_mask: 0,
            tsc_deadline: 0,
            tsc_adjust: 0,
            tsc_aux: 0,
            sysenter_cs: 0,
            sysenter_esp: 0,
            sysenter_eip: 0,
        }
    }
}

impl Default for VcpuMpState {
    fn default() -> Self {
        Self::Runnable
    }
}

// TODO: consider removing the following part from vcpu.rs
fn read_dtable_from_vmcs(base_field: VmcsGuestNW, limit_field: VmcsGuest32) -> Result<VcpuDtable> {
    Ok(VcpuDtable {
        base: base_field.read().map_err(Error::from)? as u64,
        limit: limit_field.read().map_err(Error::from)? as u16,
        padding: [0; 3],
    })
}

fn read_segment_from_vmcs(
    selector_field: VmcsGuest16,
    base_field: VmcsGuestNW,
    limit_field: VmcsGuest32,
    rights_field: VmcsGuest32,
) -> Result<VcpuSegment> {
    let rights = rights_field.read().map_err(Error::from)?;
    Ok(VcpuSegment {
        base: base_field.read().map_err(Error::from)? as u64,
        limit: limit_field.read().map_err(Error::from)?,
        selector: selector_field.read().map_err(Error::from)?,
        type_: (rights & 0x0f) as u8,
        s: ((rights >> 4) & 0x1) as u8,
        dpl: ((rights >> 5) & 0x3) as u8,
        present: ((rights >> 7) & 0x1) as u8,
        avl: ((rights >> 12) & 0x1) as u8,
        l: ((rights >> 13) & 0x1) as u8,
        db: ((rights >> 14) & 0x1) as u8,
        g: ((rights >> 15) & 0x1) as u8,
        unusable: ((rights >> 16) & 0x1) as u8,
        padding: 0,
    })
}

pub(crate) fn reset_vcpu_for_init_locked(state: &mut VcpuState, vcpu_id: u32) {
    // *state = VcpuState::with_vcpu_id(vcpu_id);
    // (*state).arch.msrs.efer   = 0;
    // (*state).arch.regs.rflags = 0x2;
}

pub(crate) fn start_vcpu_from_sipi_locked(state: &mut VcpuState, vector: u8, vcpu_id: u32) {
    state.arch.regs = VcpuRegs {
        rip: 0,
        rflags: 0x2,
        ..VcpuRegs::default()
    };
    state.arch.sregs = real_mode_sregs(vector);
    state.arch.msrs.efer = 0;
    // state.arch.msrs.fs_base = 0;
    // state.arch.msrs.gs_base = 0;
    // state.arch.msrs.kernel_gs_base = 0;
    state.arch.msrs.tsc_aux = u64::from(vcpu_id);
    // state.vmcs_initialized = false;
    // state.vmcs_launched = false;
    // state.exception = ExceptionState::default();
    // state.interrupt = InterruptState::default();
    // timer_deactivate_locked(state);
    state.mp_state = VcpuMpState::Runnable;
}

fn real_mode_sregs(startup_vector: u8) -> VcpuSregs {
    let code_base = u64::from(startup_vector) << 12;
    let code_selector = u16::from(startup_vector) << 8;
    let data = real_mode_data_segment(0, 0);

    VcpuSregs {
        cs: real_mode_code_segment(code_selector, code_base),
        ds: data,
        es: data,
        fs: data,
        gs: data,
        ss: data,
        tr: real_mode_system_segment(0x20, 0, 0x0b),
        ldt: VcpuSegment {
            unusable: 1,
            ..VcpuSegment::default()
        },
        cr0: (Cr0Flags::EXTENSION_TYPE | Cr0Flags::NUMERIC_ERROR).bits(),
        cr4: Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS.bits(),
        ..VcpuSregs::default()
    }
}

fn real_mode_code_segment(selector: u16, base: u64) -> VcpuSegment {
    real_mode_segment(selector, base, 0x0b, 1)
}

fn real_mode_data_segment(selector: u16, base: u64) -> VcpuSegment {
    real_mode_segment(selector, base, 0x03, 1)
}

fn real_mode_system_segment(selector: u16, base: u64, type_: u8) -> VcpuSegment {
    real_mode_segment(selector, base, type_, 0)
}

fn real_mode_segment(selector: u16, base: u64, type_: u8, s: u8) -> VcpuSegment {
    VcpuSegment {
        base,
        limit: 0xffff,
        selector,
        type_,
        present: 1,
        dpl: 0,
        db: 0,
        s,
        l: 0,
        g: 0,
        avl: 0,
        unusable: 0,
        padding: 0,
    }
}
