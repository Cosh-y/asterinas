use x86_64::registers::control::Cr0Flags;

use super::{
    control_regs::VcpuControlRegisters,
    types::{VcpuDtable, VcpuMsrs, VcpuRegs, VcpuSegment, VcpuSregs, X86GprIndex},
    vmcs::{Vmcs, VmcsGuestState},
};
use crate::{arch::cpu::context::FpuContext, prelude::*};

/// Stores the execution context and run state of a guest vCPU.
///
/// The kernel uses it to configure the vCPU-visible context, including
/// general-purpose registers, special registers, and MSRs.
///
/// OSTD uses it to provide [`crate::vm::GuestMode`] with the state needed to
/// run the vCPU. Before entering the vCPU, `GuestMode` loads the context into
/// hardware. After a VM exit, `GuestMode` synchronizes the hardware vCPU state
/// back into this context.
pub struct GuestContext {
    /// The guest architectural state.
    arch: VcpuArchState,

    /// The vCPU run state.
    run: VcpuRunState,

    /// The VMCS owned by this vCPU.
    pub(crate) vmcs: Vmcs,

    pub(crate) tsc_offset: i64,
}

pub(crate) struct VcpuArchState {
    /// General-purpose registers.
    regs: VcpuRegs,
    /// Special registers and descriptor tables provided by userspace.
    sregs: VcpuSregs,
    /// VMX control-register state split into guest-visible and hardware values.
    control_regs: VcpuControlRegisters,
    /// Guest-visible MSRs emulated by the hypervisor.
    msrs: VcpuMsrs,
    /// FPU/SIMD context.
    fpu: FpuContext,
}

impl GuestContext {
    /// Creates a guest vCPU context.
    ///
    /// The bootstrap vCPU, whose ID is zero, starts in the runnable state.
    /// Other vCPUs start in wait-for-SIPI state and become runnable after
    /// [`Self::receive_sipi`] accepts a startup vector.
    ///
    /// Returns an error if the VMX virtualization environment is not ready.
    pub fn new(id: u32) -> Result<Self> {
        Ok(Self {
            arch: VcpuArchState::default(),
            run: if id == 0 {
                VcpuRunState::Runnable
            } else {
                VcpuRunState::WaitForSipi
            },
            vmcs: Vmcs::new()?,
            tsc_offset: 0,
        })
    }

    /// Moves an AP vCPU from wait-for-SIPI state to runnable state.
    ///
    /// The startup vector is used to rebuild the vCPU's real-mode startup
    /// state. Calling this method for a vCPU that is not waiting for SIPI has
    /// no effect.
    pub fn receive_sipi(&mut self, vector: u8) {
        if self.run != VcpuRunState::WaitForSipi {
            return;
        }

        self.arch.regs = VcpuRegs {
            rip: 0,
            rflags: 0x2,
            ..VcpuRegs::default()
        };
        self.arch.set_sregs(VcpuSregs::with_startup(vector));
        self.arch.set_efer(0);
        // self.arch.msrs.tsc_aux = u64::from(vcpu_id);
        self.run = VcpuRunState::Runnable;
    }

    /// Returns the guest general-purpose register state.
    pub fn regs(&self) -> VcpuRegs {
        self.arch.regs
    }

    /// Replaces the guest general-purpose register state.
    ///
    /// This method stores the values as guest-visible state. The caller is
    /// responsible for choosing register values that make sense for the guest
    /// execution mode and entry point.
    pub fn set_regs(&mut self, regs: VcpuRegs) {
        self.arch.regs = regs;
    }

    /// Returns the guest special-register state.
    ///
    /// The returned state contains the guest-visible control-register values,
    /// not the VMX-adjusted hardware values used internally for VM entry.
    pub fn sregs(&self) -> VcpuSregs {
        self.arch.sregs()
    }

    /// Replaces the guest special-register state.
    ///
    /// This method stores the supplied guest-visible state and rebuilds the
    /// VMX control-register state derived from CR0 and CR4. The caller remains
    /// responsible for providing architecturally valid guest state.
    pub fn set_sregs(&mut self, sregs: VcpuSregs) {
        self.arch.set_sregs(sregs);
    }

    /// Returns a guest general-purpose register.
    pub fn gpr(&self, reg: X86GprIndex) -> u64 {
        self.arch.gpr(reg)
    }

    /// Updates a guest general-purpose register.
    ///
    /// The `width_byte` argument controls whether the low 1, 2, 4, or 8 bytes
    /// are updated. The caller is responsible for using a register and width
    /// that match the emulated guest instruction.
    pub fn set_gpr(&mut self, reg: X86GprIndex, width_byte: u8, value: u64) {
        self.arch.set_gpr(reg, width_byte, value);
    }

    /// Advances the guest instruction pointer.
    ///
    /// The caller is responsible for passing the length of the instruction
    /// that has actually been consumed or emulated.
    pub fn advance_rip(&mut self, len: u64) {
        self.arch.advance_rip(len);
    }

    /// Returns the guest instruction pointer.
    pub fn rip(&self) -> u64 {
        self.arch.rip()
    }

    /// Returns the vCPU execution state.
    pub fn run_state(&self) -> VcpuRunState {
        self.run
    }

    /// Sets the vCPU execution state.
    pub fn set_run_state(&mut self, state: VcpuRunState) {
        self.run = state;
    }

    /// Returns whether the guest vCPU is currently running.
    pub fn is_running(&self) -> bool {
        self.run == VcpuRunState::Running
    }

    /// Returns the guest-visible TSC value.
    pub fn guest_tsc(&self) -> u64 {
        use crate::arch::read_tsc;
        let tsc = read_tsc() as i64 + self.tsc_offset;
        if tsc < 0 { 0 } else { tsc as u64 }
    }

    /// Sets the offset added to the host TSC when computing the guest TSC.
    pub fn set_tsc_offset(&mut self, offset: i64) {
        self.tsc_offset = offset;
    }

    /// Adds a delta to the offset used when computing the guest TSC.
    pub fn adjust_tsc_offset(&mut self, delta: i64) {
        self.tsc_offset = self.tsc_offset.wrapping_add(delta);
    }

    /// Returns the guest-visible value of a stored guest MSR.
    ///
    /// Unsupported MSR indexes return `None`.
    pub fn read_msr(&self, index: u32) -> Option<u64> {
        self.arch.try_msr(index)
    }

    /// Sets the guest-visible value of a stored guest MSR.
    ///
    /// Returns `false` if the MSR index is not supported by this context.
    pub fn write_msr(&mut self, index: u32, value: u64) -> bool {
        self.arch.set_msr(index, value)
    }

    pub(crate) fn vmcs_guest_state(&self) -> VmcsGuestState {
        self.arch.vmcs_guest_state()
    }

    pub(crate) fn arch_mut(&mut self) -> &mut VcpuArchState {
        &mut self.arch
    }

    pub(crate) fn arch(&self) -> &VcpuArchState {
        &self.arch
    }
}

impl Default for VcpuArchState {
    fn default() -> Self {
        let sregs = VcpuSregs::with_startup(0);
        Self {
            regs: VcpuRegs {
                rflags: 0x2,
                ..VcpuRegs::default()
            },
            sregs,
            control_regs: VcpuControlRegisters::from_sregs(&sregs),
            msrs: VcpuMsrs::default(),
            fpu: FpuContext::new(),
        }
    }
}

impl Default for GuestContext {
    fn default() -> Self {
        Self::new(0).expect("failed to create guest context")
    }
}

/// Describes whether a guest vCPU may enter guest mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VcpuRunState {
    /// The vCPU is not initialized for execution.
    Uninitialized,
    /// The vCPU is waiting for a startup IPI.
    WaitForSipi,
    /// The vCPU is ready to enter guest mode.
    #[default]
    Runnable,
    /// The vCPU is currently executing in guest mode.
    Running,
    /// The vCPU Halted.
    Halted,
}

impl VcpuArchState {
    pub(crate) fn regs_mut_ptr(&mut self) -> *mut VcpuRegs {
        &mut self.regs
    }

    pub(crate) fn sregs(&self) -> VcpuSregs {
        let mut sregs = self.sregs;
        sregs.cr0 = self.cr0();
        sregs.cr4 = self.cr4();
        sregs
    }

    pub(crate) fn set_sregs(&mut self, sregs: VcpuSregs) {
        self.sregs = sregs;
        self.control_regs = VcpuControlRegisters::from_sregs(&sregs);
        self.msrs.efer = sregs.efer;
        self.msrs.apic_base = sregs.apic_base;
        self.msrs.fs_base = sregs.fs.base;
        self.msrs.gs_base = sregs.gs.base;
    }

    pub fn gpr(&self, reg: X86GprIndex) -> u64 {
        match reg {
            X86GprIndex::Rax => self.regs.rax,
            X86GprIndex::Rbx => self.regs.rbx,
            X86GprIndex::Rcx => self.regs.rcx,
            X86GprIndex::Rdx => self.regs.rdx,
            X86GprIndex::Rsi => self.regs.rsi,
            X86GprIndex::Rdi => self.regs.rdi,
            X86GprIndex::Rbp => self.regs.rbp,
            X86GprIndex::Rsp => self.regs.rsp,
            X86GprIndex::R8 => self.regs.r8,
            X86GprIndex::R9 => self.regs.r9,
            X86GprIndex::R10 => self.regs.r10,
            X86GprIndex::R11 => self.regs.r11,
            X86GprIndex::R12 => self.regs.r12,
            X86GprIndex::R13 => self.regs.r13,
            X86GprIndex::R14 => self.regs.r14,
            X86GprIndex::R15 => self.regs.r15,
        }
    }

    pub fn set_gpr(&mut self, reg: X86GprIndex, width_byte: u8, value: u64) {
        let slot = match reg {
            X86GprIndex::Rax => &mut self.regs.rax,
            X86GprIndex::Rbx => &mut self.regs.rbx,
            X86GprIndex::Rcx => &mut self.regs.rcx,
            X86GprIndex::Rdx => &mut self.regs.rdx,
            X86GprIndex::Rsi => &mut self.regs.rsi,
            X86GprIndex::Rdi => &mut self.regs.rdi,
            X86GprIndex::Rbp => &mut self.regs.rbp,
            X86GprIndex::Rsp => &mut self.regs.rsp,
            X86GprIndex::R8 => &mut self.regs.r8,
            X86GprIndex::R9 => &mut self.regs.r9,
            X86GprIndex::R10 => &mut self.regs.r10,
            X86GprIndex::R11 => &mut self.regs.r11,
            X86GprIndex::R12 => &mut self.regs.r12,
            X86GprIndex::R13 => &mut self.regs.r13,
            X86GprIndex::R14 => &mut self.regs.r14,
            X86GprIndex::R15 => &mut self.regs.r15,
        };

        *slot = match width_byte {
            1 => (*slot & !0xff) | (value & 0xff),
            2 => (*slot & !0xffff) | (value & 0xffff),
            4 => (*slot & !0xffff_ffff) | (value & 0xffff_ffff),
            _ => value,
        };
    }

    pub fn advance_rip(&mut self, len: u64) {
        self.regs.rip += len;
    }

    pub(crate) fn rip(&self) -> u64 {
        self.regs.rip
    }

    pub(crate) fn set_rip(&mut self, value: u64) {
        self.regs.rip = value;
    }

    pub(crate) fn rflags(&self) -> u64 {
        self.regs.rflags
    }

    pub(crate) fn set_rflags(&mut self, value: u64) {
        self.regs.rflags = value;
    }

    pub(crate) fn msr(&self, index: u32) -> u64 {
        self.try_msr(index).unwrap_or_else(|| {
            error!("get unknown msr {:x}, return 0.", index);
            0
        })
    }

    fn try_msr(&self, index: u32) -> Option<u64> {
        use x86::msr::*;

        Some(match index {
            IA32_TSC_ADJUST => self.msrs.tsc_adjust,
            IA32_APIC_BASE => self.msrs.apic_base,
            IA32_SYSENTER_CS => self.msrs.sysenter_cs,
            IA32_SYSENTER_ESP => self.msrs.sysenter_esp,
            IA32_SYSENTER_EIP => self.msrs.sysenter_eip,
            IA32_EFER => self.msrs.efer,
            IA32_PAT => self.msrs.pat,
            IA32_FS_BASE => self.msrs.fs_base,
            IA32_GS_BASE => self.msrs.gs_base,
            IA32_KERNEL_GSBASE => self.msrs.kernel_gs_base,
            IA32_TSC_AUX => self.msrs.tsc_aux,
            IA32_STAR => self.msrs.star,
            IA32_LSTAR => self.msrs.lstar,
            IA32_CSTAR => self.msrs.cstar,
            IA32_FMASK => self.msrs.syscall_mask,
            IA32_MISC_ENABLE => self.msrs.misc_enable,
            _ => return None,
        })
    }

    pub(crate) fn set_msr(&mut self, index: u32, value: u64) -> bool {
        use x86::msr::*;
        match index {
            IA32_TSC_ADJUST => self.msrs.tsc_adjust = value,
            IA32_APIC_BASE => {
                self.msrs.apic_base = value;
                self.sregs.apic_base = value;
            }
            IA32_SYSENTER_CS => self.msrs.sysenter_cs = value,
            IA32_SYSENTER_ESP => self.msrs.sysenter_esp = value,
            IA32_SYSENTER_EIP => self.msrs.sysenter_eip = value,
            IA32_EFER => self.set_efer(value),
            IA32_PAT => self.msrs.pat = value,
            IA32_KERNEL_GSBASE => self.msrs.kernel_gs_base = value,
            IA32_TSC_AUX => self.msrs.tsc_aux = value,
            IA32_STAR => self.msrs.star = value,
            IA32_LSTAR => self.msrs.lstar = value,
            IA32_CSTAR => self.msrs.cstar = value,
            IA32_FMASK => self.msrs.syscall_mask = value,
            IA32_FS_BASE => self.set_fs_base(value),
            IA32_GS_BASE => self.set_gs_base(value),
            IA32_MISC_ENABLE => self.msrs.misc_enable = value,
            _ => return false,
        }

        true
    }

    pub(crate) fn cr0(&self) -> u64 {
        self.control_regs.cr0().guest_value()
    }

    pub(crate) fn cr2(&self) -> u64 {
        self.sregs.cr2
    }

    pub(crate) fn cr3(&self) -> u64 {
        self.sregs.cr3
    }

    pub(crate) fn cr4(&self) -> u64 {
        self.control_regs.cr4().guest_value()
    }

    pub(crate) fn control_regs(&self) -> VcpuControlRegisters {
        self.control_regs
    }

    pub(crate) fn set_control_regs_from_vmcs(&mut self, control_regs: VcpuControlRegisters) {
        self.control_regs = control_regs;
        self.sregs.cr0 = self.control_regs.cr0().guest_value();
        self.sregs.cr4 = self.control_regs.cr4().guest_value();
    }

    pub(crate) fn set_cr2(&mut self, value: u64) {
        self.sregs.cr2 = value;
    }

    pub(crate) fn set_cr3(&mut self, value: u64) {
        self.sregs.cr3 = value;
    }

    pub(crate) fn set_cs(&mut self, segment: VcpuSegment) {
        self.sregs.cs = segment;
    }

    pub(crate) fn set_ds(&mut self, segment: VcpuSegment) {
        self.sregs.ds = segment;
    }

    pub(crate) fn set_es(&mut self, segment: VcpuSegment) {
        self.sregs.es = segment;
    }

    pub(crate) fn set_fs(&mut self, segment: VcpuSegment) {
        self.sregs.fs = segment;
        self.msrs.fs_base = segment.base;
    }

    pub(crate) fn set_gs(&mut self, segment: VcpuSegment) {
        self.sregs.gs = segment;
        self.msrs.gs_base = segment.base;
    }

    pub(crate) fn set_ss(&mut self, segment: VcpuSegment) {
        self.sregs.ss = segment;
    }

    pub(crate) fn set_tr(&mut self, segment: VcpuSegment) {
        self.sregs.tr = segment;
    }

    pub(crate) fn set_ldt(&mut self, segment: VcpuSegment) {
        self.sregs.ldt = segment;
    }

    pub(crate) fn set_gdt(&mut self, table: VcpuDtable) {
        self.sregs.gdt = table;
    }

    pub(crate) fn set_idt(&mut self, table: VcpuDtable) {
        self.sregs.idt = table;
    }

    pub(crate) fn set_fs_base(&mut self, value: u64) {
        self.sregs.fs.base = value;
        self.msrs.fs_base = value;
    }

    pub(crate) fn set_gs_base(&mut self, value: u64) {
        self.sregs.gs.base = value;
        self.msrs.gs_base = value;
    }

    pub(crate) fn set_efer(&mut self, value: u64) {
        self.msrs.efer = value;
        self.sregs.efer = self.msrs.efer;
    }

    pub(crate) fn load_fpu(&mut self) {
        self.fpu.load();
    }

    pub(crate) fn save_fpu(&mut self) {
        self.fpu.save();
    }

    fn vmcs_guest_state(&self) -> VmcsGuestState {
        VmcsGuestState {
            regs: self.regs,
            sregs: self.sregs,
            control_regs: self.control_regs,
            msrs: self.msrs,
        }
    }
}

impl VcpuSregs {
    fn with_startup(startup_vector: u8) -> Self {
        let code_base = u64::from(startup_vector) << 12;
        let code_selector = u16::from(startup_vector) << 8;
        let data = VcpuSegment::real_mode_data_segment(0, 0);

        Self {
            cs: VcpuSegment::real_mode_code_segment(code_selector, code_base),
            ds: data,
            es: data,
            fs: data,
            gs: data,
            ss: data,
            tr: VcpuSegment::real_mode_system_segment(0x20, 0, 0x0b),
            ldt: VcpuSegment {
                unusable: 1,
                ..VcpuSegment::default()
            },
            idt: VcpuDtable {
                base: 0,
                limit: 0x03ff,
                padding: [0; 3],
            },
            cr0: (Cr0Flags::EXTENSION_TYPE | Cr0Flags::NUMERIC_ERROR).bits(),
            ..VcpuSregs::default()
        }
    }
}

impl VcpuSegment {
    fn real_mode_code_segment(selector: u16, base: u64) -> Self {
        Self::real_mode_segment(selector, base, 0x0b, 1)
    }

    fn real_mode_data_segment(selector: u16, base: u64) -> Self {
        Self::real_mode_segment(selector, base, 0x03, 1)
    }

    fn real_mode_system_segment(selector: u16, base: u64, type_: u8) -> Self {
        Self::real_mode_segment(selector, base, type_, 0)
    }

    fn real_mode_segment(selector: u16, base: u64, type_: u8, s: u8) -> Self {
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
}
