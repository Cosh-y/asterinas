//! Guest virtualization support.

mod gpm_space;
mod host_context;
mod interrupt;
mod timer;

use host_context::*;
use x86::msr::*;

pub use self::{
    gpm_space::GuestPhysMemSpace, interrupt::GuestInterruptPort, timer::GuestTimerPort,
};
pub use crate::arch::vm::vmx::{VmxGuard, acquire_vmx};
use crate::{
    Error,
    arch::vm::{
        GuestContext, GuestExitInfo, VcpuDtable, X86GprIndex, VcpuSegment,
        context::VcpuRunState,
        control_regs::{VcpuControlRegister, VcpuControlRegisters},
        interrupt::resume_from_halted,
        vmx::{
            Msr, VmcsControl32, VmcsControl64, VmcsControlNW, VmcsGuest16, VmcsGuest32,
            VmcsGuest64, VmcsGuestNW, VmcsReadOnly32, exit_info,
        },
        x86::write_cr2_raw,
    },
    prelude::*,
};

/*
/// Initializes guest virtualization support on this platform.
pub fn init() -> Result<()> {
    crate::arch::vm::vmx::init_vmx()
}
*/

/// Runs guest vCPU code in an isolated guest execution mode.
///
/// `GuestMode` is the OSTD-side execution object for a guest vCPU. It enters
/// guest execution with the vCPU context and kernel-provided policy ports
/// supplied to [`Self::execute`] until a VM exit must be handled outside OSTD.
///
/// Here is a sample code on how to use `GuestMode`.
///
/// ```no_run
/// use ostd::{
///     arch::vm::GuestContext,
///     prelude::*,
///     vm::{GuestInterruptPort, GuestMode, GuestPhysMemSpace, GuestTimerPort},
/// };
///
/// fn run_guest(
///     context: &mut GuestContext,
///     interrupt_port: &dyn GuestInterruptPort,
///     timer_port: &dyn GuestTimerPort,
///     guest_mem: &GuestPhysMemSpace,
/// ) -> Result<()> {
///     let mut guest_mode = GuestMode::new();
///
///     loop {
///         let _run_result = guest_mode.execute(context, guest_mem, interrupt_port, timer_port)?;
///         todo!("handle the userspace-visible VM exit");
///     }
/// }
/// ```
pub struct GuestMode;

/// Describes why a guest run returned to the kernel client.
pub enum GuestRunResult {
    /// The vCPU exited VMX non-root mode and needs higher-level handling.
    VmExit(GuestExitInfo),
    /// The vCPU is waiting for a startup IPI and was not entered.
    WaitForSipi,
}

impl GuestMode {
    /// Creates a guest execution object.
    ///
    /// Creating this value does not enter the guest; use [`Self::execute`] to
    /// run the vCPU with a guest context and kernel-provided policy ports.
    pub fn new() -> Self {
        GuestMode
    }

    /// Runs the guest with the supplied
    /// guest context and guest physical memory space.
    ///
    /// The `interrupt_port` and `timer_port` arguments provide kernel
    /// policy for pending guest interrupts and guest timers
    /// while the vCPU is running.
    ///
    /// The method returns when guest execution needs handling by the kernel
    /// client. [`GuestRunResult::VmExit`] carries the architecture-specific
    /// exit information for policy or device emulation, and
    /// [`GuestRunResult::WaitForSipi`] indicates that the vCPU is waiting for
    /// its startup signal and was not entered.
    ///
    /// After handling the returned event and updating `context` or the guest
    /// device model as needed, call this method again to resume guest
    /// execution.
    pub fn execute<I, T>(
        &mut self,
        context: &mut GuestContext,
        guest_mem: &GuestPhysMemSpace,
        interrupt_port: &I,
        timer_port: &T,
    ) -> Result<GuestRunResult>
    where
        I: GuestInterruptPort + ?Sized,
        T: GuestTimerPort + ?Sized,
    {
        if context.run_state() == VcpuRunState::WaitForSipi {
            return Ok(GuestRunResult::WaitForSipi);
        }

        // VMCS state is per-pCPU while loaded. Keep this run on one pCPU, then
        // clear the VMCS before returning so the next RSH_RUN may migrate safely.
        let _preempt_guard = crate::task::disable_preempt();
        self.enter_run(context, guest_mem)?;
        let run_result = self.run_loop(context, interrupt_port, timer_port);
        self.leave_run(context);
        run_result
    }

    fn run_loop<I, T>(
        &mut self,
        context: &mut GuestContext,
        interrupt_port: &I,
        timer_port: &T,
    ) -> Result<GuestRunResult>
    where
        I: GuestInterruptPort + ?Sized,
        T: GuestTimerPort + ?Sized,
    {
        loop {
            let irq_guard = crate::irq::disable_local();

            let host_context = self.prepare_vmentry(context, interrupt_port, timer_port)?;
            let run_result = self.vmlaunch_or_vmresume(context);
            self.complete_vmexit(context, host_context, run_result)?;

            use crate::arch::vm::exit::vmexit_handler;
            let exit_info = exit_info()?;
            let exit_info = vmexit_handler(context, &exit_info)?;
            drop(irq_guard);

            // Deliver handling of vmexit to kernel client or userspace.
            if let Some(exit_info) = exit_info {
                return Ok(GuestRunResult::VmExit(exit_info));
            }
        }
    }

    fn enter_run(&self, context: &mut GuestContext, guest_mem: &GuestPhysMemSpace) -> Result<()> {
        self.init_vmcs(context, guest_mem)?;

        if context.run_state() == VcpuRunState::Halted {
            resume_from_halted()?;
            context.set_run_state(VcpuRunState::Runnable);
        }

        if context.run_state() == VcpuRunState::Runnable {
            context.set_run_state(VcpuRunState::Running);
        } else {
            error!("unexpected run state.");
        }
        Ok(())
    }

    fn leave_run(&self, context: &mut GuestContext) {
        if let Err(err) = context.vmcs.quit() {
            error!("errno: {:?}", err);
            error!("unexpect condition: failed to quit vmcs")
        }
        if context.run_state() == VcpuRunState::Running {
            context.set_run_state(VcpuRunState::Runnable);
        }
    }

    fn init_vmcs(&self, context: &mut GuestContext, guest_mem: &GuestPhysMemSpace) -> Result<()> {
        context.vmcs.load()?;
        if context.vmcs.initialized() {
            return Ok(());
        }

        let vmcs_guest_state = context.vmcs_guest_state();
        context.vmcs.init(vmcs_guest_state, guest_mem.eptp())?;

        Ok(())
    }

    fn prepare_vmentry<I, T>(
        &self,
        context: &mut GuestContext,
        interrupt_port: &I,
        timer_port: &T,
    ) -> Result<HostContext>
    where
        I: GuestInterruptPort + ?Sized,
        T: GuestTimerPort + ?Sized,
    {
        self.prepare_preemption_timer(context, timer_port)?;
        self.prepare_interrupt(interrupt_port)?;
        let host_context = HostContext::save();
        if let Err(err) = self.load_guest_context(context) {
            host_context.load();
            return Err(err);
        }
        Ok(host_context)
    }

    fn vmlaunch_or_vmresume(&self, context: &mut GuestContext) -> Result<()> {
        let launched: u64 = if context.vmcs.launched() { 1 } else { 0 };

        use crate::arch::vm::vmx::vcpu_run;
        let ret = vcpu_run(context.arch_mut().regs_mut_ptr(), launched);
        if ret != 0 {
            log_vcpu_run_failure(launched);
            return Err(Error::InvalidArgs);
        }

        context.vmcs.set_launched(true);
        Ok(())
    }

    fn complete_vmexit(
        &self,
        context: &mut GuestContext,
        host_context: HostContext,
        run_result: Result<()>,
    ) -> Result<()> {
        let save_guest_context_result = self.save_guest_context(context);
        host_context.load();

        run_result?;
        save_guest_context_result?;

        Ok(())
    }

    fn prepare_interrupt<I: GuestInterruptPort + ?Sized>(
        &self,
        interrupt_port: &I,
    ) -> Result<Option<u8>> {
        VmcsControl32::VMENTRY_INTERRUPTION_INFO_FIELD.write(0)?;

        let pending_vector = interrupt_port.check_pending_interrupt();

        let Some(vector) = pending_vector else {
            return Ok(None);
        };

        // why?
        if vector < 32 {
            return Ok(None);
        }

        use crate::arch::vm::interrupt::*;
        let intr_info = u32::from(vector) | INTR_INFO_VALID_MASK | INTR_TYPE_EXT_INTR;
        let injectable = vmx_interrupt_injectable()?;

        if !injectable {
            enable_interrupt_window_exiting()?;
            return Ok(None);
        }
        disable_interrupt_window_exiting()?;

        // inject interrupt through VMCS
        VmcsControl32::VMENTRY_INTERRUPTION_INFO_FIELD.write(intr_info)?;
        interrupt_port.accept_interrupt(vector);
        Ok(Some(vector))
    }

    fn prepare_preemption_timer<T: GuestTimerPort + ?Sized>(
        &self,
        context: &GuestContext,
        timer_port: &T,
    ) -> Result<()> {
        let guest_tsc = context.guest_tsc();
        let timer_deadline = timer_port.check_deadline(guest_tsc);
        let gap = timer_deadline
            .map(|deadline| deadline.saturating_sub(guest_tsc).max(1))
            .unwrap_or(500_000);
        let timer_value = vmx_preemption_timer_ticks(gap);
        VmcsGuest32::VMX_PREEMPTION_TIMER_VALUE.write(timer_value)?;
        VmcsControl64::TSC_OFFSET.write(context.tsc_offset as u64)?;
        Ok(())
    }

    fn load_guest_context(&self, context: &mut GuestContext) -> Result<()> {
        let cr2 = context.arch().cr2();
        write_cr2_raw(cr2);
        self.load_guest_run_msrs(context);
        context.arch_mut().load_fpu();

        VmcsGuestNW::RIP.write(context.arch().rip() as usize)?;
        VmcsGuestNW::RSP.write(context.arch().gpr(X86GprIndex::Rsp) as usize)?;
        // TODO: why | 0x2 ?
        VmcsGuestNW::RFLAGS.write((context.arch().rflags() | 0x2) as usize)?;

        write_control_registers_to_vmcs(context.arch().control_regs())?;

        use x86::{msr::*, vmx::vmcs::control::EntryControls};
        use x86_64::registers::model_specific::EferFlags;
        let guest_efer = context.arch().msr(IA32_EFER);
        VmcsGuest64::IA32_EFER.write(guest_efer)?;
        let mut entry = VmcsControl32::VMENTRY_CONTROLS.read()?;
        if guest_efer & EferFlags::LONG_MODE_ACTIVE.bits() != 0 {
            entry |= EntryControls::IA32E_MODE_GUEST.bits();
        } else {
            entry &= !EntryControls::IA32E_MODE_GUEST.bits();
        }
        VmcsControl32::VMENTRY_CONTROLS.write(entry)?;

        let guest_cr3 = context.arch().cr3();
        VmcsGuestNW::CR3.write(guest_cr3 as usize)?;

        VmcsGuest64::IA32_PAT.write(context.arch().msr(IA32_PAT))?;
        VmcsGuestNW::FS_BASE.write(context.arch().msr(IA32_FS_BASE) as usize)?;
        VmcsGuestNW::GS_BASE.write(context.arch().msr(IA32_GS_BASE) as usize)?;
        VmcsGuest32::IA32_SYSENTER_CS.write(context.arch().msr(IA32_SYSENTER_CS) as u32)?;
        VmcsGuestNW::IA32_SYSENTER_ESP.write(context.arch().msr(IA32_SYSENTER_ESP) as usize)?;
        VmcsGuestNW::IA32_SYSENTER_EIP.write(context.arch().msr(IA32_SYSENTER_EIP) as usize)?;

        Ok(())
    }

    fn save_guest_context(&self, context: &mut GuestContext) -> Result<()> {
        context.arch_mut().save_fpu();
        self.save_guest_run_msrs(context)?;
        use x86_64::registers::control::Cr2;
        context.arch_mut().set_cr2(Cr2::read_raw());

        context.arch_mut().set_rip(VmcsGuestNW::RIP.read()? as u64);
        context
            .arch_mut()
            .set_gpr(X86GprIndex::Rsp, 8, VmcsGuestNW::RSP.read()? as u64);
        context
            .arch_mut()
            .set_rflags(VmcsGuestNW::RFLAGS.read()? as u64);

        let guest_cr3 = VmcsGuestNW::CR3.read()?;
        context.arch_mut().set_cr3(guest_cr3 as u64);

        context
            .arch_mut()
            .set_control_regs_from_vmcs(read_control_registers_from_vmcs()?);

        let guest_efer = VmcsGuest64::IA32_EFER.read()?;
        context.arch_mut().set_msr(IA32_EFER, guest_efer);

        context.arch_mut().set_gdt(read_dtable_from_vmcs(
            VmcsGuestNW::GDTR_BASE,
            VmcsGuest32::GDTR_LIMIT,
        )?);
        context.arch_mut().set_idt(read_dtable_from_vmcs(
            VmcsGuestNW::IDTR_BASE,
            VmcsGuest32::IDTR_LIMIT,
        )?);

        context.arch_mut().set_cs(read_segment_from_vmcs(
            VmcsGuest16::CS_SELECTOR,
            VmcsGuestNW::CS_BASE,
            VmcsGuest32::CS_LIMIT,
            VmcsGuest32::CS_ACCESS_RIGHTS,
        )?);
        context.arch_mut().set_ds(read_segment_from_vmcs(
            VmcsGuest16::DS_SELECTOR,
            VmcsGuestNW::DS_BASE,
            VmcsGuest32::DS_LIMIT,
            VmcsGuest32::DS_ACCESS_RIGHTS,
        )?);
        context.arch_mut().set_es(read_segment_from_vmcs(
            VmcsGuest16::ES_SELECTOR,
            VmcsGuestNW::ES_BASE,
            VmcsGuest32::ES_LIMIT,
            VmcsGuest32::ES_ACCESS_RIGHTS,
        )?);
        context.arch_mut().set_fs(read_segment_from_vmcs(
            VmcsGuest16::FS_SELECTOR,
            VmcsGuestNW::FS_BASE,
            VmcsGuest32::FS_LIMIT,
            VmcsGuest32::FS_ACCESS_RIGHTS,
        )?);
        context.arch_mut().set_gs(read_segment_from_vmcs(
            VmcsGuest16::GS_SELECTOR,
            VmcsGuestNW::GS_BASE,
            VmcsGuest32::GS_LIMIT,
            VmcsGuest32::GS_ACCESS_RIGHTS,
        )?);
        context.arch_mut().set_ss(read_segment_from_vmcs(
            VmcsGuest16::SS_SELECTOR,
            VmcsGuestNW::SS_BASE,
            VmcsGuest32::SS_LIMIT,
            VmcsGuest32::SS_ACCESS_RIGHTS,
        )?);
        context.arch_mut().set_tr(read_segment_from_vmcs(
            VmcsGuest16::TR_SELECTOR,
            VmcsGuestNW::TR_BASE,
            VmcsGuest32::TR_LIMIT,
            VmcsGuest32::TR_ACCESS_RIGHTS,
        )?);
        context.arch_mut().set_ldt(read_segment_from_vmcs(
            VmcsGuest16::LDTR_SELECTOR,
            VmcsGuestNW::LDTR_BASE,
            VmcsGuest32::LDTR_LIMIT,
            VmcsGuest32::LDTR_ACCESS_RIGHTS,
        )?);

        Ok(())
    }

    // TODO: understand this two functions.
    fn load_guest_run_msrs(&self, context: &GuestContext) {
        Msr::IA32_STAR.write(context.arch().msr(IA32_STAR));
        Msr::IA32_LSTAR.write(context.arch().msr(IA32_LSTAR));
        Msr::IA32_CSTAR.write(context.arch().msr(IA32_CSTAR));
        Msr::IA32_FMASK.write(context.arch().msr(IA32_FMASK));
        Msr::IA32_KERNEL_GSBASE.write(context.arch().msr(IA32_KERNEL_GSBASE));
    }

    fn save_guest_run_msrs(&self, context: &mut GuestContext) -> Result<()> {
        let star = Msr::IA32_STAR.read();
        let lstar = Msr::IA32_LSTAR.read();
        let cstar = Msr::IA32_CSTAR.read();
        let syscall_mask = Msr::IA32_FMASK.read();
        let kernel_gs_base = Msr::IA32_KERNEL_GSBASE.read();
        let fs_base = VmcsGuestNW::FS_BASE.read()? as u64;
        let gs_base = VmcsGuestNW::GS_BASE.read()? as u64;

        context.arch_mut().set_msr(IA32_STAR, star);
        context.arch_mut().set_msr(IA32_LSTAR, lstar);
        context.arch_mut().set_msr(IA32_CSTAR, cstar);
        context.arch_mut().set_msr(IA32_FMASK, syscall_mask);
        context
            .arch_mut()
            .set_msr(IA32_KERNEL_GSBASE, kernel_gs_base);
        context.arch_mut().set_msr(IA32_FS_BASE, fs_base);
        context.arch_mut().set_msr(IA32_GS_BASE, gs_base);
        Ok(())
    }
}

fn vmx_preemption_timer_ticks(tsc_cycles: u64) -> u32 {
    let rate = (Msr::IA32_VMX_MISC.read() & 0x1f) as u32;
    let rounding = (1_u64 << rate).saturating_sub(1);
    (tsc_cycles.saturating_add(rounding) >> rate) as u32
}

fn read_dtable_from_vmcs(base_field: VmcsGuestNW, limit_field: VmcsGuest32) -> Result<VcpuDtable> {
    Ok(VcpuDtable {
        base: base_field.read()? as u64,
        limit: limit_field.read()? as u16,
        padding: [0; 3],
    })
}

fn write_control_registers_to_vmcs(control_regs: VcpuControlRegisters) -> Result<()> {
    write_control_register_to_vmcs(
        control_regs.cr0(),
        VmcsGuestNW::CR0,
        VmcsControlNW::CR0_GUEST_HOST_MASK,
        VmcsControlNW::CR0_READ_SHADOW,
    )?;
    write_control_register_to_vmcs(
        control_regs.cr4(),
        VmcsGuestNW::CR4,
        VmcsControlNW::CR4_GUEST_HOST_MASK,
        VmcsControlNW::CR4_READ_SHADOW,
    )
}

fn write_control_register_to_vmcs(
    reg: VcpuControlRegister,
    real_field: VmcsGuestNW,
    mask_field: VmcsControlNW,
    shadow_field: VmcsControlNW,
) -> Result<()> {
    real_field.write(reg.real() as usize)?;
    mask_field.write(reg.host_mask() as usize)?;
    shadow_field.write(reg.read_shadow() as usize)
}

fn read_control_registers_from_vmcs() -> Result<VcpuControlRegisters> {
    let cr0 = read_control_register_state_from_vmcs(
        VmcsGuestNW::CR0,
        VmcsControlNW::CR0_GUEST_HOST_MASK,
        VmcsControlNW::CR0_READ_SHADOW,
    )?;
    let cr4 = read_control_register_state_from_vmcs(
        VmcsGuestNW::CR4,
        VmcsControlNW::CR4_GUEST_HOST_MASK,
        VmcsControlNW::CR4_READ_SHADOW,
    )?;
    Ok(VcpuControlRegisters::from_vmcs(cr0, cr4))
}

fn read_control_register_state_from_vmcs(
    value_field: VmcsGuestNW,
    mask_field: VmcsControlNW,
    shadow_field: VmcsControlNW,
) -> Result<VcpuControlRegister> {
    let real = value_field.read()? as u64;
    let mask = mask_field.read()? as u64;
    let shadow = shadow_field.read()? as u64;
    Ok(VcpuControlRegister::from_vmcs(mask, shadow, real))
}

fn read_segment_from_vmcs(
    selector_field: VmcsGuest16,
    base_field: VmcsGuestNW,
    limit_field: VmcsGuest32,
    rights_field: VmcsGuest32,
) -> Result<VcpuSegment> {
    let rights = rights_field.read()?;
    Ok(VcpuSegment {
        base: base_field.read()? as u64,
        limit: limit_field.read()?,
        selector: selector_field.read()?,
        type_: (rights & 0x0f) as u8,
        present: ((rights >> 7) & 0x1) as u8,
        dpl: ((rights >> 5) & 0x3) as u8,
        db: ((rights >> 14) & 0x1) as u8,
        s: ((rights >> 4) & 0x1) as u8,
        l: ((rights >> 13) & 0x1) as u8,
        g: ((rights >> 15) & 0x1) as u8,
        avl: ((rights >> 12) & 0x1) as u8,
        unusable: ((rights >> 16) & 0x1) as u8,
        padding: 0,
    })
}

fn log_vcpu_run_failure(launched: u64) {
    error!(
        "hypervisor: vcpu_run failed, launched={} vm_instruction_error={:?} \
             guest_rip={:?} guest_rsp={:?} guest_rflags={:?} guest_cr0={:?} \
             guest_cr3={:?} guest_cr4={:?} guest_efer={:?} pin_ctls={:?} \
             primary_ctls={:?} secondary_ctls={:?} exit_ctls={:?} entry_ctls={:?} \
             eptp={:?}",
        launched,
        VmcsReadOnly32::VM_INSTRUCTION_ERROR.read().ok(),
        VmcsGuestNW::RIP.read().ok(),
        VmcsGuestNW::RSP.read().ok(),
        VmcsGuestNW::RFLAGS.read().ok(),
        VmcsGuestNW::CR0.read().ok(),
        VmcsGuestNW::CR3.read().ok(),
        VmcsGuestNW::CR4.read().ok(),
        VmcsGuest64::IA32_EFER.read().ok(),
        VmcsControl32::PINBASED_EXEC_CONTROLS.read().ok(),
        VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS.read().ok(),
        VmcsControl32::SECONDARY_PROCBASED_EXEC_CONTROLS.read().ok(),
        VmcsControl32::VMEXIT_CONTROLS.read().ok(),
        VmcsControl32::VMENTRY_CONTROLS.read().ok(),
        VmcsControl64::EPTP.read().ok(),
    );
}
