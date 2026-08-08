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
use crate::{
    Error,
    arch::vm::{
        GuestContext, GuestExitInfo, GuestInterrupt, GuestTimerInstant, VcpuDtable, VcpuSegment,
        X86GprIndex,
        context::VcpuRunState,
        control_regs::{VcpuControlRegister, VcpuControlRegisters},
        interrupt::resume_from_halted,
        vmx::{
            Msr, VmcsControl32, VmcsControl64, VmcsControlNW, VmcsGuest16, VmcsGuest32,
            VmcsGuest64, VmcsGuestNW, VmcsReadOnly32, VmxExitInfo, VmxExitReason, VmxGuard,
            acquire_vmx, exit_info,
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
/// # fn handle_vm_exit(guest_result: ostd::vm::GuestRunResult) {}
/// #
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
///     let mut guest_mode = GuestMode::new()?;
///
///     loop {
///         let run_result = guest_mode.execute(context, guest_mem, interrupt_port, timer_port)?;
///         // Handle VM exit according to the exit reason recorded in `run_result`.
///         handle_vm_exit(run_result);
///     }
/// }
/// ```
pub struct GuestMode {
    _vmx_guard: VmxGuard,
}

/// Describes why a guest run returned to the kernel client.
pub enum GuestRunResult {
    /// The vCPU exited and needs higher-level handling.
    VmExit(GuestExitInfo),
    /// A host interrupt ended this run so the kernel can reach a scheduling point.
    #[cfg(target_arch = "x86_64")]
    HostInterrupt,
    /// The vCPU is waiting for a startup IPI and was not entered (x86 only).
    #[cfg(target_arch = "x86_64")]
    WaitForSipi,
}

impl GuestMode {
    /// Creates a guest execution object.
    ///
    /// Creating this value does not enter the guest; use [`Self::execute`] to
    /// run the vCPU with a guest context and kernel-provided policy ports.
    ///
    /// # Errors
    ///
    /// Returns an error if the platform does not support virtualization.
    pub fn new() -> Result<Self> {
        let vmx_guard = acquire_vmx()?;
        init_host_contexts();
        Ok(Self {
            _vmx_guard: vmx_guard,
        })
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
    /// VM-exit information.
    ///
    /// After handling the returned event and updating `context` or the guest
    /// device model as needed, call this method again to resume guest
    /// execution.
    pub fn execute<I, T>(
        &self,
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

        // Keep VMCS operations and guest execution on one pCPU.
        // VMCS may remain active on this pCPU after the run returns.
        let _preempt_guard = crate::task::disable_preempt();
        self.enter_run(context, guest_mem)?;
        let run_result = with_host_context(&_preempt_guard, |host_context| {
            // The same host task remains pinned throughout this execution. VM exits
            // restore this per-CPU snapshot before running kernel handlers, so
            // internal exits can reuse it instead of repeating XSAVE and RDMSR.
            host_context.save();
            context
                .vmcs
                .write_host_run_msrs(host_context.run_msr_values())?;
            self.run_loop(context, interrupt_port, timer_port, host_context)
        });
        self.leave_run(context);
        run_result
    }

    fn run_loop<I, T>(
        &self,
        context: &mut GuestContext,
        interrupt_port: &I,
        timer_port: &T,
        host_context: &HostContext,
    ) -> Result<GuestRunResult>
    where
        I: GuestInterruptPort + ?Sized,
        T: GuestTimerPort + ?Sized,
    {
        loop {
            let irq_guard = crate::irq::disable_local();

            self.prepare_vmentry(context, interrupt_port, timer_port, host_context)?;
            let run_result = self.vmlaunch_or_vmresume(context);
            let exit_info = self.complete_vmexit(context, run_result, host_context)?;
            let host_interrupt =
                exit_info.exit_reason == VmxExitReason::EXTERNAL_INTERRUPT as u32;

            use crate::arch::vm::exit::vmexit_handler;
            let exit_info = vmexit_handler(context, &exit_info)?;
            drop(irq_guard);

            // Guest execution is pinned with preemption disabled while its VMCS is
            // current. Return after a physical interrupt so the kernel client can
            // release that guard and give competing host tasks a scheduling point.
            if host_interrupt {
                return Ok(GuestRunResult::HostInterrupt);
            }

            // Deliver VM-exit handling to the kernel client or userspace.
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
        if context.run_state() == VcpuRunState::Running {
            context.set_run_state(VcpuRunState::Runnable);
        }
    }

    fn init_vmcs(&self, context: &mut GuestContext, guest_mem: &GuestPhysMemSpace) -> Result<()> {
        let was_initialized = context.vmcs.initialized();
        let vmcs_guest_state = context.vmcs_guest_state();
        context.vmcs.load(vmcs_guest_state, guest_mem.eptp())?;
        // Initial VMCS setup consumes the complete cached state. On later
        // runs `Vmcs::load` only activates the existing VMCS, so mutations are
        // applied by `load_guest_context` below.
        if !was_initialized {
            context.clear_vmcs_dirty();
        }
        Ok(())
    }

    fn prepare_vmentry<I, T>(
        &self,
        context: &mut GuestContext,
        interrupt_port: &I,
        timer_port: &T,
        host_context: &HostContext,
    ) -> Result<()>
    where
        I: GuestInterruptPort + ?Sized,
        T: GuestTimerPort + ?Sized,
    {
        self.prepare_preemption_timer(context, timer_port)?;
        self.prepare_interrupt(interrupt_port)?;
        if let Err(err) = self.load_guest_context(context) {
            host_context.load();
            return Err(err);
        }
        Ok(())
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
        run_result: Result<()>,
        host_context: &HostContext,
    ) -> Result<VmxExitInfo> {
        if let Err(err) = run_result {
            // VM entry did not complete, so the VM-exit MSR-load list cannot
            // be relied upon. Restore the complete host snapshot manually.
            host_context.load();
            return Err(err);
        }

        let complete_result = (|| {
            let exit_info = exit_info()?;
            self.save_guest_context(context, &exit_info)?;
            Ok(exit_info)
        })();
        host_context.load_after_vmexit();
        complete_result
    }

    fn prepare_interrupt<I: GuestInterruptPort + ?Sized>(
        &self,
        interrupt_port: &I,
    ) -> Result<Option<u8>> {
        VmcsControl32::VMENTRY_INTERRUPTION_INFO_FIELD.write(0)?;

        use crate::arch::vm::interrupt::*;
        if interrupt_port.query_pending_nmi() {
            if !vmx_nmi_injectable()? {
                enable_nmi_window_exiting()?;
                return Ok(None);
            }
            disable_nmi_window_exiting()?;

            const NMI_VECTOR: u8 = 2;
            let intr_info = u32::from(NMI_VECTOR) | INTR_INFO_VALID_MASK | INTR_TYPE_NMI;
            VmcsControl32::VMENTRY_INTERRUPTION_INFO_FIELD.write(intr_info)?;
            interrupt_port.accept_nmi();
            return Ok(Some(NMI_VECTOR));
        }

        let pending_vector = interrupt_port.query_pending_interrupt();

        let Some(vector) = pending_vector else {
            return Ok(None);
        };

        let vector = vector.vector;

        // why?
        if vector < 32 {
            return Ok(None);
        }

        let intr_info = u32::from(vector) | INTR_INFO_VALID_MASK | INTR_TYPE_EXT_INTR;
        let injectable = vmx_interrupt_injectable()?;

        if !injectable {
            enable_interrupt_window_exiting()?;
            return Ok(None);
        }
        disable_interrupt_window_exiting()?;

        // inject interrupt through VMCS
        VmcsControl32::VMENTRY_INTERRUPTION_INFO_FIELD.write(intr_info)?;
        interrupt_port.accept_interrupt(GuestInterrupt { vector });
        Ok(Some(vector))
    }

    fn prepare_preemption_timer<T: GuestTimerPort + ?Sized>(
        &self,
        context: &GuestContext,
        timer_port: &T,
    ) -> Result<()> {
        let guest_tsc = context.guest_tsc();
        let timer_deadline = timer_port.poll_deadline(GuestTimerInstant { tsc: guest_tsc });
        let timer_value =
            vmx_preemption_timer_value(guest_tsc, timer_deadline.map(|deadline| deadline.tsc));
        VmcsGuest32::VMX_PREEMPTION_TIMER_VALUE.write(timer_value)?;
        VmcsControl64::TSC_OFFSET.write(context.tsc_offset as u64)?;
        Ok(())
    }

    fn load_guest_context(&self, context: &mut GuestContext) -> Result<()> {
        let cr2 = context.arch().cr2();
        write_cr2_raw(cr2);
        self.load_guest_run_msrs(context)?;
        context.arch_mut().load_fpu();

        self.apply_dirty_vmcs_state(context)?;
        Ok(())
    }

    fn apply_dirty_vmcs_state(&self, context: &mut GuestContext) -> Result<()> {
        let dirty = context.vmcs_dirty();
        if dirty.rip {
            VmcsGuestNW::RIP.write(context.arch().rip() as usize)?;
        }
        if dirty.rsp {
            VmcsGuestNW::RSP.write(context.arch().gpr(X86GprIndex::Rsp) as usize)?;
        }
        if dirty.rflags {
            // Architectural bit 1 is fixed to one.
            VmcsGuestNW::RFLAGS.write((context.arch().rflags() | 0x2) as usize)?;
        }
        if dirty.sregs {
            write_sregs_to_vmcs(context)?;
        } else {
            if dirty.efer {
                write_guest_efer_to_vmcs(context)?;
            }
            if dirty.pat {
                VmcsGuest64::IA32_PAT.write(context.arch().msr(IA32_PAT))?;
            }
            if dirty.fs_base {
                VmcsGuestNW::FS_BASE.write(context.arch().msr(IA32_FS_BASE) as usize)?;
            }
            if dirty.gs_base {
                VmcsGuestNW::GS_BASE.write(context.arch().msr(IA32_GS_BASE) as usize)?;
            }
            if dirty.sysenter_cs {
                VmcsGuest32::IA32_SYSENTER_CS.write(context.arch().msr(IA32_SYSENTER_CS) as u32)?;
            }
            if dirty.sysenter_esp {
                VmcsGuestNW::IA32_SYSENTER_ESP
                    .write(context.arch().msr(IA32_SYSENTER_ESP) as usize)?;
            }
            if dirty.sysenter_eip {
                VmcsGuestNW::IA32_SYSENTER_EIP
                    .write(context.arch().msr(IA32_SYSENTER_EIP) as usize)?;
            }
            if dirty.run_msrs {
                self.load_guest_run_msrs(context)?;
            }
        }
        context.clear_vmcs_dirty();
        Ok(())
    }

    fn save_guest_context(
        &self,
        context: &mut GuestContext,
        exit_info: &VmxExitInfo,
    ) -> Result<()> {
        context.arch_mut().save_fpu();
        self.save_guest_run_msrs(context)?;
        use x86_64::registers::control::Cr2;
        context.arch_mut().set_cr2(Cr2::read_raw());

        // VM-exit information already contains RIP, so do not issue a second
        // VMREAD for the hottest field.
        context.arch_mut().set_rip(exit_info.guest_rip);

        match VmxExitReason::try_from(exit_info.exit_reason) {
            Ok(VmxExitReason::CR_ACCESS) => save_full_sregs_from_vmcs(context)?,
            Ok(VmxExitReason::EPT_VIOLATION) => {
                save_address_translation_state_from_vmcs(context, true)?;
                context
                    .arch_mut()
                    .set_gpr(X86GprIndex::Rsp, 8, VmcsGuestNW::RSP.read()? as u64);
            }
            Ok(VmxExitReason::IO_INSTRUCTION) => {
                let is_input = exit_info.exit_qualification & (1 << 3) != 0;
                let is_string = exit_info.exit_qualification & (1 << 4) != 0;
                let is_dword = exit_info.exit_qualification & 0b111 == 3;
                if is_string {
                    save_pio_string_state_from_vmcs(context)?;
                } else if is_input && is_dword {
                    save_address_translation_state_from_vmcs(context, true)?;
                }
            }
            Ok(VmxExitReason::MSR_WRITE)
                if context.arch().gpr(X86GprIndex::Rcx) as u32 == IA32_EFER =>
            {
                save_full_sregs_from_vmcs(context)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Materializes VMCS-backed general registers into the safe guest cache.
    ///
    /// This is intentionally a narrow OSTD operation: callers request a
    /// coherent architectural view but never gain access to VMREAD/VMWRITE.
    pub fn synchronize_regs(&self, context: &mut GuestContext) -> Result<()> {
        if !context.vmcs.initialized() {
            return Ok(());
        }
        let _preempt_guard = crate::task::disable_preempt();
        context.vmcs.activate_for_access()?;
        self.apply_dirty_vmcs_state(context)?;
        context.arch_mut().set_rip(VmcsGuestNW::RIP.read()?);
        context
            .arch_mut()
            .set_gpr(X86GprIndex::Rsp, 8, VmcsGuestNW::RSP.read()? as u64);
        context
            .arch_mut()
            .set_rflags(VmcsGuestNW::RFLAGS.read()? as u64);
        Ok(())
    }

    /// Materializes all VMCS-backed special registers into the guest cache.
    pub fn synchronize_sregs(&self, context: &mut GuestContext) -> Result<()> {
        if !context.vmcs.initialized() {
            return Ok(());
        }
        let _preempt_guard = crate::task::disable_preempt();
        context.vmcs.activate_for_access()?;
        self.apply_dirty_vmcs_state(context)?;
        save_full_sregs_from_vmcs(context)
    }

    /// Materializes MSRs that execute directly in guest mode.
    pub fn synchronize_msrs(&self, context: &mut GuestContext) -> Result<()> {
        if !context.vmcs.initialized() {
            return Ok(());
        }
        let _preempt_guard = crate::task::disable_preempt();
        context.vmcs.activate_for_access()?;
        self.apply_dirty_vmcs_state(context)?;
        self.save_guest_run_msrs(context)?;
        context.arch_mut().set_msr(
            IA32_SYSENTER_CS,
            u64::from(VmcsGuest32::IA32_SYSENTER_CS.read()?),
        );
        context.arch_mut().set_msr(
            IA32_SYSENTER_ESP,
            VmcsGuestNW::IA32_SYSENTER_ESP.read()? as u64,
        );
        context.arch_mut().set_msr(
            IA32_SYSENTER_EIP,
            VmcsGuestNW::IA32_SYSENTER_EIP.read()? as u64,
        );
        Ok(())
    }

    // TODO: understand this two functions.
    fn load_guest_run_msrs(&self, context: &GuestContext) -> Result<()> {
        context.vmcs.write_guest_run_msrs([
            context.arch().msr(IA32_STAR),
            context.arch().msr(IA32_LSTAR),
            context.arch().msr(IA32_CSTAR),
            context.arch().msr(IA32_FMASK),
            context.arch().msr(IA32_KERNEL_GSBASE),
        ])
    }

    fn save_guest_run_msrs(&self, context: &mut GuestContext) -> Result<()> {
        let [star, lstar, cstar, syscall_mask, kernel_gs_base] =
            context.vmcs.read_guest_run_msrs()?;
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

fn vmx_preemption_timer_value(current_tsc: u64, deadline_tsc: Option<u64>) -> u32 {
    let rate = (Msr::IA32_VMX_MISC.read() & 0x1f) as u32;
    vmx_preemption_timer_value_at_rate(current_tsc, deadline_tsc, rate)
}

fn vmx_preemption_timer_value_at_rate(
    current_tsc: u64,
    deadline_tsc: Option<u64>,
    rate: u32,
) -> u32 {
    let Some(deadline_tsc) = deadline_tsc else {
        // Keep the control enabled and soft-disable the timer. Host external
        // interrupts still cause VM exits, and this value is reloaded before
        // every entry, so the timer cannot become a hidden polling source.
        return u32::MAX;
    };

    let tsc_cycles = deadline_tsc.saturating_sub(current_tsc);
    if tsc_cycles == 0 {
        return 0;
    }

    let rounding = (1_u64 << rate).saturating_sub(1);
    let ticks = tsc_cycles.saturating_add(rounding) >> rate;
    u32::try_from(ticks).unwrap_or(u32::MAX).max(1)
}

#[cfg(ktest)]
mod tests {
    use super::vmx_preemption_timer_value_at_rate;
    use crate::prelude::*;

    #[ktest]
    fn preemption_timer_soft_disables_without_deadline() {
        assert_eq!(vmx_preemption_timer_value_at_rate(100, None, 5), u32::MAX);
    }

    #[ktest]
    fn preemption_timer_expires_immediately_for_elapsed_deadline() {
        assert_eq!(vmx_preemption_timer_value_at_rate(100, Some(100), 5), 0);
        assert_eq!(vmx_preemption_timer_value_at_rate(101, Some(100), 5), 0);
    }

    #[ktest]
    fn preemption_timer_rounds_up_to_hardware_ticks() {
        assert_eq!(vmx_preemption_timer_value_at_rate(0, Some(1), 5), 1);
        assert_eq!(vmx_preemption_timer_value_at_rate(0, Some(32), 5), 1);
        assert_eq!(vmx_preemption_timer_value_at_rate(0, Some(33), 5), 2);
    }

    #[ktest]
    fn preemption_timer_saturates_long_deadlines() {
        assert_eq!(
            vmx_preemption_timer_value_at_rate(0, Some(u64::MAX), 0),
            u32::MAX
        );
        assert_eq!(
            vmx_preemption_timer_value_at_rate(0, Some(u64::MAX), 31),
            u32::MAX
        );
    }
}

fn read_dtable_from_vmcs(base_field: VmcsGuestNW, limit_field: VmcsGuest32) -> Result<VcpuDtable> {
    Ok(VcpuDtable {
        base: base_field.read()? as u64,
        limit: limit_field.read()? as u16,
        padding: [0; 3],
    })
}

fn write_sregs_to_vmcs(context: &GuestContext) -> Result<()> {
    let arch = context.arch();
    let sregs = arch.sregs();

    write_control_registers_to_vmcs(arch.control_regs())?;
    VmcsGuestNW::CR3.write(arch.cr3() as usize)?;

    use x86::msr::*;
    write_guest_efer_to_vmcs(context)?;

    VmcsGuest64::IA32_PAT.write(arch.msr(IA32_PAT))?;
    VmcsGuest32::IA32_SYSENTER_CS.write(arch.msr(IA32_SYSENTER_CS) as u32)?;
    VmcsGuestNW::IA32_SYSENTER_ESP.write(arch.msr(IA32_SYSENTER_ESP) as usize)?;
    VmcsGuestNW::IA32_SYSENTER_EIP.write(arch.msr(IA32_SYSENTER_EIP) as usize)?;

    write_segment_to_vmcs(
        &sregs.cs,
        VmcsGuest16::CS_SELECTOR,
        VmcsGuestNW::CS_BASE,
        VmcsGuest32::CS_LIMIT,
        VmcsGuest32::CS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.ds,
        VmcsGuest16::DS_SELECTOR,
        VmcsGuestNW::DS_BASE,
        VmcsGuest32::DS_LIMIT,
        VmcsGuest32::DS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.es,
        VmcsGuest16::ES_SELECTOR,
        VmcsGuestNW::ES_BASE,
        VmcsGuest32::ES_LIMIT,
        VmcsGuest32::ES_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.fs,
        VmcsGuest16::FS_SELECTOR,
        VmcsGuestNW::FS_BASE,
        VmcsGuest32::FS_LIMIT,
        VmcsGuest32::FS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.gs,
        VmcsGuest16::GS_SELECTOR,
        VmcsGuestNW::GS_BASE,
        VmcsGuest32::GS_LIMIT,
        VmcsGuest32::GS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.ss,
        VmcsGuest16::SS_SELECTOR,
        VmcsGuestNW::SS_BASE,
        VmcsGuest32::SS_LIMIT,
        VmcsGuest32::SS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.tr,
        VmcsGuest16::TR_SELECTOR,
        VmcsGuestNW::TR_BASE,
        VmcsGuest32::TR_LIMIT,
        VmcsGuest32::TR_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.ldt,
        VmcsGuest16::LDTR_SELECTOR,
        VmcsGuestNW::LDTR_BASE,
        VmcsGuest32::LDTR_LIMIT,
        VmcsGuest32::LDTR_ACCESS_RIGHTS,
    )?;
    VmcsGuestNW::GDTR_BASE.write(sregs.gdt.base as usize)?;
    VmcsGuest32::GDTR_LIMIT.write(u32::from(sregs.gdt.limit))?;
    VmcsGuestNW::IDTR_BASE.write(sregs.idt.base as usize)?;
    VmcsGuest32::IDTR_LIMIT.write(u32::from(sregs.idt.limit))?;
    Ok(())
}

fn write_guest_efer_to_vmcs(context: &GuestContext) -> Result<()> {
    use x86::vmx::vmcs::control::EntryControls;
    use x86_64::registers::model_specific::EferFlags;

    let guest_efer = context.arch().msr(IA32_EFER);
    VmcsGuest64::IA32_EFER.write(guest_efer)?;
    let mut entry = VmcsControl32::VMENTRY_CONTROLS.read()?;
    if guest_efer & EferFlags::LONG_MODE_ACTIVE.bits() != 0 {
        entry |= EntryControls::IA32E_MODE_GUEST.bits();
    } else {
        entry &= !EntryControls::IA32E_MODE_GUEST.bits();
    }
    VmcsControl32::VMENTRY_CONTROLS.write(entry)
}

fn write_segment_to_vmcs(
    segment: &VcpuSegment,
    selector_field: VmcsGuest16,
    base_field: VmcsGuestNW,
    limit_field: VmcsGuest32,
    rights_field: VmcsGuest32,
) -> Result<()> {
    selector_field.write(segment.selector)?;
    base_field.write(segment.base as usize)?;
    limit_field.write(segment.limit)?;
    rights_field.write(segment_access_rights(segment))
}

fn segment_access_rights(segment: &VcpuSegment) -> u32 {
    let mut rights = u32::from(segment.type_ & 0x0f);
    rights |= u32::from(segment.s & 0x1) << 4;
    rights |= u32::from(segment.dpl & 0x3) << 5;
    rights |= u32::from(segment.present & 0x1) << 7;
    rights |= u32::from(segment.avl & 0x1) << 12;
    rights |= u32::from(segment.l & 0x1) << 13;
    rights |= u32::from(segment.db & 0x1) << 14;
    rights |= u32::from(segment.g & 0x1) << 15;
    rights |= u32::from(segment.unusable & 0x1) << 16;
    rights
}

fn save_address_translation_state_from_vmcs(
    context: &mut GuestContext,
    save_cs: bool,
) -> Result<()> {
    let guest_cr3 = VmcsGuestNW::CR3.read()?;
    context.arch_mut().set_cr3(guest_cr3 as u64);
    context
        .arch_mut()
        .set_control_regs_from_vmcs(read_control_registers_from_vmcs()?);
    let guest_efer = VmcsGuest64::IA32_EFER.read()?;
    context.arch_mut().set_msr(IA32_EFER, guest_efer);
    if save_cs {
        context.arch_mut().set_cs(read_segment_from_vmcs(
            VmcsGuest16::CS_SELECTOR,
            VmcsGuestNW::CS_BASE,
            VmcsGuest32::CS_LIMIT,
            VmcsGuest32::CS_ACCESS_RIGHTS,
        )?);
    }
    Ok(())
}

fn save_pio_string_state_from_vmcs(context: &mut GuestContext) -> Result<()> {
    save_address_translation_state_from_vmcs(context, true)?;
    context
        .arch_mut()
        .set_rflags(VmcsGuestNW::RFLAGS.read()? as u64);
    save_data_segments_from_vmcs(context)
}

fn save_data_segments_from_vmcs(context: &mut GuestContext) -> Result<()> {
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
    Ok(())
}

fn save_full_sregs_from_vmcs(context: &mut GuestContext) -> Result<()> {
    save_address_translation_state_from_vmcs(context, true)?;
    save_data_segments_from_vmcs(context)?;
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
    context.arch_mut().set_gdt(read_dtable_from_vmcs(
        VmcsGuestNW::GDTR_BASE,
        VmcsGuest32::GDTR_LIMIT,
    )?);
    context.arch_mut().set_idt(read_dtable_from_vmcs(
        VmcsGuestNW::IDTR_BASE,
        VmcsGuest32::IDTR_LIMIT,
    )?);
    Ok(())
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
