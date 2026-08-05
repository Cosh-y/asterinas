// SPDX-License-Identifier: MPL-2.0

use super::{
    context::{GuestContext, VcpuRunState},
    exit::{GuestExitInfo, vmexit_handler},
    host_context::{HostContext, init_host_contexts, with_host_context},
    interrupt::*,
    types::{GuestInterrupt, GuestTimerInstant},
    vmcs_state::{
        apply_dirty_vmcs_state, load_guest_run_msrs, save_guest_context, synchronize_vmcs_state,
    },
    vmx::{
        Msr, VmcsControl32, VmcsControl64, VmcsGuest32, VmcsGuest64, VmcsGuestNW, VmcsReadOnly32,
        VmxExitInfo, VmxExitReason, VmxGuard, acquire_vmx, exit_info, vcpu_run,
    },
    x86::write_cr2_raw,
};
use crate::{
    Error,
    prelude::*,
    vm::{GuestInterruptPort, GuestPhysMemSpace, GuestTimerPort},
};

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
///     let guest_mode = GuestMode::new()?;
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
    HostInterrupt,
    /// The vCPU is waiting for a startup IPI and was not entered.
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

    /// Runs the guest with the supplied context and guest physical memory space.
    ///
    /// The `interrupt_port` and `timer_port` arguments provide kernel policy for
    /// pending guest interrupts and guest timers while the vCPU is running.
    ///
    /// The method returns when guest execution needs handling by the kernel
    /// client. [`GuestRunResult::VmExit`] carries the architecture-specific
    /// VM-exit information.
    ///
    /// After handling the returned event and updating `context` or the guest
    /// device model as needed, call this method again to resume guest execution.
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
        if context.run_state().waits_for_startup() {
            return Ok(GuestRunResult::WaitForSipi);
        }

        // Keep VMCS operations and guest execution on one pCPU.
        // VMCS may remain active on this pCPU after the run returns.
        let preempt_guard = crate::task::disable_preempt();
        self.enter_run(context, guest_mem)?;
        let run_result = with_host_context(&preempt_guard, |host_context| {
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

    /// Materializes all VMCS-backed guest state into the safe context cache.
    ///
    /// The caller must ensure that the guest vCPU is not running. This method
    /// makes the VMCS current on a pinned pCPU before accessing its fields.
    pub fn synchronize_state(&self, context: &mut GuestContext) -> Result<()> {
        if !context.vmcs.initialized() {
            return Ok(());
        }

        let _preempt_guard = crate::task::disable_preempt();
        context.vmcs.activate_for_access()?;
        apply_dirty_vmcs_state(context)?;
        synchronize_vmcs_state(context)
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
            let host_interrupt = exit_info.exit_reason == VmxExitReason::EXTERNAL_INTERRUPT as u32;

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
            save_guest_context(context, &exit_info)?;
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

        let Some(vector) = interrupt_port.query_pending_interrupt() else {
            return Ok(None);
        };
        let vector = vector.vector;

        if vector < 32 {
            return Ok(None);
        }

        let intr_info = u32::from(vector) | INTR_INFO_VALID_MASK | INTR_TYPE_EXT_INTR;
        if !vmx_interrupt_injectable()? {
            enable_interrupt_window_exiting()?;
            return Ok(None);
        }
        disable_interrupt_window_exiting()?;

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
        write_cr2_raw(context.arch().cr2());
        load_guest_run_msrs(context)?;
        context.arch_mut().load_fpu();
        apply_dirty_vmcs_state(context)
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
