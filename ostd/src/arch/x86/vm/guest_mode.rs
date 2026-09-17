// SPDX-License-Identifier: MPL-2.0

use x86::{
    msr::{IA32_KERNEL_GSBASE, IA32_VMX_MISC, rdmsr, wrmsr},
    vmx::vmcs::{control, guest},
};

use super::{
    GuestContext, GuestExitInfo, GuestInterrupt, GuestTimerInstant, VmxExitReason, exit,
    vmx::{VmxGuard, vmcs::Vmcs},
};
use crate::{
    Error,
    arch::{read_tsc, trap::syscall},
    irq,
    prelude::*,
    task,
    user::UserModeHooks,
    vm::{GuestInterruptSource, GuestPhysMemSpace, GuestTimer},
};

/// An execution mode for isolated guests.
///
/// [`Self::execute`] runs a guest vCPU until a VM exit or kernel event needs handling outside OSTD.
///
/// # Examples
///
/// ```no_run
/// # fn handle_vm_exit(reason: ostd::arch::vm::GuestReturnReason) {}
/// #
/// use ostd::{
///     arch::vm::{GuestContext, GuestMode},
///     prelude::*,
///     user::UserModeHooks,
///     vm::{GuestInterruptSource, GuestPhysMemSpace, GuestTimer},
/// };
///
/// fn run_guest(
///     context: &mut GuestContext,
///     interrupt_source: &dyn GuestInterruptSource,
///     timer: &dyn GuestTimer,
///     guest_mem: &GuestPhysMemSpace,
///     hooks: &impl UserModeHooks,
/// ) -> Result<()> {
///     let guest_mode = GuestMode::new()?;
///
///     loop {
///         let return_reason = guest_mode.execute(context, guest_mem, interrupt_source, timer, hooks)?;
///         // Handle VM exit according to the exit reason recorded in `return_reason`.
///         handle_vm_exit(return_reason);
///     }
/// }
/// ```
pub struct GuestMode {
    vmx_guard: VmxGuard,
}

/// The reason guest execution returned to the kernel client.
#[derive(Debug)]
pub enum GuestReturnReason {
    /// An exit that requires higher-level handling.
    VmExit(GuestExitInfo),
    /// A pending kernel event reported by [`UserModeHooks::has_kernel_event`].
    KernelEvent,
}

impl GuestMode {
    /// Creates a guest execution object without entering a guest.
    pub fn new() -> Result<Self> {
        let vmx_guard = VmxGuard::acquire_vmx()?;
        Ok(Self { vmx_guard })
    }

    /// Runs the guest until a VM exit or kernel event needs handling by the kernel client.
    ///
    /// The `interrupt_source` supplies pending guest interrupts,
    /// and `timer` supplies guest timer deadlines.
    ///
    /// The caller can use [`UserModeHooks::pre_user_run`] to prepare guest state
    /// that does not affect soundness, such as FPU state. The hook runs with local
    /// IRQs disabled before each guest entry.
    ///
    /// After handling a VM exit internally, [`UserModeHooks::has_kernel_event`] determines
    /// whether execution returns with [`GuestReturnReason::KernelEvent`].
    ///
    /// # Panics
    ///
    /// Must be called in task context with local IRQs and preemption enabled.
    #[track_caller]
    pub fn execute<
        I: GuestInterruptSource + ?Sized,
        T: GuestTimer + ?Sized,
        H: UserModeHooks + ?Sized,
    >(
        &self,
        context: &mut GuestContext,
        guest_mem: &GuestPhysMemSpace,
        interrupt_source: &I,
        timer: &T,
        hooks: &H,
    ) -> Result<GuestReturnReason> {
        task::atomic_mode::might_sleep();
        let (arch, vmcs) = context.arch_and_vmcs();
        let mut needs_controls_sync = true;

        loop {
            task::scheduler::might_preempt();

            let preempt_guard = task::disable_preempt();
            needs_controls_sync |= vmcs.load(&self.vmx_guard)?;
            let irq_guard = irq::disable_local();
            if needs_controls_sync {
                // SAFETY:
                // 1. `load()` made this VMCS current.
                // 2. The borrowed `guest_mem` ensures EPT isolation and frame lifetimes.
                unsafe { vmcs.sync_controls(guest_mem.eptp(), arch.sregs.efer, &irq_guard) }?;
                needs_controls_sync = false;
            }

            // SAFETY: The same VMCS remains current under these guards.
            unsafe { vmcs.load_guest_context(arch, &self.vmx_guard, &irq_guard) }?;
            // SAFETY: The same VMCS remains current under these guards.
            let injected = unsafe { prepare_events(vmcs, interrupt_source, timer, &irq_guard) }?;

            hooks.pre_user_run(&irq_guard);
            // SAFETY: The VMCS loaded above remains current while preemption is
            // disabled and the `context` and VMX guard are held.
            unsafe { vmcs.sync_host(&irq_guard) }?;
            let host_kernel_gs_base = unsafe { rdmsr(IA32_KERNEL_GSBASE) };
            // SAFETY: Host MSRs are restored on every path below,
            // before the IRQ and preemption guards can be dropped.
            let result = unsafe { arch.load_run_state(&irq_guard) };
            // SAFETY:
            // 1. The VMCS remains current, `sync_controls` enforces EPT isolation,
            //    and `load()` and `sync_host` install the host state.
            // 2. The `context` and `guest_mem` keep the VMCS and EPT alive;
            // 3. The code below restores host MSRs not managed by VMX before
            //    IRQs or preemption can resume, including on entry failure.
            let result = result.and_then(|()| unsafe { vmcs.run(&mut arch.regs, &irq_guard) });
            // SAFETY: VM exit leaves this VMCS current, and the guards remain held.
            let result = result.and_then(|()| unsafe { exit::exit_info(vmcs, &irq_guard) });
            if result.is_ok() {
                arch.save_run_state(&irq_guard);
            }
            syscall::configure_msrs(&irq_guard);
            // SAFETY: The saved MSR value belongs to the current task on this CPU.
            unsafe { wrmsr(IA32_KERNEL_GSBASE, host_kernel_gs_base) };
            let exit = result?;
            // SAFETY: The VMCS remains current under the held guards.
            unsafe { vmcs.save_guest_context(arch, &irq_guard) }?;
            if let Some(interrupt) = injected {
                interrupt_source.accept_interrupt(interrupt);
            }
            // `ACK_INTERRUPT_ON_EXIT` is clear, so enabling IRQs delivers
            // pending host interrupts through the normal IRQ path.
            drop(irq_guard);
            drop(preempt_guard);

            match VmxExitReason::try_from(exit.exit_reason) {
                Ok(VmxExitReason::EXTERNAL_INTERRUPT | VmxExitReason::INTERRUPT_WINDOW) => {}
                _ => return Ok(GuestReturnReason::VmExit(exit)),
            }

            if hooks.has_kernel_event() {
                return Ok(GuestReturnReason::KernelEvent);
            }
        }
    }
}

/// Prepares the guest timer and pending interrupt for VM entry.
///
/// # Safety
///
/// The VMCS must be current on this CPU.
unsafe fn prepare_events<I: GuestInterruptSource + ?Sized, T: GuestTimer + ?Sized>(
    vmcs: &Vmcs,
    interrupt_source: &I,
    timer: &T,
    irq_guard: &irq::DisabledLocalIrqGuard,
) -> Result<Option<GuestInterrupt>> {
    // SAFETY:
    // 1. The caller ensures this VMCS is current.
    // 2. The timer only bounds guest execution and does not change host state.
    unsafe {
        vmcs.write(
            guest::VMX_PREEMPTION_TIMER_VALUE,
            preemption_timer(timer, irq_guard) as usize,
            irq_guard,
        )
    }?;

    // Recompute injection and window exiting on every entry, including after
    // an interrupt-window exit or an earlier failed VM entry.
    // SAFETY:
    // 1. The caller ensures this VMCS is current.
    // 2. These writes do not affect host state.
    let primary = unsafe {
        vmcs.write(control::VMENTRY_INTERRUPTION_INFO_FIELD, 0, irq_guard)?;
        let primary = vmcs.read(control::PRIMARY_PROCBASED_EXEC_CONTROLS, irq_guard)?
            & !(control::PrimaryControls::INTERRUPT_WINDOW_EXITING.bits() as usize);
        vmcs.write(control::PRIMARY_PROCBASED_EXEC_CONTROLS, primary, irq_guard)?;
        primary
    };

    let Some(interrupt) = interrupt_source.query_pending_interrupt() else {
        return Ok(None);
    };
    if interrupt.vector < 32 {
        return Err(Error::InvalidArgs);
    }
    // SAFETY: The caller ensures this VMCS is current.
    let (rflags, blocking) = unsafe {
        (
            vmcs.read(guest::RFLAGS, irq_guard)?,
            vmcs.read(guest::INTERRUPTIBILITY_STATE, irq_guard)?,
        )
    };
    // Intel SDM, Vol. 3C, Section 24.4.2: STI and MOV-SS block external interrupts.
    if rflags & (1 << 9) == 0 || blocking & 3 != 0 {
        // SAFETY:
        // 1. The caller ensures this VMCS is current.
        // 2. These writes do not affect host state.
        unsafe {
            vmcs.write(
                control::PRIMARY_PROCBASED_EXEC_CONTROLS,
                primary | control::PrimaryControls::INTERRUPT_WINDOW_EXITING.bits() as usize,
                irq_guard,
            )
        }?;
        return Ok(None);
    }
    // SAFETY:
    // 1. The caller ensures this VMCS is current.
    // 2. These writes do not affect host state.
    unsafe {
        vmcs.write(
            control::VMENTRY_INTERRUPTION_INFO_FIELD,
            (1 << 31) | usize::from(interrupt.vector),
            irq_guard,
        )
    }?;
    Ok(Some(interrupt))
}

/// Converts the guest deadline to a VMX preemption-timer count.
///
/// # Safety
///
/// This CPU must support VMX.
unsafe fn preemption_timer<T: GuestTimer + ?Sized>(
    timer: &T,
    _irq_guard: &irq::DisabledLocalIrqGuard,
) -> u32 {
    let now = read_tsc();
    let Some(deadline) = timer.poll_deadline(GuestTimerInstant { tsc: now }) else {
        // Keep the timer enabled with its longest interval. This is finite,
        // so expiration may still return a timer exit to the client.
        return u32::MAX;
    };
    // SAFETY: The caller guarantees VMX support, so `IA32_VMX_MISC` exists.
    let rate = unsafe { rdmsr(IA32_VMX_MISC) } & 0x1f;
    let cycles = deadline.tsc.saturating_sub(now);
    u32::try_from(cycles.saturating_add((1 << rate) - 1) >> rate).unwrap_or(u32::MAX)
}
