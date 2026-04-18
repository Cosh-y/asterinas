//! Two main functions
//! 1. handle external interrupts like keyboard interrupt
//! 2. inject virtual interrupts to VM through VMCS VM Entry Control

use ostd::arch::virt::*;

use super::error::*;
use crate::emulate::apic::{lapic_kick_to_service, Lapic};

const EVENT_TYPE_HWEXC: u32 = 3;
const EVENT_TYPE_EXTINT: u32 = 0;

const INTR_INFO_DELIVER_CODE_MASK: u32 = 0x800;
const INTR_INFO_VALID_MASK: u32 = 0x8000_0000;
const INTR_INFO_VECTOR_MASK: u32 = 0xFF;

const INTR_TYPE_HARD_EXCEPTION: u32 = EVENT_TYPE_HWEXC << 8;
const INTR_TYPE_EXT_INTR: u32 = EVENT_TYPE_EXTINT << 8;

/// x86 general-protection fault vector.
const X86_TRAP_GP: u32 = 13;

/// Guest RFLAGS.IF bit.
const RFLAGS_IF: usize = 1 << 9;

/// Interruptibility-state blocking bits (STI / MOV-SS blocking).
const BLOCKING_BY_STI: u32 = 1 << 0;
const BLOCKING_BY_MOV_SS: u32 = 1 << 1;

/// Tracks a pending hardware exception that needs to be injected into the guest.
#[derive(Debug, Default)]
pub struct ExceptionState {
    pub pending: bool,
    pub injected: bool,
    pub has_error_code: bool,
    pub nr: u32,
    pub error_code: u32,
}

/// Tracks a pending virtual interrupt (for interrupt-window injection).
#[derive(Debug, Default)]
pub struct InterruptState {
    pub pending: bool,
    pub intr_info: u32,
}

/// Handle a VM-exit caused by an external interrupt.
///
/// Reads the VM-exit interruption-information field from the VMCS and treats
/// it as a notification that guest execution was preempted by a host external
/// interrupt.
pub fn handle_external_interrupt() -> Result<()> {
    let exit_intr_info = VmcsReadOnly32::VMEXIT_INTERRUPTION_INFO
        .read()
        .map_err(Error::from)?;

    if (exit_intr_info & INTR_INFO_VALID_MASK) == 0 {
        return Ok(());
    }

    let vector = exit_intr_info & INTR_INFO_VECTOR_MASK;

    // RustShyper intentionally leaves "acknowledge interrupt on exit"
    // disabled in the VMCS. That keeps the host interrupt pending across the
    // VM-exit, so once the IRQ-disable guard around the VM-exit critical
    // section is released, Asterinas can receive and process the interrupt via
    // its normal trap/IRQ path without any explicit handoff from RustShyper.
    let _ = vector;

    Ok(())
}

/// Return `true` if there is a pending exception that has not yet been
/// injected into the guest VMCS.
///
/// Mirrors `has_pending_exception` from `injection.c`.
pub fn has_pending_exception(exc: &ExceptionState) -> bool {
    exc.pending && !exc.injected
}

/// Queue a hardware exception for delivery on the next VM-entry.
///
/// Mirrors `inject_queue_exception` from `injection.c`.
pub fn inject_queue_exception(exc: &mut ExceptionState, nr: u32, error_code: u32) {
    if has_pending_exception(exc) {
        log::warn!(
            "VCPU: Has pending exception #{} when injecting exception #{}",
            exc.nr,
            nr
        );
    }
    exc.pending = true;
    exc.nr = nr;
    exc.has_error_code = true;
    exc.error_code = error_code;
    exc.injected = false;
}

/// Write the queued exception into the VMCS VM-entry interruption-information
/// field so that it is delivered to the guest on the next VM-entry.
///
/// Mirrors `inject_pending_exception` from `injection.c`.
pub fn inject_pending_exception(
    exc: &mut ExceptionState,
    perf_exception_count: &mut [u32; 32],
) -> Result<()> {
    let mut intr_info = exc.nr | INTR_INFO_VALID_MASK | INTR_TYPE_HARD_EXCEPTION;

    if exc.has_error_code {
        VmcsControl32::VMENTRY_EXCEPTION_ERR_CODE
            .write(exc.error_code)
            .map_err(Error::from)?;
        intr_info |= INTR_INFO_DELIVER_CODE_MASK;
    }

    VmcsControl32::VMENTRY_INTERRUPTION_INFO_FIELD
        .write(intr_info)
        .map_err(Error::from)?;

    if exc.nr < 32 {
        perf_exception_count[exc.nr as usize] =
            perf_exception_count[exc.nr as usize].saturating_add(1);
    }

    exc.pending = false;
    exc.injected = true;

    Ok(())
}

/// Inject a virtual interrupt (vector >= 32) into the guest.
///
/// If the guest is not currently ready to receive an interrupt (i.e.
/// `IF` is clear or blocking-by-STI/MOV-SS is active), the injection is
/// deferred and interrupt-window exiting is enabled so that the
/// hypervisor re-enters and injects when the guest opens its window.
///
/// Mirrors `inject_interrupt` from `injection.c`.
pub fn inject_interrupt(
    lapic: &mut Lapic,
    intr: &mut InterruptState,
    perf_interrupt_count: &mut [u32; 224],
    vector: u32,
) -> Result<()> {
    if vector < 32 {
        // Only external interrupts (>= 32) are handled here.
        return Ok(());
    }

    let intr_info = vector | INTR_INFO_VALID_MASK | INTR_TYPE_EXT_INTR;

    perf_interrupt_count[(vector - 32) as usize] =
        perf_interrupt_count[(vector - 32) as usize].saturating_add(1);

    if vmx_interrupt_injectable()? {
        VmcsControl32::VMENTRY_INTERRUPTION_INFO_FIELD
            .write(intr_info)
            .map_err(Error::from)?;
        lapic_kick_to_service(lapic, vector as u8);
    } else {
        // Defer until the guest opens an interrupt window.
        intr.intr_info = intr_info;
        intr.pending = true;
        enable_interrupt_window_exiting()?;
    }

    Ok(())
}

/// Handle a VM-exit caused by "interrupt-window exiting".
///
/// Disables the interrupt-window exiting control bit and, if a deferred
/// interrupt is pending, injects it now.
///
/// Mirrors `handle_interrupt_window` from `injection.c`.
pub fn handle_interrupt_window(lapic: &mut Lapic, intr: &mut InterruptState) -> Result<()> {
    disable_interrupt_window_exiting()?;
    try_inject_pending_interrupt(lapic, intr)
}

/// Queue a general-protection fault (#GP, vector 13) with error code 0.
///
/// Mirrors `inject_gp_fault` from `injection.c`.
pub fn inject_gp_fault(exc: &mut ExceptionState) {
    inject_queue_exception(exc, X86_TRAP_GP, 0);
}

/// Check whether the guest is currently in a state where an external interrupt
/// can be injected (RFLAGS.IF == 1 and no blocking-by-STI/MOV-SS).
///
/// Mirrors `vmx_interrupt_injectable` from the C VMX backend.
fn vmx_interrupt_injectable() -> Result<bool> {
    let rflags = VmcsGuestNW::RFLAGS.read().map_err(Error::from)?;
    let interruptibility = VmcsGuest32::INTERRUPTIBILITY_STATE
        .read()
        .map_err(Error::from)?;

    let if_set = (rflags & RFLAGS_IF) != 0;
    let not_blocking = (interruptibility & (BLOCKING_BY_STI | BLOCKING_BY_MOV_SS)) == 0;

    Ok(if_set && not_blocking)
}

/// If a deferred interrupt is pending and the guest is now interruptible,
/// write it into the VM-entry interruption-information field.
pub fn try_inject_pending_interrupt(lapic: &mut Lapic, intr: &mut InterruptState) -> Result<()> {
    if !intr.pending {
        return Ok(());
    }

    if !vmx_interrupt_injectable()? {
        enable_interrupt_window_exiting()?;
        return Ok(());
    }

    intr.pending = false;
    disable_interrupt_window_exiting()?;

    VmcsControl32::VMENTRY_INTERRUPTION_INFO_FIELD
        .write(intr.intr_info)
        .map_err(Error::from)?;
    lapic_kick_to_service(lapic, (intr.intr_info & INTR_INFO_VECTOR_MASK) as u8);
    Ok(())
}

/// Enable interrupt-window exiting in the primary processor-based controls.
fn enable_interrupt_window_exiting() -> Result<()> {
    let cur = VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS
        .read()
        .map_err(Error::from)?;
    VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS
        .write(cur | (1 << 2))
        .map_err(Error::from)?;
    Ok(())
}

/// Disable interrupt-window exiting in the primary processor-based controls.
fn disable_interrupt_window_exiting() -> Result<()> {
    let cur = VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS
        .read()
        .map_err(Error::from)?;
    VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS
        .write(cur & !(1 << 2))
        .map_err(Error::from)?;
    Ok(())
}
