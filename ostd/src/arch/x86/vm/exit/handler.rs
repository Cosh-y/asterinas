use crate::{
    Error,
    arch::vm::{
        VcpuRunState,
        context::GuestContext,
        exit::GuestExitInfo,
        vmx::{
            VmcsGuest16, VmcsGuest32, VmcsGuest64, VmcsGuestNW, VmcsReadOnly32, VmxExitInfo,
            VmxExitReason,
        },
    },
    prelude::*,
};

const PAUSE_INSN_LENGTH: u64 = 2;

pub fn vmexit_handler(
    context: &mut GuestContext,
    exit_info: &VmxExitInfo,
) -> Result<Option<GuestExitInfo>> {
    if exit_info.entry_failure {
        log_vmentry_guest_state(exit_info);
        return Err(Error::InvalidArgs);
    }

    match VmxExitReason::try_from(exit_info.exit_reason) {
        Ok(VmxExitReason::EXTERNAL_INTERRUPT) => {
            // Hypervisor intentionally leaves "acknowledge interrupt on exit"
            // disabled in the VMCS. That keeps the host interrupt pending across the
            // VM-exit, so once the IRQ-disable guard around the VM-exit critical
            // section is released, Asterinas can receive and process the interrupt via
            // its normal trap/IRQ path without any explicit handoff from Hypervisor.
            Ok(None)
        }
        Ok(VmxExitReason::INTERRUPT_WINDOW) => {
            handle_interrupt_window()?;
            Ok(None)
        }
        Ok(VmxExitReason::NMI_WINDOW) => {
            super::super::interrupt::disable_nmi_window_exiting()?;
            Ok(None)
        }
        Ok(VmxExitReason::CPUID) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Ok(VmxExitReason::CR_ACCESS) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Ok(VmxExitReason::MSR_READ) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Ok(VmxExitReason::MSR_WRITE) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Ok(VmxExitReason::PAUSE_INSTRUCTION) => {
            context.advance_rip(PAUSE_INSN_LENGTH);
            Ok(Some(GuestExitInfo::from(*exit_info)))
        }
        Ok(VmxExitReason::HLT) => {
            context.set_run_state(VcpuRunState::Halted);
            Ok(Some(GuestExitInfo::from(*exit_info)))
        }
        Ok(VmxExitReason::PREEMPTION_TIMER) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Ok(VmxExitReason::IO_INSTRUCTION) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Ok(VmxExitReason::TRIPLE_FAULT) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Ok(VmxExitReason::VMCALL) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Ok(VmxExitReason::EPT_VIOLATION) => {
            // Guest access to APIC or UART through MMIO.
            // APIC: emulate in kernel client.
            // UART: emulate in userspace client.
            Ok(Some(GuestExitInfo::from(*exit_info)))
        }
        Ok(_) => Ok(Some(GuestExitInfo::from(*exit_info))),
        Err(_) => Ok(Some(GuestExitInfo::from(*exit_info))),
    }
}

/// Handle a VM-exit caused by "interrupt-window exiting".
///
/// "interrupt-window exiting" means vcpu is ready to accept interrupts.
/// Disable "interrupt-window exiting" here.
/// Inject pending interrupts before the next VM-entry.
fn handle_interrupt_window() -> Result<()> {
    super::super::interrupt::disable_interrupt_window_exiting()
}

fn log_vmentry_guest_state(exit_info: &VmxExitInfo) {
    let vm_instruction_error = VmcsReadOnly32::VM_INSTRUCTION_ERROR.read().ok();
    let guest_rsp = VmcsGuestNW::RSP.read().ok();
    let guest_rflags = VmcsGuestNW::RFLAGS.read().ok();
    let guest_cr0 = VmcsGuestNW::CR0.read().ok();
    let guest_cr3 = VmcsGuestNW::CR3.read().ok();
    let guest_cr4 = VmcsGuestNW::CR4.read().ok();
    let guest_efer = VmcsGuest64::IA32_EFER.read().ok();
    let cs_selector = VmcsGuest16::CS_SELECTOR.read().ok();
    let ss_selector = VmcsGuest16::SS_SELECTOR.read().ok();
    let tr_selector = VmcsGuest16::TR_SELECTOR.read().ok();
    let ldtr_selector = VmcsGuest16::LDTR_SELECTOR.read().ok();
    let cs_ar = VmcsGuest32::CS_ACCESS_RIGHTS.read().ok();
    let ss_ar = VmcsGuest32::SS_ACCESS_RIGHTS.read().ok();
    let tr_ar = VmcsGuest32::TR_ACCESS_RIGHTS.read().ok();
    let ldtr_ar = VmcsGuest32::LDTR_ACCESS_RIGHTS.read().ok();
    let exit_reason_name = VmxExitReason::try_from(exit_info.exit_reason).ok();

    error!(
        "hypervisor: VM-entry failure: exit_reason={:#x} ({:?}), vm_instruction_error={:?}",
        exit_info.exit_reason, exit_reason_name, vm_instruction_error
    );
    error!(
        "hypervisor: entry rip={:#x}, rsp={:?}, rflags={:?}, qualification={:#x}",
        exit_info.guest_rip, guest_rsp, guest_rflags, exit_info.exit_qualification
    );
    error!(
        "hypervisor: control cr0={:?}, cr3={:?}, cr4={:?}, efer={:?}",
        guest_cr0, guest_cr3, guest_cr4, guest_efer
    );
    error!(
        "hypervisor: segments cs={:?}/{:?}, ss={:?}/{:?}, tr={:?}/{:?}, ldtr={:?}/{:?}",
        cs_selector, cs_ar, ss_selector, ss_ar, tr_selector, tr_ar, ldtr_selector, ldtr_ar
    );
}
