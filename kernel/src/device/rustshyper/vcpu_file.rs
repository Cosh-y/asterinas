// SPDX-License-Identifier: MPL-2.0

//! VCPU file descriptor implementation

use ostd::{
    arch::vm::{GuestContext, GuestExitInfo, VcpuRegs, VcpuSregs, VmxExitReason},
    vm::{GuestInterruptPort, GuestMode, GuestTimerPort},
};

use super::{
    apic::{Lapic, emulate_apic_mmio},
    ioctl_defs,
    vm_file::Vm,
};
use crate::{
    fs::{
        file::{FileLike, file_table::FdFlags},
        pseudofs::AnonInodeFs,
        vfs::path::Path,
    },
    prelude::*,
    util::ioctl::{RawIoctl, dispatch_ioctl},
};

// Periodically return timer exits so a busy vCPU cannot monopolize scheduling.
const PREEMPTION_TIMER_USER_EXIT_INTERVAL: u64 = 16;
const HLT_WAKEUP_WAIT_TSC_DIVISOR: u64 = 10_000;
const HLT_WAKEUP_WAIT_FALLBACK_TICKS: u64 = 250_000;

/// VCPU file descriptor
pub struct VcpuFile {
    vcpu: Arc<Vcpu>,
    pseudo_path: Path,
}

impl VcpuFile {
    /// Creates a new VCPU file
    pub fn new(vcpu: Arc<Vcpu>) -> Self {
        let pseudo_path = AnonInodeFs::new_path(|_| "anon_inode:[rustshyper-vcpu]".to_string());
        Self { vcpu, pseudo_path }
    }
}

impl FileLike for VcpuFile {
    fn read(&self, _writer: &mut VmWriter) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot read from VCPU file");
    }

    fn write(&self, _reader: &mut VmReader) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot write to VCPU file");
    }

    fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        use ioctl_defs::*;

        dispatch_ioctl!(match raw_ioctl {
            cmd @ Run => {
                let mut guest_mode =
                    GuestMode::new(&self.vcpu.guest_context, &self.vcpu.lapic, &self.vcpu.lapic);
                let mut consecutive_preemption_timer_exits = 0_u64;

                loop {
                    let eptp = self.vcpu.vm()?.guest_mem().eptp();
                    let exit_info = match guest_mode.execute(eptp as u64) {
                        Ok(exit_info) => exit_info,
                        Err(err) => {
                            error!("rustshyper: GuestMode::execute failed: {:?}", err);
                            return Err(err.into());
                        }
                    };
                    let exit_info = match VmxExitReason::try_from(exit_info.exit_reason) {
                        Ok(VmxExitReason::IO_INSTRUCTION) => Some(exit_info),
                        Ok(VmxExitReason::EPT_VIOLATION) => {
                            let handled = emulate_apic_mmio(
                                self.vcpu.clone(),
                                exit_info.guest_phys_addr as u64,
                            )
                            .inspect_err(|err| {
                                error!(
                                    "rustshyper: APIC MMIO handling failed: reason={:#x}, len={}, \
                                     rip={:#x}, gpa={:#x}, qualification={:#x}, err={:?}",
                                    exit_info.exit_reason,
                                    exit_info.instruction_len,
                                    exit_info.guest_rip,
                                    exit_info.guest_phys_addr,
                                    exit_info.exit_qualification,
                                    err
                                );
                            })?;
                            if handled {
                                consecutive_preemption_timer_exits = 0;
                                None
                            } else {
                                Some(exit_info)
                            }
                        }
                        Ok(VmxExitReason::PREEMPTION_TIMER) => {
                            consecutive_preemption_timer_exits =
                                consecutive_preemption_timer_exits.saturating_add(1);
                            if consecutive_preemption_timer_exits
                                >= PREEMPTION_TIMER_USER_EXIT_INTERVAL
                            {
                                Some(exit_info)
                            } else {
                                None
                            }
                        }
                        Ok(VmxExitReason::HLT) => {
                            consecutive_preemption_timer_exits = 0;
                            self.vcpu
                                .guest_context()
                                .advance_rip(exit_info.instruction_len as _);
                            if self.vcpu.wait_for_hlt_wakeup() {
                                None
                            } else {
                                Some(exit_info)
                            }
                        }
                        Ok(_) => Some(exit_info),
                        Err(_) => panic!("Unknown exit reason: {:?}", exit_info.exit_reason),
                    };
                    if let Some(exit_info) = exit_info {
                        // Return to userspace with exit info.
                        let run_state = build_run_state_message(exit_info);
                        cmd.write(&run_state)?;
                        return Ok(0);
                    }
                }
            }
            cmd @ GetRegs => {
                let regs = self.vcpu.get_regs()?;
                cmd.write(&regs)?;
                Ok(0)
            }
            cmd @ SetRegs => {
                let regs = cmd.read()?;
                self.vcpu.set_regs(regs)?;
                Ok(0)
            }
            cmd @ GetSregs => {
                let sregs = self.vcpu.get_sregs()?;
                cmd.write(&sregs)?;
                Ok(0)
            }
            cmd @ SetSregs => {
                let sregs = cmd.read()?;
                self.vcpu.set_sregs(sregs)?;
                Ok(0)
            }
            cmd @ InjectInterrupt => {
                // let vector = cmd.read()?;
                // self.vcpu.inject_interrupt(vector)?;
                Ok(0)
            }
            _ => {
                return_errno_with_message!(Errno::ENOTTY, "unknown VCPU ioctl command");
            }
        })
    }

    fn path(&self) -> &Path {
        &self.pseudo_path
    }

    fn dump_proc_fdinfo(self: Arc<Self>, _fd_flags: FdFlags) -> Box<dyn core::fmt::Display> {
        Box::new("rustshyper_vcpu\n")
    }
}

impl crate::process::signal::Pollable for VcpuFile {
    fn poll(
        &self,
        _mask: crate::events::IoEvents,
        _poller: Option<&mut crate::process::signal::PollHandle>,
    ) -> crate::events::IoEvents {
        // VCPUs don't support polling
        crate::events::IoEvents::empty()
    }
}

fn build_run_state_message(exit_info: GuestExitInfo) -> ioctl_defs::RunStateMessage {
    ioctl_defs::RunStateMessage {
        exit_reason: exit_info.exit_reason,
        instruction_len: exit_info.instruction_len,
        guest_rip: exit_info.guest_rip as u64,
        guest_phys_addr: exit_info.guest_phys_addr as u64,
        exit_qualification: exit_info.exit_qualification,
    }
}
