// SPDX-License-Identifier: MPL-2.0

//! VCPU file descriptor implementation

use aster_rustshyper::vm::Vcpu;

use super::ioctl_defs;
use crate::{
    fs::{
        file::{FileLike, file_table::FdFlags},
        pseudofs::AnonInodeFs,
        vfs::path::Path,
    },
    prelude::*,
    util::ioctl::{RawIoctl, dispatch_ioctl},
};

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

    /// Gets the underlying VCPU
    pub fn vcpu(&self) -> &Arc<Vcpu> {
        &self.vcpu
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
                let run_state = self.vcpu.run().map_err(|err| {
                    match err.message() {
                        Some(msg) => {
                            error!(
                                "rustshyper: RSH_RUN failed: errno={:?}, msg={}",
                                err.errno(),
                                msg
                            );
                        }
                        None => {
                            error!("rustshyper: RSH_RUN failed: errno={:?}", err.errno());
                        }
                    }
                    Error::from(err)
                })?;
                let user_run_state = RunStateMessage {
                    exit_reason: run_state.exit_reason,
                    instruction_len: run_state.instruction_len,
                    guest_rip: run_state.guest_rip,
                    guest_phys_addr: run_state.guest_phys_addr,
                    exit_qualification: run_state.exit_qualification,
                    io: IoExitInfo {
                        port: run_state.io.port,
                        size: run_state.io.size,
                        is_in: run_state.io.is_in,
                        is_string: run_state.io.is_string,
                        is_repeat: run_state.io.is_repeat,
                        reserved: run_state.io.reserved,
                        count: run_state.io.count,
                        padding: 0,
                        data: run_state.io.data,
                    },
                    mmio: MmioInfo {
                        phys_addr: run_state.mmio.phys_addr,
                        data: run_state.mmio.data,
                        len: run_state.mmio.len,
                        is_write: run_state.mmio.is_write,
                        reserved: run_state.mmio.reserved,
                    },
                };
                cmd.write(&user_run_state)?;
                Ok(0)
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
                let vector = cmd.read()?;
                self.vcpu.inject_interrupt(vector)?;
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
