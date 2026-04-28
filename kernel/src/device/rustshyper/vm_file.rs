// SPDX-License-Identifier: MPL-2.0

//! VM file descriptor implementation

use aster_rustshyper::vm::{MemoryRegion, Vm};
use ostd::task::Task;

use super::{ioctl_defs, vcpu_file::VcpuFile};
use crate::{
    fs::{
        file::{FileLike, file_table::FdFlags},
        pseudofs::AnonInodeFs,
        vfs::path::Path,
    },
    prelude::*,
    process::posix_thread::AsThreadLocal,
    util::ioctl::{RawIoctl, dispatch_ioctl},
};

/// VM file descriptor
pub struct VmFile {
    vm: Arc<Vm>,
    pseudo_path: Path,
}

impl TryFrom<ioctl_defs::UserMemoryRegion> for MemoryRegion {
    type Error = Error;

    fn try_from(user_region: ioctl_defs::UserMemoryRegion) -> Result<Self> {
        Ok(Self {
            slot: user_region.slot,
            flags: user_region.flags,
            guest_phys_addr: user_region.guest_phys_addr,
            memory_size: usize::try_from(user_region.memory_size).map_err(|_| {
                Error::with_message(
                    Errno::EINVAL,
                    "user memory region size does not fit in usize",
                )
            })?,
            userspace_addr: usize::try_from(user_region.userspace_addr).map_err(|_| {
                Error::with_message(
                    Errno::EINVAL,
                    "user memory region address does not fit in usize",
                )
            })?,
        })
    }
}

impl VmFile {
    /// Creates a new VM file
    pub fn new(vm: Arc<Vm>) -> Self {
        let pseudo_path = AnonInodeFs::new_path(|_| "anon_inode:[rustshyper-vm]".to_string());
        Self { vm, pseudo_path }
    }

}

fn log_vm_error(context: &str, err: &Error) {
    match err.message() {
        Some(msg) => {
            error!(
                "rustshyper: {} failed: errno={:?}, msg={}",
                context,
                err.error(),
                msg
            );
        }
        None => {
            error!("rustshyper: {} failed: errno={:?}", context, err.error());
        }
    }
}

impl FileLike for VmFile {
    fn read(&self, _writer: &mut VmWriter) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot read from VM file");
    }

    fn write(&self, _reader: &mut VmReader) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot write to VM file");
    }

    fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        use ioctl_defs::*;

        dispatch_ioctl!(match raw_ioctl {
            cmd @ CreateVcpu => {
                let vcpu_id = cmd.read()?;

                // Create the VCPU
                let vcpu = self.vm.create_vcpu(vcpu_id)?;

                // Create a file descriptor for the VCPU
                let vcpu_file = Arc::new(VcpuFile::new(vcpu));

                // Insert into the current process's file table
                let current = Task::current().unwrap();
                let mut file_table = current.as_thread_local().unwrap().borrow_file_table_mut();
                let mut file_table_locked = file_table.unwrap().write();
                let vcpu_fd = file_table_locked.insert(vcpu_file, FdFlags::empty());

                Ok(vcpu_fd.into())
            }
            cmd @ SetUserMemoryRegion => {
                let user_region: UserMemoryRegion = cmd.read()?;
                let region = MemoryRegion::try_from(user_region).inspect_err(|err| {
                    log_vm_error("convert UserMemoryRegion", err);
                })?;

                let current = match Task::current() {
                    Some(current) => current,
                    None => {
                        error!("rustshyper: no current task found for rustshyper ioctl");
                        return Err(Error::new(Errno::ESRCH));
                    }
                };
                let thread_local = match current.as_thread_local() {
                    Some(thread_local) => thread_local,
                    None => {
                        error!("rustshyper: current task has no ThreadLocal for rustshyper ioctl");
                        return Err(Error::new(Errno::EFAULT));
                    }
                };
                let vm_space = {
                    let vmar = thread_local.vmar().borrow();
                    match vmar.as_ref() {
                        Some(vmar) => vmar.vm_space().clone(),
                        None => {
                            error!(
                                "rustshyper: current thread has no active VMAR for rustshyper ioctl"
                            );
                            return Err(Error::new(Errno::EFAULT));
                        }
                    }
                };

                self.vm.set_memory_region(region, &vm_space).map_err(|err| {
                    let kernel_err: Error = err.into();
                    error!(
                        "rustshyper: set_memory_region ioctl failed for slot={} gpa={:#x} size={:#x} hva={:#x}",
                        user_region.slot,
                        user_region.guest_phys_addr,
                        user_region.memory_size,
                        user_region.userspace_addr
                    );
                    log_vm_error("set_memory_region", &kernel_err);
                    kernel_err
                })?;
                Ok(0)
            }
            cmd @ InjectIrq => {
                let irq = cmd.read()?;
                self.vm.inject_irq_line(irq as usize)?;
                Ok(0)
            }
            GetDirtyLog => {
                // TODO: Implement dirty log tracking
                return_errno_with_message!(Errno::ENOSYS, "GetDirtyLog not yet implemented");
            }
            _ => {
                return_errno_with_message!(Errno::ENOTTY, "unknown VM ioctl command");
            }
        })
    }

    fn path(&self) -> &Path {
        &self.pseudo_path
    }

    fn dump_proc_fdinfo(self: Arc<Self>, _fd_flags: FdFlags) -> Box<dyn core::fmt::Display> {
        Box::new(alloc::format!("vm_id: {}\n", self.vm.id()))
    }
}

impl crate::process::signal::Pollable for VmFile {
    fn poll(
        &self,
        _mask: crate::events::IoEvents,
        _poller: Option<&mut crate::process::signal::PollHandle>,
    ) -> crate::events::IoEvents {
        // VMs don't support polling
        crate::events::IoEvents::empty()
    }
}
