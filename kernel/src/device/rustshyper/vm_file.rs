// SPDX-License-Identifier: MPL-2.0

//! VM file descriptor implementation

use aster_rustshyper::vm::{MemoryRegion, Vm};

use super::{ioctl_defs, vcpu_file::VcpuFile};
use crate::{
    fs::{file_handle::FileLike, file_table::FdFlags},
    prelude::*,
    util::ioctl::{dispatch_ioctl, RawIoctl},
};

/// VM file descriptor
pub struct VmFile {
    vm: Arc<Vm>,
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
        Self { vm }
    }

    /// Gets the underlying VM
    pub fn vm(&self) -> &Arc<Vm> {
        &self.vm
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
                let current = current_thread!();
                let file_table = current.file_table();
                let mut file_table_locked = file_table.unwrap().write();
                let vcpu_fd = file_table_locked.insert(vcpu_file, FdFlags::empty());

                Ok(vcpu_fd)
            }
            cmd @ SetUserMemoryRegion => {
                let user_region: UserMemoryRegion = cmd.read()?;
                let region = MemoryRegion::try_from(user_region)?;

                self.vm.set_memory_region(region)?;
                Ok(0)
            }
            cmd @ InjectIrq => {
                let irq = cmd.read()?;
                self.vm.inject_irq_line(irq as usize)?;
                Ok(0)
            }
            cmd @ GetDirtyLog => {
                // TODO: Implement dirty log tracking
                return_errno_with_message!(Errno::ENOSYS, "GetDirtyLog not yet implemented");
            }
            _ => {
                return_errno_with_message!(Errno::ENOTTY, "unknown VM ioctl command");
            }
        })
    }

    fn path(&self) -> &crate::fs::path::Path {
        // VM files don't have a real path
        static VM_PATH: spin::Once<crate::fs::path::Path> = spin::Once::new();
        VM_PATH
            .call_once(|| crate::fs::path::Path::new("anon_inode:[rustshyper-vm]".into()).unwrap())
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
