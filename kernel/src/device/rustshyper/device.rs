// SPDX-License-Identifier: MPL-2.0

//! RustShyper main device implementation

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU32, Ordering};

use device_id::{DeviceId, MajorId, MinorId};
use aster_rustshyper::vm::Vm;

use super::{
    ioctl_defs, vm_file::VmFile, RUSTSHYPER_API_VERSION, RUSTSHYPER_MAJOR, RUSTSHYPER_MINOR,
};
use crate::{
    events::IoEvents,
    fs::{
        device::{Device, DeviceType},
        inode_handle::FileIo,
        utils::{InodeIo, StatusFlags},
    },
    prelude::*,
    process::signal::{PollHandle, Pollable},
    util::ioctl::{dispatch_ioctl, RawIoctl},
};

/// The main RustShyper device (/dev/rustshyper)
pub struct RustShyperDevice {
    /// Next VM ID to allocate
    next_vm_id: Arc<AtomicU32>,
}

impl RustShyperDevice {
    /// Creates a new RustShyper device
    pub fn new() -> Self {
        Self {
            next_vm_id: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl Device for RustShyperDevice {
    fn type_(&self) -> DeviceType {
        DeviceType::Char
    }

    fn id(&self) -> DeviceId {
        DeviceId::new(
            MajorId::new(RUSTSHYPER_MAJOR),
            MinorId::new(RUSTSHYPER_MINOR),
        )
    }

    fn devtmpfs_path(&self) -> Option<String> {
        Some("rustshyper".into())
    }

    fn open(&self) -> Result<Box<dyn FileIo>> {
        Ok(Box::new(RustShyperDeviceFile {
            next_vm_id: self.next_vm_id.clone(),
        }))
    }
}

/// File handle for the RustShyper device
struct RustShyperDeviceFile {
    next_vm_id: Arc<AtomicU32>,
}

impl RustShyperDeviceFile {
    fn alloc_vm_id(&self) -> u32 {
        self.next_vm_id.fetch_add(1, Ordering::Relaxed)
    }
}

impl Pollable for RustShyperDeviceFile {
    fn poll(&self, mask: IoEvents, _poller: Option<&mut PollHandle>) -> IoEvents {
        let events = IoEvents::IN | IoEvents::OUT;
        events & mask
    }
}

impl InodeIo for RustShyperDeviceFile {
    fn read_at(
        &self,
        _offset: usize,
        _writer: &mut VmWriter,
        _status_flags: StatusFlags,
    ) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot read from RustShyper device");
    }

    fn write_at(
        &self,
        _offset: usize,
        _reader: &mut VmReader,
        _status_flags: StatusFlags,
    ) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot write to RustShyper device");
    }
}

impl FileIo for RustShyperDeviceFile {
    fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        use ioctl_defs::*;

        dispatch_ioctl!(match raw_ioctl {
            GetApiVersion => {
                Ok(RUSTSHYPER_API_VERSION)
            }
            CreateVm => {
                // Allocate a new VM ID
                let vm_id = self.alloc_vm_id();

                // Create the VM
                let vm = Vm::new(vm_id)?;

                // Create a file descriptor for the VM
                let vm_file = Arc::new(VmFile::new(vm));

                // Insert into the current process's file table
                let current = current_thread!();
                let file_table = current.file_table();
                let mut file_table_locked = file_table.unwrap().write();
                let vm_fd =
                    file_table_locked.insert(vm_file, crate::fs::file_table::FdFlags::empty());

                Ok(vm_fd)
            }
            cmd @ CheckExtension => {
                let _extension = cmd.get();
                // For now, no extensions are supported
                // Return 0 to indicate not supported
                Ok(0)
            }
            _ => {
                return_errno_with_message!(Errno::ENOTTY, "unknown device ioctl command");
            }
        })
    }

    fn check_seekable(&self) -> Result<()> {
        return_errno_with_message!(Errno::ESPIPE, "the device is not seekable");
    }

    fn is_offset_aware(&self) -> bool {
        false
    }
}
