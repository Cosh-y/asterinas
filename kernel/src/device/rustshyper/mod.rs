// SPDX-License-Identifier: MPL-2.0

//! RustShyper - Asterinas hypervisor device
//!
//! This module provides KVM-like hypervisor functionality through a character device.

use device_id::{DeviceId, MajorId, MinorId};

use crate::{
    device::registry::char,
    fs::{
        device::{Device, DeviceType},
        inode_handle::FileIo,
        path::PathResolver,
    },
    prelude::*,
};

mod device;
mod ioctl_defs;
mod vcpu_file;
mod vm_file;

pub use device::RustShyperDevice;
pub use ostd::arch::virt::{VcpuRegs, VcpuSregs};
pub use vcpu_file::VcpuFile;
pub use vm_file::VmFile;

/// API version constant
pub const RUSTSHYPER_API_VERSION: i32 = 1;

/// RustShyper device major ID
const RUSTSHYPER_MAJOR: u16 = 10;
/// RustShyper device minor ID
const RUSTSHYPER_MINOR: u16 = 232;

/// Initializes the RustShyper device
pub(super) fn init_in_first_process(path_resolver: &PathResolver) -> Result<()> {
    let device = Arc::new(RustShyperDevice::new());
    char::register(device)?;
    Ok(())
}
