// SPDX-License-Identifier: MPL-2.0

use crate::util::ioctl::{ioc, InData, InOutData, NoData, OutData, PassByVal};

// Main device ioctls
pub(super) type GetApiVersion = ioc!(RSH_GET_API_VERSION, b'H', 0x00, NoData);
pub(super) type CreateVm = ioc!(RSH_CREATE_VM, b'H', 0x01, NoData);
pub(super) type CheckExtension = ioc!(RSH_CHECK_EXTENSION, b'H', 0x03, InData<i32, PassByVal>);

// VM ioctls
pub(super) type CreateVcpu = ioc!(RSH_CREATE_VCPU, b'H', 0x41, InData<u32>);
pub(super) type SetUserMemoryRegion = ioc!(
    RSH_SET_USER_MEMORY_REGION,
    b'H',
    0x46,
    InData<UserMemoryRegion>
);
pub(super) type GetDirtyLog = ioc!(RSH_GET_DIRTY_LOG, b'H', 0x42, InOutData<DirtyLog>);

// VCPU ioctls
pub(super) type Run = ioc!(RSH_RUN, b'H', 0x80, OutData<RunStateMessage>);
pub(super) type GetRegs = ioc!(RSH_GET_REGS, b'H', 0x81, OutData<VcpuRegs>);
pub(super) type SetRegs = ioc!(RSH_SET_REGS, b'H', 0x82, InData<VcpuRegs>);
pub(super) type GetSregs = ioc!(RSH_GET_SREGS, b'H', 0x83, OutData<VcpuSregs>);
pub(super) type SetSregs = ioc!(RSH_SET_SREGS, b'H', 0x84, InData<VcpuSregs>);

// Re-export types from parent module
pub(super) use super::{VcpuRegs, VcpuSregs};

/// User memory region structure for VM memory mapping
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod)]
pub(super) struct UserMemoryRegion {
    /// Slot number
    pub slot: u32,
    /// Flags (e.g., read-only)
    pub flags: u32,
    /// Guest physical address
    pub guest_phys_addr: u64,
    /// Memory size in bytes
    pub memory_size: u64,
    /// Userspace virtual address
    pub userspace_addr: u64,
}

/// Dirty log structure for tracking modified pages
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod)]
pub(super) struct DirtyLog {
    /// Memory slot
    pub slot: u32,
    /// Padding
    pub padding: u32,
    /// Pointer to dirty bitmap
    pub dirty_bitmap: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod)]
pub(super) struct IoExitInfo {
    pub port: u16,
    pub size: u8,
    pub is_in: u8,
    pub is_string: u8,
    pub is_repeat: u8,
    pub reserved: [u8; 2],
    pub count: u32,
    pub data: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod)]
pub(super) struct MmioInfo {
    pub phys_addr: u64,
    pub data: u64,
    pub len: u32,
    pub is_write: u8,
    pub reserved: [u8; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod)]
pub(super) struct RunStateMessage {
    pub exit_reason: u32,
    pub instruction_len: u32,
    pub guest_rip: u64,
    pub guest_phys_addr: u64,
    pub exit_qualification: u64,
    pub io: IoExitInfo,
    pub mmio: MmioInfo,
}
