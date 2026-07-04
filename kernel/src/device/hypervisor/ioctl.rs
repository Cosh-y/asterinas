//! Ioctl api compatible with Linux KVM.
//! KVM api: https://www.kernel.org/doc/html/latest/virt/kvm/api.html

use ostd::arch::vm::{
    GuestCpuidEntry, VcpuDtable as ArchVcpuDtable, VcpuRegs as ArchVcpuRegs, VcpuRunState,
    VcpuSegment as ArchVcpuSegment, VcpuSregs as ArchVcpuSregs,
};

use crate::{
    prelude::*,
    util::ioctl::{InData, InOutData, NoData, OutData, ioc},
};

const KVM_INTERRUPT_BITMAP_WORDS: usize = (256 + 63) / 64;
const KVM_APIC_REG_SIZE: usize = 0x400;
pub(super) const KVM_MEM_READONLY: u32 = 1 << 1;

pub(super) const KVM_API_VERSION: i32 = 12;
pub(super) const KVM_RECOMMENDED_VCPUS: i32 = 1;
pub(super) const KVM_MAX_VCPUS: i32 = 64;
pub(super) const KVM_MAX_CPUID_ENTRIES: usize = 100;
pub(super) const KVM_MAX_MSR_ENTRIES: usize = 100;
pub(super) const KVM_MAX_MCE_BANKS: i32 = 32;

pub(super) const KVM_CAP_IRQCHIP: usize = 0;
pub(super) const KVM_CAP_HLT: usize = 1;
pub(super) const KVM_CAP_USER_MEMORY: usize = 3;
pub(super) const KVM_CAP_SET_TSS_ADDR: usize = 4;
pub(super) const KVM_CAP_VAPIC: usize = 6;
pub(super) const KVM_CAP_EXT_CPUID: usize = 7;
pub(super) const KVM_CAP_NR_VCPUS: usize = 9;
pub(super) const KVM_CAP_NR_MEMSLOTS: usize = 10;
pub(super) const KVM_CAP_MP_STATE: usize = 14;
pub(super) const KVM_CAP_COALESCED_MMIO: usize = 15;
pub(super) const KVM_CAP_DESTROY_MEMORY_REGION_WORKS: usize = 21;
pub(super) const KVM_CAP_IRQ_ROUTING: usize = 25;
pub(super) const KVM_CAP_IRQ_INJECT_STATUS: usize = 26;
pub(super) const KVM_CAP_JOIN_MEMORY_REGIONS_WORKS: usize = 30;
pub(super) const KVM_CAP_MCE: usize = 31;
pub(super) const KVM_CAP_IRQFD: usize = 32;
pub(super) const KVM_CAP_PIT2: usize = 33;
pub(super) const KVM_CAP_PIT_STATE2: usize = 35;
pub(super) const KVM_CAP_IOEVENTFD: usize = 36;
pub(super) const KVM_CAP_SET_IDENTITY_MAP_ADDR: usize = 37;
pub(super) const KVM_CAP_ADJUST_CLOCK: usize = 39;
pub(super) const KVM_CAP_INTERNAL_ERROR_DATA: usize = 40;
pub(super) const KVM_CAP_VCPU_EVENTS: usize = 41;
pub(super) const KVM_CAP_DEBUGREGS: usize = 50;
pub(super) const KVM_CAP_X86_ROBUST_SINGLESTEP: usize = 51;
pub(super) const KVM_CAP_ENABLE_CAP: usize = 54;
pub(super) const KVM_CAP_XSAVE: usize = 55;
pub(super) const KVM_CAP_GET_TSC_KHZ: usize = 61;
pub(super) const KVM_CAP_MAX_VCPUS: usize = 66;
pub(super) const KVM_CAP_TSC_DEADLINE_TIMER: usize = 72;
pub(super) const KVM_CAP_SIGNAL_MSI: usize = 77;
pub(super) const KVM_CAP_ENABLE_CAP_VM: usize = 98;
pub(super) const KVM_CAP_SPLIT_IRQCHIP: usize = 121;
pub(super) const KVM_CAP_IOEVENTFD_ANY_LENGTH: usize = 122;
pub(super) const KVM_CAP_MAX_VCPU_ID: usize = 128;
pub(super) const KVM_CAP_IMMEDIATE_EXIT: usize = 136;

pub(super) const KVM_IRQCHIP_PIC_MASTER: u32 = 0;
pub(super) const KVM_IRQCHIP_PIC_SLAVE: u32 = 1;
pub(super) const KVM_IRQCHIP_IOAPIC: u32 = 2;
pub(super) const KVM_IRQ_ROUTING_IRQCHIP: u32 = 1;
pub(super) const KVM_IRQ_ROUTING_MSI: u32 = 2;
pub(super) const KVM_MAX_IRQ_ROUTES: usize = 4096;
pub(super) const KVM_MAX_NR_MEMSLOTS: i32 = 32;
const KVM_IRQCHIP_PAYLOAD_SIZE: usize = 512;

pub(super) const KVM_MP_STATE_RUNNABLE: u32 = 0;
pub(super) const KVM_MP_STATE_UNINITIALIZED: u32 = 1;
pub(super) const KVM_MP_STATE_HALTED: u32 = 3;

pub(super) const KVM_COALESCED_MMIO_PAGE_OFFSET: usize = 2;
pub(super) const KVM_RUN_MMAP_SIZE: usize = (KVM_COALESCED_MMIO_PAGE_OFFSET + 1) * PAGE_SIZE;
pub(super) const KVM_RUN_STRUCT_SIZE: usize = 2352;
const KVM_RUN_EXIT_DATA_OFFSET: usize = 32;
const KVM_RUN_EXIT_DATA_SIZE: usize = KVM_RUN_STRUCT_SIZE - KVM_RUN_EXIT_DATA_OFFSET;

pub(super) const KVM_RUN_IMMEDIATE_EXIT_OFFSET: usize = 1;
pub(super) const KVM_RUN_EXIT_REASON_OFFSET: usize = 8;
pub(super) const KVM_RUN_READY_FOR_INTERRUPT_INJECTION_OFFSET: usize = 12;
pub(super) const KVM_RUN_IF_FLAG_OFFSET: usize = 13;
pub(super) const KVM_RUN_FLAGS_OFFSET: usize = 14;
pub(super) const KVM_RUN_CR8_OFFSET: usize = 16;
pub(super) const KVM_RUN_APIC_BASE_OFFSET: usize = 24;

pub(super) const KVM_RUN_IO_DIRECTION_OFFSET: usize = 32;
pub(super) const KVM_RUN_IO_SIZE_OFFSET: usize = 33;
pub(super) const KVM_RUN_IO_PORT_OFFSET: usize = 34;
pub(super) const KVM_RUN_IO_COUNT_OFFSET: usize = 36;
pub(super) const KVM_RUN_IO_DATA_OFFSET_OFFSET: usize = 40;
pub(super) const KVM_RUN_IO_DATA_OFFSET: usize = 2560;

pub(super) const KVM_RUN_MMIO_PHYS_ADDR_OFFSET: usize = 32;
pub(super) const KVM_RUN_MMIO_DATA_OFFSET: usize = 40;
pub(super) const KVM_RUN_MMIO_LEN_OFFSET: usize = 48;
pub(super) const KVM_RUN_MMIO_IS_WRITE_OFFSET: usize = 52;

pub(super) const KVM_EXIT_IO: u32 = 2;
pub(super) const KVM_EXIT_HLT: u32 = 5;
pub(super) const KVM_EXIT_MMIO: u32 = 6;
pub(super) const KVM_EXIT_INTERNAL_ERROR: u32 = 17;

pub(super) const KVM_EXIT_IO_IN: u8 = 0;
pub(super) const KVM_EXIT_IO_OUT: u8 = 1;

pub(super) const IA32_TSC_DEADLINE: u32 = 0x6e0;

// KVM _IO commands may still pass scalar values in the ioctl argument.
// The command word itself encodes no direction or data size for them.

// System ioctls.
pub(super) type GetApiVersion = ioc!(KVM_GET_API_VERSION, 0xAE, 0x00, NoData);
pub(super) type CreateVm = ioc!(KVM_CREATE_VM, 0xAE, 0x01, NoData);
pub(super) type GetMsrIndexList = ioc!(KVM_GET_MSR_INDEX_LIST, 0xAE, 0x02, InOutData<MsrList>);
pub(super) type CheckExtension = ioc!(KVM_CHECK_EXTENSION, 0xAE, 0x03, NoData);
pub(super) type GetVcpuMmapSize = ioc!(KVM_GET_VCPU_MMAP_SIZE, 0xAE, 0x04, NoData);
pub(super) type GetSupportedCpuid =
    ioc!(KVM_GET_SUPPORTED_CPUID, 0xAE, 0x05, InOutData<VcpuCpuid2>);
pub(super) type X86GetMceCapSupported =
    ioc!(KVM_X86_GET_MCE_CAP_SUPPORTED, 0xAE, 0x9d, OutData<u64>);
pub(super) type GetStatsFd = ioc!(KVM_GET_STATS_FD, 0xAE, 0xce, NoData);

// VM ioctls.
pub(super) type CreateVcpu = ioc!(KVM_CREATE_VCPU, 0xAE, 0x41, NoData);
pub(super) type SetNrMmuPages = ioc!(KVM_SET_NR_MMU_PAGES, 0xAE, 0x44, NoData);
pub(super) type SetUserMemoryRegion = ioc!(
    KVM_SET_USER_MEMORY_REGION,
    0xAE,
    0x46,
    InData<UserMemoryRegion>
);
pub(super) type SetTssAddr = ioc!(KVM_SET_TSS_ADDR, 0xAE, 0x47, NoData);
pub(super) type SetIdentityMapAddr = ioc!(KVM_SET_IDENTITY_MAP_ADDR, 0xAE, 0x48, InData<u64>);
pub(super) type CreateIrqchip = ioc!(KVM_CREATE_IRQCHIP, 0xAE, 0x60, NoData);
pub(super) type IrqLine = ioc!(KVM_IRQ_LINE, 0xAE, 0x61, InData<IrqLevel>);
pub(super) type GetIrqchip = ioc!(KVM_GET_IRQCHIP, 0xAE, 0x62, InOutData<IrqChip>);
pub(super) type SetIrqchip = ioc!(KVM_SET_IRQCHIP, 0xAE, 0x63, OutData<IrqChip>);
pub(super) type IrqLineStatus = ioc!(KVM_IRQ_LINE_STATUS, 0xAE, 0x67, InOutData<IrqLevel>);
pub(super) type RegisterCoalescedMmio = ioc!(
    KVM_REGISTER_COALESCED_MMIO,
    0xAE,
    0x67,
    InData<CoalescedMmioZone>
);
pub(super) type UnregisterCoalescedMmio = ioc!(
    KVM_UNREGISTER_COALESCED_MMIO,
    0xAE,
    0x68,
    InData<CoalescedMmioZone>
);
pub(super) type SetGsiRouting = ioc!(KVM_SET_GSI_ROUTING, 0xAE, 0x6a, InData<IrqRouting>);
pub(super) type IrqFd = ioc!(KVM_IRQFD, 0xAE, 0x76, InData<IrqFdConfig>);
pub(super) type CreatePit2 = ioc!(KVM_CREATE_PIT2, 0xAE, 0x77, InData<PitConfig>);
pub(super) type IoEventFd = ioc!(KVM_IOEVENTFD, 0xAE, 0x79, InData<IoEventFdConfig>);
pub(super) type SetClock = ioc!(KVM_SET_CLOCK, 0xAE, 0x7b, InData<ClockData>);
pub(super) type GetClock = ioc!(KVM_GET_CLOCK, 0xAE, 0x7c, OutData<ClockData>);
pub(super) type SignalMsi = ioc!(KVM_SIGNAL_MSI, 0xAE, 0xa5, InData<MsiMessage>);
pub(super) type EnableCap = ioc!(KVM_ENABLE_CAP, 0xAE, 0xa3, InData<EnableCapData>);

// VCPU ioctls.
pub(super) type Run = ioc!(KVM_RUN, 0xAE, 0x80, NoData);
pub(super) type GetRegs = ioc!(KVM_GET_REGS, 0xAE, 0x81, OutData<VcpuRegs>);
pub(super) type SetRegs = ioc!(KVM_SET_REGS, 0xAE, 0x82, InData<VcpuRegs>);
pub(super) type GetSregs = ioc!(KVM_GET_SREGS, 0xAE, 0x83, OutData<VcpuSregs>);
pub(super) type SetSregs = ioc!(KVM_SET_SREGS, 0xAE, 0x84, InData<VcpuSregs>);
pub(super) type GetMsrs = ioc!(KVM_GET_MSRS, 0xAE, 0x88, InOutData<VcpuMsrs>);
pub(super) type SetMsrs = ioc!(KVM_SET_MSRS, 0xAE, 0x89, InData<VcpuMsrs>);
pub(super) type SetFpu = ioc!(KVM_SET_FPU, 0xAE, 0x8d, InData<VcpuFpu>);
pub(super) type GetLapic = ioc!(KVM_GET_LAPIC, 0xAE, 0x8e, OutData<LapicState>);
pub(super) type SetLapic = ioc!(KVM_SET_LAPIC, 0xAE, 0x8f, InData<LapicState>);
pub(super) type SetCpuid2 = ioc!(KVM_SET_CPUID2, 0xAE, 0x90, InData<VcpuCpuid2>);
pub(super) type TprAccessReporting =
    ioc!(KVM_TPR_ACCESS_REPORTING, 0xAE, 0x92, InOutData<TprAccessCtl>);
pub(super) type SetVapicAddr = ioc!(KVM_SET_VAPIC_ADDR, 0xAE, 0x93, InData<VapicAddr>);
pub(super) type GetMpState = ioc!(KVM_GET_MP_STATE, 0xAE, 0x98, OutData<MpState>);
pub(super) type SetMpState = ioc!(KVM_SET_MP_STATE, 0xAE, 0x99, InData<MpState>);
pub(super) type X86SetupMce = ioc!(KVM_X86_SETUP_MCE, 0xAE, 0x9c, InData<u64>);
pub(super) type GetVcpuEvents = ioc!(KVM_GET_VCPU_EVENTS, 0xAE, 0x9f, OutData<VcpuEvents>);
pub(super) type SetVcpuEvents = ioc!(KVM_SET_VCPU_EVENTS, 0xAE, 0xa0, InData<VcpuEvents>);
pub(super) type GetDebugRegs = ioc!(KVM_GET_DEBUGREGS, 0xAE, 0xa1, OutData<DebugRegs>);
pub(super) type SetTscKhz = ioc!(KVM_SET_TSC_KHZ, 0xAE, 0xa2, NoData);
pub(super) type SetDebugRegs = ioc!(KVM_SET_DEBUGREGS, 0xAE, 0xa2, InData<DebugRegs>);
pub(super) type GetTscKhz = ioc!(KVM_GET_TSC_KHZ, 0xAE, 0xa3, NoData);
pub(super) type GetXsave = ioc!(KVM_GET_XSAVE, 0xAE, 0xa4, OutData<XsaveState>);
pub(super) type SetXsave = ioc!(KVM_SET_XSAVE, 0xAE, 0xa5, InData<XsaveState>);

pub(super) fn check_extension(extension: usize) -> i32 {
    match extension {
        KVM_CAP_IRQCHIP
        | KVM_CAP_HLT
        | KVM_CAP_USER_MEMORY
        | KVM_CAP_SET_TSS_ADDR
        | KVM_CAP_VAPIC
        | KVM_CAP_EXT_CPUID
        | KVM_CAP_MP_STATE
        | KVM_CAP_DESTROY_MEMORY_REGION_WORKS
        | KVM_CAP_IRQ_INJECT_STATUS
        | KVM_CAP_JOIN_MEMORY_REGIONS_WORKS
        | KVM_CAP_IRQFD
        | KVM_CAP_PIT2
        | KVM_CAP_PIT_STATE2
        | KVM_CAP_IOEVENTFD
        | KVM_CAP_SET_IDENTITY_MAP_ADDR
        | KVM_CAP_ADJUST_CLOCK
        | KVM_CAP_INTERNAL_ERROR_DATA
        | KVM_CAP_VCPU_EVENTS
        | KVM_CAP_DEBUGREGS
        | KVM_CAP_X86_ROBUST_SINGLESTEP
        | KVM_CAP_ENABLE_CAP
        | KVM_CAP_XSAVE
        | KVM_CAP_GET_TSC_KHZ
        | KVM_CAP_TSC_DEADLINE_TIMER
        | KVM_CAP_SIGNAL_MSI
        | KVM_CAP_ENABLE_CAP_VM
        | KVM_CAP_IOEVENTFD_ANY_LENGTH
        | KVM_CAP_IMMEDIATE_EXIT => 1,
        KVM_CAP_NR_VCPUS => KVM_RECOMMENDED_VCPUS,
        KVM_CAP_NR_MEMSLOTS => KVM_MAX_NR_MEMSLOTS,
        KVM_CAP_IRQ_ROUTING => KVM_MAX_IRQ_ROUTES as i32,
        KVM_CAP_MCE => KVM_MAX_MCE_BANKS,
        KVM_CAP_MAX_VCPUS => KVM_MAX_VCPUS,
        KVM_CAP_MAX_VCPU_ID => KVM_MAX_VCPUS,
        KVM_CAP_COALESCED_MMIO => KVM_COALESCED_MMIO_PAGE_OFFSET as i32,
        // TODO: Report capabilities from the actual hypervisor implementation.
        _ => 0,
    }
}

/// The x86 `struct kvm_msr_list`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct MsrList {
    pub nmsrs: u32,
    pub indices: [u32; 0],
}

/// The x86 `struct kvm_userspace_memory_region`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct UserMemoryRegion {
    pub slot: u32,
    pub flags: u32,
    pub guest_phys_addr: u64,
    pub memory_size: u64,
    pub userspace_addr: u64,
}

/// The common `struct kvm_irq_level`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct IrqLevel {
    pub irq: u32,
    pub level: u32,
}

/// The common `struct kvm_coalesced_mmio_zone`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct CoalescedMmioZone {
    pub addr: u64,
    pub size: u32,
    pub pio: u32,
}

/// The common `struct kvm_irqchip`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct IrqChip {
    pub chip_id: u32,
    pub pad: u32,
    pub chip: [u8; KVM_IRQCHIP_PAYLOAD_SIZE],
}

impl Default for IrqChip {
    fn default() -> Self {
        Self {
            chip_id: 0,
            pad: 0,
            chip: [0; KVM_IRQCHIP_PAYLOAD_SIZE],
        }
    }
}

/// The common `struct kvm_ioeventfd`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct IoEventFdConfig {
    pub datamatch: u64,
    pub addr: u64,
    pub len: u32,
    pub fd: i32,
    pub flags: u32,
    pub pad: [u8; 36],
}

impl Default for IoEventFdConfig {
    fn default() -> Self {
        Self {
            datamatch: 0,
            addr: 0,
            len: 0,
            fd: 0,
            flags: 0,
            pad: [0; 36],
        }
    }
}

/// The common `struct kvm_enable_cap`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct EnableCapData {
    pub cap: u32,
    pub flags: u32,
    pub args: [u64; 4],
    pub pad: [u8; 64],
}

impl Default for EnableCapData {
    fn default() -> Self {
        Self {
            cap: 0,
            flags: 0,
            args: [0; 4],
            pad: [0; 64],
        }
    }
}

/// The common `struct kvm_irqfd`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct IrqFdConfig {
    pub fd: u32,
    pub gsi: u32,
    pub flags: u32,
    pub resamplefd: u32,
    pub pad: [u8; 16],
}

/// The common `struct kvm_clock_data`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct ClockData {
    pub clock: u64,
    pub flags: u32,
    pub pad0: u32,
    pub realtime: u64,
    pub host_tsc: u64,
    pub pad: [u32; 4],
}

/// The common `struct kvm_msi`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct MsiMessage {
    pub address_lo: u32,
    pub address_hi: u32,
    pub data: u32,
    pub flags: u32,
    pub devid: u32,
    pub pad: [u8; 12],
}

/// The common `struct kvm_pit_config`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct PitConfig {
    pub flags: u32,
    pub pad: [u32; 15],
}

/// The x86 `struct kvm_tpr_access_ctl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct TprAccessCtl {
    pub enabled: u32,
    pub flags: u32,
    pub reserved: [u32; 8],
}

/// The x86 `struct kvm_vapic_addr`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VapicAddr {
    pub vapic_addr: u64,
}

/// The common `struct kvm_mp_state`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct MpState {
    pub mp_state: u32,
}

impl From<VcpuRunState> for MpState {
    fn from(state: VcpuRunState) -> Self {
        Self {
            mp_state: match state {
                VcpuRunState::Runnable | VcpuRunState::Running => KVM_MP_STATE_RUNNABLE,
                VcpuRunState::Uninitialized | VcpuRunState::WaitForSipi => {
                    KVM_MP_STATE_UNINITIALIZED
                }
                VcpuRunState::Halted => KVM_MP_STATE_HALTED,
            },
        }
    }
}

impl TryFrom<MpState> for VcpuRunState {
    type Error = Error;

    fn try_from(state: MpState) -> core::result::Result<Self, Self::Error> {
        match state.mp_state {
            KVM_MP_STATE_RUNNABLE => Ok(Self::Runnable),
            KVM_MP_STATE_UNINITIALIZED => Ok(Self::WaitForSipi),
            KVM_MP_STATE_HALTED => Ok(Self::Halted),
            _ => Err(Error::with_message(
                Errno::EINVAL,
                "unsupported KVM MP state",
            )),
        }
    }
}

/// The x86 `struct kvm_debugregs`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct DebugRegs {
    pub db: [u64; 4],
    pub dr6: u64,
    pub dr7: u64,
    pub flags: u64,
    pub reserved: [u64; 9],
}

/// The x86 exception portion of `struct kvm_vcpu_events`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuEventException {
    pub injected: u8,
    pub nr: u8,
    pub has_error_code: u8,
    pub pending: u8,
    pub error_code: u32,
}

/// The x86 interrupt portion of `struct kvm_vcpu_events`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuEventInterrupt {
    pub injected: u8,
    pub nr: u8,
    pub soft: u8,
    pub shadow: u8,
}

/// The x86 NMI portion of `struct kvm_vcpu_events`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuEventNmi {
    pub injected: u8,
    pub pending: u8,
    pub masked: u8,
    pub pad: u8,
}

/// The x86 SMI portion of `struct kvm_vcpu_events`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuEventSmi {
    pub smm: u8,
    pub pending: u8,
    pub smm_inside_nmi: u8,
    pub latched_init: u8,
}

/// The x86 triple-fault portion of `struct kvm_vcpu_events`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuEventTripleFault {
    pub pending: u8,
}

/// The x86 `struct kvm_vcpu_events`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuEvents {
    pub exception: VcpuEventException,
    pub interrupt: VcpuEventInterrupt,
    pub nmi: VcpuEventNmi,
    pub sipi_vector: u32,
    pub flags: u32,
    pub smi: VcpuEventSmi,
    pub triple_fault: VcpuEventTripleFault,
    pub reserved: [u8; 26],
    pub exception_has_payload: u8,
    pub exception_payload: u64,
}

/// The x86 `struct kvm_xsave`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct XsaveState {
    pub region: [u32; 1024],
}

impl Default for XsaveState {
    fn default() -> Self {
        Self { region: [0; 1024] }
    }
}

/// The common `struct kvm_irq_routing_entry`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct IrqRoutingEntry {
    pub gsi: u32,
    pub type_: u32,
    pub flags: u32,
    pub pad: u32,
    pub data: [u32; 8],
}

/// The common `struct kvm_irq_routing`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct IrqRouting {
    pub nr: u32,
    pub flags: u32,
    pub entries: [IrqRoutingEntry; 0],
}

/// The x86 `struct kvm_regs`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuRegs {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub rflags: u64,
}

impl From<ArchVcpuRegs> for VcpuRegs {
    fn from(regs: ArchVcpuRegs) -> Self {
        Self {
            rax: regs.rax,
            rbx: regs.rbx,
            rcx: regs.rcx,
            rdx: regs.rdx,
            rsi: regs.rsi,
            rdi: regs.rdi,
            rsp: regs.rsp,
            rbp: regs.rbp,
            r8: regs.r8,
            r9: regs.r9,
            r10: regs.r10,
            r11: regs.r11,
            r12: regs.r12,
            r13: regs.r13,
            r14: regs.r14,
            r15: regs.r15,
            rip: regs.rip,
            rflags: regs.rflags,
        }
    }
}

impl From<VcpuRegs> for ArchVcpuRegs {
    fn from(regs: VcpuRegs) -> Self {
        Self {
            rax: regs.rax,
            rbx: regs.rbx,
            rcx: regs.rcx,
            rdx: regs.rdx,
            rsi: regs.rsi,
            rdi: regs.rdi,
            rbp: regs.rbp,
            rsp: regs.rsp,
            r8: regs.r8,
            r9: regs.r9,
            r10: regs.r10,
            r11: regs.r11,
            r12: regs.r12,
            r13: regs.r13,
            r14: regs.r14,
            r15: regs.r15,
            rip: regs.rip,
            rflags: regs.rflags,
        }
    }
}

/// The x86 `struct kvm_segment`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuSegment {
    pub base: u64,
    pub limit: u32,
    pub selector: u16,
    pub type_: u8,
    pub present: u8,
    pub dpl: u8,
    pub db: u8,
    pub s: u8,
    pub l: u8,
    pub g: u8,
    pub avl: u8,
    pub unusable: u8,
    pub padding: u8,
}

impl From<ArchVcpuSegment> for VcpuSegment {
    fn from(segment: ArchVcpuSegment) -> Self {
        Self {
            base: segment.base,
            limit: segment.limit,
            selector: segment.selector,
            type_: segment.type_,
            present: segment.present,
            dpl: segment.dpl,
            db: segment.db,
            s: segment.s,
            l: segment.l,
            g: segment.g,
            avl: segment.avl,
            unusable: segment.unusable,
            padding: segment.padding,
        }
    }
}

impl From<VcpuSegment> for ArchVcpuSegment {
    fn from(segment: VcpuSegment) -> Self {
        Self {
            base: segment.base,
            limit: segment.limit,
            selector: segment.selector,
            type_: segment.type_,
            present: segment.present,
            dpl: segment.dpl,
            db: segment.db,
            s: segment.s,
            l: segment.l,
            g: segment.g,
            avl: segment.avl,
            unusable: segment.unusable,
            padding: segment.padding,
        }
    }
}

/// The x86 `struct kvm_dtable`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuDtable {
    pub base: u64,
    pub limit: u16,
    pub padding: [u16; 3],
}

impl From<ArchVcpuDtable> for VcpuDtable {
    fn from(dtable: ArchVcpuDtable) -> Self {
        Self {
            base: dtable.base,
            limit: dtable.limit,
            padding: dtable.padding,
        }
    }
}

impl From<VcpuDtable> for ArchVcpuDtable {
    fn from(dtable: VcpuDtable) -> Self {
        Self {
            base: dtable.base,
            limit: dtable.limit,
            padding: dtable.padding,
        }
    }
}

/// The x86 `struct kvm_sregs`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuSregs {
    pub cs: VcpuSegment,
    pub ds: VcpuSegment,
    pub es: VcpuSegment,
    pub fs: VcpuSegment,
    pub gs: VcpuSegment,
    pub ss: VcpuSegment,
    pub tr: VcpuSegment,
    pub ldt: VcpuSegment,
    pub gdt: VcpuDtable,
    pub idt: VcpuDtable,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub cr8: u64,
    pub efer: u64,
    pub apic_base: u64,
    pub interrupt_bitmap: [u64; KVM_INTERRUPT_BITMAP_WORDS],
}

impl From<ArchVcpuSregs> for VcpuSregs {
    fn from(sregs: ArchVcpuSregs) -> Self {
        Self {
            cs: sregs.cs.into(),
            ds: sregs.ds.into(),
            es: sregs.es.into(),
            fs: sregs.fs.into(),
            gs: sregs.gs.into(),
            ss: sregs.ss.into(),
            tr: sregs.tr.into(),
            ldt: sregs.ldt.into(),
            gdt: sregs.gdt.into(),
            idt: sregs.idt.into(),
            cr0: sregs.cr0,
            cr2: sregs.cr2,
            cr3: sregs.cr3,
            cr4: sregs.cr4,
            cr8: 0,
            efer: sregs.efer,
            apic_base: sregs.apic_base,
            interrupt_bitmap: sregs.interrupt_bitmap,
        }
    }
}

impl From<VcpuSregs> for ArchVcpuSregs {
    fn from(sregs: VcpuSregs) -> Self {
        Self {
            cs: sregs.cs.into(),
            ds: sregs.ds.into(),
            es: sregs.es.into(),
            fs: sregs.fs.into(),
            gs: sregs.gs.into(),
            ss: sregs.ss.into(),
            tr: sregs.tr.into(),
            ldt: sregs.ldt.into(),
            gdt: sregs.gdt.into(),
            idt: sregs.idt.into(),
            cr0: sregs.cr0,
            cr2: sregs.cr2,
            cr3: sregs.cr3,
            cr4: sregs.cr4,
            efer: sregs.efer,
            apic_base: sregs.apic_base,
            interrupt_bitmap: sregs.interrupt_bitmap,
        }
    }
}

/// The x86 `struct kvm_lapic_state`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct LapicState {
    pub regs: [u8; KVM_APIC_REG_SIZE],
}

impl Default for LapicState {
    fn default() -> Self {
        Self {
            regs: [0; KVM_APIC_REG_SIZE],
        }
    }
}

/// The x86 `struct kvm_fpu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuFpu {
    pub fpr: [[u8; 16]; 8],
    pub fcw: u16,
    pub fsw: u16,
    pub ftwx: u8,
    pub pad1: u8,
    pub last_opcode: u16,
    pub last_ip: u64,
    pub last_dp: u64,
    pub xmm: [[u8; 16]; 16],
    pub mxcsr: u32,
    pub pad2: u32,
}

/// The x86 `struct kvm_msr_entry`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuMsrEntry {
    pub index: u32,
    pub reserved: u32,
    pub data: u64,
}

/// The x86 `struct kvm_msrs`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuMsrs {
    pub nmsrs: u32,
    pub pad: u32,
    pub entries: [VcpuMsrEntry; 0],
}

/// The x86 `struct kvm_cpuid_entry2`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuCpuidEntry2 {
    pub function: u32,
    pub index: u32,
    pub flags: u32,
    pub eax: u32,
    pub ebx: u32,
    pub ecx: u32,
    pub edx: u32,
    pub padding: [u32; 3],
}

impl From<GuestCpuidEntry> for VcpuCpuidEntry2 {
    fn from(entry: GuestCpuidEntry) -> Self {
        Self {
            function: entry.function,
            index: entry.index,
            flags: entry.flags,
            eax: entry.eax,
            ebx: entry.ebx,
            ecx: entry.ecx,
            edx: entry.edx,
            padding: [0; 3],
        }
    }
}

impl From<VcpuCpuidEntry2> for GuestCpuidEntry {
    fn from(entry: VcpuCpuidEntry2) -> Self {
        Self {
            function: entry.function,
            index: entry.index,
            flags: entry.flags,
            eax: entry.eax,
            ebx: entry.ebx,
            ecx: entry.ecx,
            edx: entry.edx,
        }
    }
}

/// The x86 `struct kvm_cpuid2`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct VcpuCpuid2 {
    pub nent: u32,
    pub padding: u32,
    pub entries: [VcpuCpuidEntry2; 0],
}

/// The x86 `struct kvm_run`.
///
/// The Linux layout contains a large union starting at byte 32. Kernel code
/// writes fields by offset so this definition can stay safe Rust while still
/// documenting the userspace ABI shape.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct KvmRun {
    pub request_interrupt_window: u8,
    pub immediate_exit: u8,
    pub padding1: [u8; 6],
    pub exit_reason: u32,
    pub ready_for_interrupt_injection: u8,
    pub if_flag: u8,
    pub flags: u16,
    pub cr8: u64,
    pub apic_base: u64,
    pub exit_data: [u8; KVM_RUN_EXIT_DATA_SIZE],
}

const _: () = assert!(size_of::<KvmRun>() == KVM_RUN_STRUCT_SIZE);
const _: () = assert!(size_of::<MsrList>() == 4);
const _: () = assert!(size_of::<CoalescedMmioZone>() == 16);
const _: () = assert!(size_of::<IrqChip>() == 520);
const _: () = assert!(size_of::<IoEventFdConfig>() == 64);
const _: () = assert!(size_of::<EnableCapData>() == 104);
const _: () = assert!(size_of::<IrqFdConfig>() == 32);
const _: () = assert!(size_of::<ClockData>() == 48);
const _: () = assert!(size_of::<MsiMessage>() == 32);
const _: () = assert!(size_of::<TprAccessCtl>() == 40);
const _: () = assert!(size_of::<VapicAddr>() == 8);
const _: () = assert!(size_of::<MpState>() == 4);
const _: () = assert!(size_of::<DebugRegs>() == 128);
const _: () = assert!(size_of::<VcpuEvents>() == 64);
const _: () = assert!(size_of::<XsaveState>() == 4096);
const _: () = assert!(size_of::<IrqRouting>() == 8);
const _: () = assert!(size_of::<IrqRoutingEntry>() == 48);
const _: () = assert!(size_of::<VcpuMsrs>() == 8);
const _: () = assert!(size_of::<VcpuCpuid2>() == 8);
const _: () = assert!(size_of::<LapicState>() == KVM_APIC_REG_SIZE);
