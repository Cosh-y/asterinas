// SPDX-License-Identifier: MPL-2.0

//! The x86 KVM ioctl definitions and register layouts.

use ostd::{arch::vm::{VcpuDescTable, VcpuRegs, VcpuSegment, VcpuSregs}, task::Task};

use crate::{
    ioc, prelude::*, util::ioctl::{InData, InOutData, NoData, OutData, RawIoctl},
};

const KVM_INTERRUPT_BITMAP_WORDS: usize = 4;

pub(super) const PIO_DATA_OFFSET: usize = PAGE_SIZE;

// System ioctls.
pub(super) type GetSupportedCpuid = ioc!(KVM_GET_SUPPORTED_CPUID, 0xae, 0x05, InOutData<KvmCpuid2>);

// VM ioctls.
pub(super) type SetTssAddr = ioc!(KVM_SET_TSS_ADDR, 0xae, 0x47, NoData);
pub(super) type IrqLine = ioc!(KVM_IRQ_LINE, 0xae, 0x61, InData<KvmGsiLevel>);
pub(super) type SetGsiRouting = ioc!(KVM_SET_GSI_ROUTING, 0xae, 0x6a, InData<KvmGsiRouting>);
// On x86, creates a virtual ioapic, a virtual PIC (two PICs, nested), and sets up future vcpus
// to have a local APIC. IRQ routing for GSIs 0-15 is set to both PIC and IOAPIC; GSI 16-23 only
// go to the IOAPIC. This ioctl can only be called before creating any vcpus.
pub(super) type CreateIrqchip = ioc!(KVM_CREATE_IRQCHIP, 0xae, 0x60, NoData);
pub(super) type GetIrqchip = ioc!(KVM_GET_IRQCHIP, 0xae, 0x62, InOutData<KvmIrqchip>);
pub(super) type SetIrqchip = ioc!(KVM_SET_IRQCHIP, 0xae, 0x63, InData<KvmIrqchip>);
pub(super) type CreatePit2 = ioc!(KVM_CREATE_PIT2, 0xae, 0x77, InData<KvmPitConfig>);
pub(super) type GetPit2 = ioc!(KVM_GET_PIT2, 0xae, 0x7f, OutData<KvmPitState2>);
pub(super) type SetPit2 = ioc!(KVM_SET_PIT2, 0xae, 0x80, InData<KvmPitState2>);

// VCPU ioctls.
pub(super) type GetRegs = ioc!(KVM_GET_REGS, 0xae, 0x81, OutData<KvmRegs>);
pub(super) type SetRegs = ioc!(KVM_SET_REGS, 0xae, 0x82, InData<KvmRegs>);
pub(super) type GetSregs = ioc!(KVM_GET_SREGS, 0xae, 0x83, OutData<KvmSregs>);
pub(super) type SetSregs = ioc!(KVM_SET_SREGS, 0xae, 0x84, InData<KvmSregs>);
pub(super) type GetMsrs = ioc!(KVM_GET_MSRS, 0xae, 0x88, InOutData<KvmMsrs>);
pub(super) type SetMsrs = ioc!(KVM_SET_MSRS, 0xae, 0x89, InData<KvmMsrs>);
pub(super) type GetFpu = ioc!(KVM_GET_FPU, 0xae, 0x8c, OutData<KvmFpu>);
pub(super) type SetFpu = ioc!(KVM_SET_FPU, 0xae, 0x8d, InData<KvmFpu>);
pub(super) type SetCpuid2 = ioc!(KVM_SET_CPUID2, 0xae, 0x90, InData<KvmCpuid2>);
// Name an interrupt vector to be injected into the vcpu.
// Used in NONE or SPLIT in-kernel irqchip mode.
pub(super) type Interrupt = ioc!(KVM_INTERRUPT, 0xae, 0x86, InData<u32>);
pub(super) type GetLapic = ioc!(KVM_GET_LAPIC, 0xae, 0x8e, OutData<KvmLapicState>);
pub(super) type SetLapic = ioc!(KVM_SET_LAPIC, 0xae, 0x8f, InData<KvmLapicState>);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct KvmRegs {
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

impl From<VcpuRegs> for KvmRegs {
    fn from(regs: VcpuRegs) -> Self {
        Self {
            rax: regs.rax as u64,
            rbx: regs.rbx as u64,
            rcx: regs.rcx as u64,
            rdx: regs.rdx as u64,
            rsi: regs.rsi as u64,
            rdi: regs.rdi as u64,
            rsp: regs.rsp as u64,
            rbp: regs.rbp as u64,
            r8: regs.r8 as u64,
            r9: regs.r9 as u64,
            r10: regs.r10 as u64,
            r11: regs.r11 as u64,
            r12: regs.r12 as u64,
            r13: regs.r13 as u64,
            r14: regs.r14 as u64,
            r15: regs.r15 as u64,
            rip: regs.rip as u64,
            rflags: regs.rflags as u64,
        }
    }
}

impl From<KvmRegs> for VcpuRegs {
    fn from(regs: KvmRegs) -> Self {
        Self {
            rax: regs.rax as usize,
            rbx: regs.rbx as usize,
            rcx: regs.rcx as usize,
            rdx: regs.rdx as usize,
            rsi: regs.rsi as usize,
            rdi: regs.rdi as usize,
            rbp: regs.rbp as usize,
            rsp: regs.rsp as usize,
            r8: regs.r8 as usize,
            r9: regs.r9 as usize,
            r10: regs.r10 as usize,
            r11: regs.r11 as usize,
            r12: regs.r12 as usize,
            r13: regs.r13 as usize,
            r14: regs.r14 as usize,
            r15: regs.r15 as usize,
            rip: regs.rip as usize,
            rflags: regs.rflags as usize,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct KvmSregs {
    pub cs: VcpuSegment,
    pub ds: VcpuSegment,
    pub es: VcpuSegment,
    pub fs: VcpuSegment,
    pub gs: VcpuSegment,
    pub ss: VcpuSegment,
    pub tr: VcpuSegment,
    pub ldt: VcpuSegment,
    pub gdt: VcpuDescTable,
    pub idt: VcpuDescTable,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub cr8: u64,
    pub efer: u64,
    pub apic_base: u64,
    pub interrupt_bitmap: [u64; KVM_INTERRUPT_BITMAP_WORDS],
}

impl From<VcpuSregs> for KvmSregs {
    fn from(sregs: VcpuSregs) -> Self {
        Self {
            cs: sregs.cs,
            ds: sregs.ds,
            es: sregs.es,
            fs: sregs.fs,
            gs: sregs.gs,
            ss: sregs.ss,
            tr: sregs.tr,
            ldt: sregs.ldt,
            gdt: sregs.gdt,
            idt: sregs.idt,
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

impl From<KvmSregs> for VcpuSregs {
    fn from(sregs: KvmSregs) -> Self {
        Self {
            cs: sregs.cs,
            ds: sregs.ds,
            es: sregs.es,
            fs: sregs.fs,
            gs: sregs.gs,
            ss: sregs.ss,
            tr: sregs.tr,
            ldt: sregs.ldt,
            gdt: sregs.gdt,
            idt: sregs.idt,
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

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmGsiLevel {
    pub gsi: u32,
    pub level: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmGsiRouting {
    nr: u32,
    flags: u32,  // 0, currently.
    entries: [KvmGsiRoutingEntry; 0],
}

pub(super) const KVM_IRQ_ROUTING_IRQCHIP: u32 = 1;
pub(super) const KVM_IRQCHIP_PIC_MASTER: u32 = 0;
pub(super) const KVM_IRQCHIP_PIC_SLAVE: u32 = 1;
pub(super) const KVM_IRQCHIP_IOAPIC: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmGsiRoutingEntry {
    pub gsi: u32,    // GSI in KVM_IRQ_LINE
    pub type_: u32,  // type of `data`, 1 `KVM_IRQ_ROUTING_IRQCHIP` currently.
    pub flags: u32,  // 0, currently.
    pub pad: u32,
    pub data: [u32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmMsrs {
    pub nmsrs: u32,
    pub padding: u32,
    pub entries: [KvmMsrEntry; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmMsrEntry {
    pub index: u32,
    pub reserved: u32,
    pub data: u64,
}

/// The x86 floating-point state used by `KVM_GET_FPU` and `KVM_SET_FPU`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
pub(super) struct KvmFpu {
    pub fpr: [[u8; 16]; 8],
    pub fcw: u16,
    pub fsw: u16,
    /// The abridged x87 tag word in FXSAVE format.
    pub ftwx: u8,
    pub pad1: u8,
    pub last_opcode: u16,
    pub last_ip: u64,
    pub last_dp: u64,
    pub xmm: [[u8; 16]; 16],
    pub mxcsr: u32,
    pub pad2: u32,
}

pub(super) const KVM_CPUID_FLAG_SIGNIFCANT_INDEX: u32 = 1 << 0;

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmCpuid2 {
    nent: u32,
    padding: u32,
    entries: [KvmCpuidEntry2; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod)]
pub(super) struct KvmCpuidEntry2 {
    pub function: u32,
    pub index: u32,
    pub flags: u32,
    pub eax: u32,
    pub ebx: u32,
    pub ecx: u32,
    pub edx: u32,
    pub padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmPitConfig {
    pub flags: u32,
    pub pad: [u32; 15],
}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmPitState2 {
    pub channels: [KvmPitChannelState; 3],
    pub flags: u32,
    pub pad: [u32; 9],
}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmPitChannelState {
    //
}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmIrqchip {
    pub chip_id: u32,
    pub pad: u32,

}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmIoapicState {
    
}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmPicState {

}

#[repr(C)]
#[derive(Clone, Copy, Pod)]
pub(super) struct KvmLapicState {
        
}


pub(super) fn read_trailing_array<T: Pod>(
    raw_ioctl: RawIoctl,
    header_size: usize,
    count: usize,
) -> Result<Vec<T>> {
    let entries_address = raw_ioctl
        .arg()
        .checked_add(header_size)
        .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
    let entries_len = count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
    let current = Task::current().unwrap();
    let thread_local = current.as_thread_local().unwrap();
    let user_space = CurrentUserSpace::new(thread_local);
    let mut reader = user_space.reader(entries_address, entries_len)?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        entries.push(reader.read_val()?);
    }
    Ok(entries)
}

pub(super) fn write_trailing_array<T: Pod>(
    raw_ioctl: RawIoctl,
    header_size: usize,
    entries: &[T],
) -> Result<()> {
    let entries_address = raw_ioctl
        .arg()
        .checked_add(header_size)
        .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
    let count = entries.len();
    let entries_len = count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
    let current = Task::current().unwrap();
    let thread_local = current.as_thread_local().unwrap();
    let user_space = CurrentUserSpace::new(thread_local);
    let mut writer = user_space.writer(entries_address, entries_len)?;
    for i in 0..count {
        writer.write_val(&entries[i])?;
    }
    Ok(())
}
