#![allow(missing_docs)]

pub type PhysAddr = u64;
pub type VirtAddr = usize;
pub type GuestPhysAddr = u64;
pub type MemSize = usize;

/// Guest segment register state.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct VcpuSegment {
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

/// Guest descriptor table state.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct VcpuDtable {
    pub base: u64,
    pub limit: u16,
    pub padding: [u16; 3],
}

/// Guest general purpose registers
///
/// This structure represents the guest CPU's general purpose registers
/// that need to be saved/restored during VM entry/exit.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct VcpuRegs {
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

/// Guest special register state.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct VcpuSregs {
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
    pub efer: u64,
    pub apic_base: u64,
    pub interrupt_bitmap: [u64; 4],
}
