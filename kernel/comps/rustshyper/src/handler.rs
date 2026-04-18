use ostd::arch::virt::*;

use super::{
    emulate::apic::{
        emulate_ioapic_read, emulate_ioapic_write, emulate_lapic_read, emulate_lapic_write,
        IOAPIC_BASE, IOAPIC_SIZE, LAPIC_BASE, LAPIC_SIZE,
    },
    error::*,
    vm::{GuestMsrState, Vcpu},
};
use crate::interrupt::{
    handle_external_interrupt, handle_interrupt_window, inject_gp_fault, inject_pending_exception,
};

const MAX_INSN_LENGTH: usize = 15;

pub const COM1: u16 = 0x3F8;
pub const COM1_MAX: u16 = COM1 + 8;

const VMX_EXIT_REASON_HLT: u32 = 12;
const VMX_EXIT_REASON_IO_INSTRUCTION: u32 = 30;
const VMX_EXIT_REASON_EPT_VIOLATION: u32 = 48;

const X86_CR0_PE: u64 = 1 << 0;
const X86_CR0_ET: u64 = 1 << 4;
const X86_CR0_NE: u64 = 1 << 5;
const X86_CR0_PG: u64 = 1 << 31;
const X86_CR4_PAE: u64 = 1 << 5;
const X86_CR4_VMXE: u64 = 1 << 13;
const X86_EFER_LMA: u64 = 1 << 10;

const MSR_IA32_TSC: u32 = 0x0000_0010;
const MSR_IA32_APIC_BASE: u32 = 0x0000_001B;
const MSR_IA32_PAT: u32 = 0x0000_0277;
const MSR_EFER: u32 = 0xC000_0080;
const MSR_STAR: u32 = 0xC000_0081;
const MSR_LSTAR: u32 = 0xC000_0082;
const MSR_CSTAR: u32 = 0xC000_0083;
const MSR_SYSCALL_MASK: u32 = 0xC000_0084;
const MSR_FS_BASE: u32 = 0xC000_0100;
const MSR_GS_BASE: u32 = 0xC000_0101;
const MSR_KERNEL_GS_BASE: u32 = 0xC000_0102;
const MSR_IA32_TSC_DEADLINE: u32 = 0x0000_06E0;
const MSR_AMD64_DE_CFG: u32 = 0xC001_1029;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IoExitInfo {
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
#[derive(Debug, Clone, Copy, Default)]
pub struct MmioInfo {
    pub phys_addr: u64,
    pub data: u64,
    pub len: u32,
    pub is_write: u8,
    pub reserved: [u8; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RunStateMessage {
    pub exit_reason: u32,
    pub instruction_len: u32,
    pub guest_rip: u64,
    pub guest_phys_addr: u64,
    pub exit_qualification: u64,
    pub io: IoExitInfo,
    pub mmio: MmioInfo,
}

#[derive(Debug, Clone, Copy)]
struct MmioInstruction {
    is_read: bool,
    size: u8,
    reg: u8,
}

pub fn vmexit_handler(vcpu: &Vcpu, exit_info: &VmxExitInfo) -> Result<Option<RunStateMessage>> {
    if exit_info.entry_failure {
        return Err(Error::with_message(
            Errno::GuestRunFailed,
            "VM-entry failure while entering guest",
        ));
    }

    match VmxExitReason::try_from(exit_info.exit_reason) {
        Ok(VmxExitReason::EXTERNAL_INTERRUPT) => {
            handle_external_interrupt()?;
            Ok(None)
        }
        Ok(VmxExitReason::TRIPLE_FAULT) | Ok(VmxExitReason::HLT) => {
            Ok(Some(build_run_state(exit_info)))
        }
        Ok(VmxExitReason::INTERRUPT_WINDOW) => {
            let mut state = vcpu.state.lock();
            handle_interrupt_window(&mut state.lapic, &mut state.interrupt)?;
            Ok(None)
        }
        Ok(VmxExitReason::CPUID) => {
            vcpu.emulate_cpuid()?;
            advance_guest_rip()?;
            Ok(None)
        }
        Ok(VmxExitReason::VMCALL) => Ok(Some(build_run_state(exit_info))),
        Ok(VmxExitReason::CR_ACCESS) => {
            emulate_cr_access(vcpu)?;
            advance_guest_rip()?;
            Ok(None)
        }
        Ok(VmxExitReason::MSR_READ) => {
            emulate_msrrw(vcpu, false)?;
            advance_guest_rip()?;
            Ok(None)
        }
        Ok(VmxExitReason::MSR_WRITE) => {
            emulate_msrrw(vcpu, true)?;
            advance_guest_rip()?;
            Ok(None)
        }
        Ok(VmxExitReason::IO_INSTRUCTION) => Ok(Some(build_io_run_state(vcpu, exit_info))),
        Ok(VmxExitReason::EPT_VIOLATION) => {
            if emulate_apic_mmio(vcpu, exit_info.guest_phys_addr)? {
                Ok(None)
            } else {
                Ok(Some(build_mmio_run_state(exit_info)))
            }
        }
        Ok(VmxExitReason::PREEMPTION_TIMER) => {
            vcpu.handle_preemption_timer_expire()?;
            Ok(None)
        }
        Ok(_) => Ok(Some(build_run_state(exit_info))),
        Err(_) => Ok(Some(build_run_state(exit_info))),
    }
}

fn build_run_state(exit_info: &VmxExitInfo) -> RunStateMessage {
    RunStateMessage {
        exit_reason: exit_info.exit_reason,
        instruction_len: instruction_len().unwrap_or(0),
        guest_rip: exit_info.guest_rip,
        guest_phys_addr: exit_info.guest_phys_addr,
        exit_qualification: exit_info.exit_qualification,
        io: IoExitInfo::default(),
        mmio: MmioInfo::default(),
    }
}

fn build_io_run_state(vcpu: &Vcpu, exit_info: &VmxExitInfo) -> RunStateMessage {
    let qualification = exit_info.exit_qualification;
    let access_size = ((qualification & 0b111) + 1) as u8;
    let is_in = ((qualification & (1 << 3)) != 0) as u8;
    let is_string = ((qualification & (1 << 4)) != 0) as u8;
    let is_repeat = ((qualification & (1 << 5)) != 0) as u8;
    let port = ((qualification >> 16) & 0xffff) as u16;
    let regs = vcpu.get_regs().unwrap_or_default();

    RunStateMessage {
        exit_reason: VMX_EXIT_REASON_IO_INSTRUCTION,
        instruction_len: instruction_len().unwrap_or(0),
        guest_rip: exit_info.guest_rip,
        guest_phys_addr: exit_info.guest_phys_addr,
        exit_qualification: qualification,
        io: IoExitInfo {
            port,
            size: access_size,
            is_in,
            is_string,
            is_repeat,
            reserved: [0; 2],
            count: 1,
            data: match access_size {
                1 => regs.rax & 0xff,
                2 => regs.rax & 0xffff,
                4 => regs.rax & 0xffff_ffff,
                _ => regs.rax,
            },
        },
        mmio: MmioInfo::default(),
    }
}

fn build_mmio_run_state(exit_info: &VmxExitInfo) -> RunStateMessage {
    let qualification = exit_info.exit_qualification;
    RunStateMessage {
        exit_reason: VMX_EXIT_REASON_EPT_VIOLATION,
        instruction_len: instruction_len().unwrap_or(0),
        guest_rip: exit_info.guest_rip,
        guest_phys_addr: exit_info.guest_phys_addr,
        exit_qualification: qualification,
        io: IoExitInfo::default(),
        mmio: MmioInfo {
            phys_addr: exit_info.guest_phys_addr,
            data: 0,
            len: 0,
            is_write: ((qualification & 0b010) != 0) as u8,
            reserved: [0; 3],
        },
    }
}

fn advance_guest_rip() -> Result<()> {
    let len = instruction_len().map_err(Error::from)? as usize;
    advance_rip(len).map_err(Error::from)
}

fn instruction_len() -> core::result::Result<u32, ostd::Error> {
    VmcsReadOnly32::VMEXIT_INSTRUCTION_LEN.read()
}

fn emulate_cr_access(vcpu: &Vcpu) -> Result<()> {
    let qualification = VmcsReadOnlyNW::EXIT_QUALIFICATION
        .read()
        .map_err(Error::from)?;
    let cr_index = (qualification & 0xF) as u8;
    let access = ((qualification >> 4) & 0b11) as u8;
    let gpr_index = ((qualification >> 8) & 0xF) as u8;

    let mut regs = vcpu.get_regs()?;
    match access {
        0 => {
            let value = read_gpr(&regs, gpr_index, 8);
            match cr_index {
                0 => emulate_cr0_write(vcpu, value)?,
                4 => emulate_cr4_write(value)?,
                3 => log::warn!("rustshyper: ignoring guest write to CR3"),
                other => log::warn!("rustshyper: ignoring guest write to CR{}", other),
            }
        }
        1 => {
            let value = match cr_index {
                0 => VmcsControlNW::CR0_READ_SHADOW.read().map_err(Error::from)? as u64,
                4 => VmcsControlNW::CR4_READ_SHADOW.read().map_err(Error::from)? as u64,
                3 => {
                    log::warn!("rustshyper: ignoring guest read from CR3");
                    0
                }
                other => {
                    log::warn!("rustshyper: ignoring guest read from CR{}", other);
                    0
                }
            };
            write_gpr(&mut regs, gpr_index, 8, value);
            vcpu.set_regs(regs)?;
        }
        other => {
            log::warn!("rustshyper: unsupported CR access type {}", other);
        }
    }

    Ok(())
}

fn emulate_cr0_write(vcpu: &Vcpu, value: u64) -> Result<()> {
    if (value & X86_CR0_PG) != 0 && (value & X86_CR0_PE) == 0 {
        queue_gp_fault(vcpu)?;
        return Ok(());
    }

    let shadow_value = value;
    let actual_value = value | X86_CR0_ET | X86_CR0_NE;

    if (value & X86_CR0_PG) != 0
        && (VmcsGuestNW::CR0.read().map_err(Error::from)? as u64 & X86_CR0_PG) == 0
    {
        let entry = VmcsControl32::VMENTRY_CONTROLS
            .read()
            .map_err(Error::from)?;
        VmcsControl32::VMENTRY_CONTROLS
            .write(entry | (1 << 9))
            .map_err(Error::from)?;
        let efer = VmcsGuest64::IA32_EFER.read().map_err(Error::from)?;
        VmcsGuest64::IA32_EFER
            .write(efer | X86_EFER_LMA)
            .map_err(Error::from)?;
    }

    VmcsControlNW::CR0_READ_SHADOW
        .write(shadow_value as usize)
        .map_err(Error::from)?;
    VmcsGuestNW::CR0
        .write(actual_value as usize)
        .map_err(Error::from)?;
    Ok(())
}

fn emulate_cr4_write(value: u64) -> Result<()> {
    let actual_value = value & !X86_CR4_VMXE;
    let shadow_value = value | X86_CR4_PAE | X86_CR4_VMXE;

    VmcsGuestNW::CR4
        .write(actual_value as usize)
        .map_err(Error::from)?;
    VmcsControlNW::CR4_READ_SHADOW
        .write(shadow_value as usize)
        .map_err(Error::from)?;
    Ok(())
}

fn emulate_msrrw(vcpu: &Vcpu, is_write: bool) -> Result<()> {
    let mut state = vcpu.state.lock();
    let msr_index = state.regs.rcx as u32;

    if is_write {
        let msr_value = (state.regs.rax as u32 as u64) | ((state.regs.rdx as u32 as u64) << 32);
        match msr_index {
            MSR_IA32_TSC => {
                state.tsc.tsc_physical = msr_value;
                state.tsc.tsc_offset = msr_value.wrapping_sub(ostd::arch::read_tsc());
            }
            MSR_IA32_APIC_BASE => {
                state.msrs.apic_base = sanitize_apic_base(msr_value);
            }
            MSR_EFER => state.msrs.efer = msr_value,
            MSR_IA32_PAT => state.msrs.pat = msr_value,
            MSR_FS_BASE => state.msrs.fs_base = msr_value,
            MSR_GS_BASE => state.msrs.gs_base = msr_value,
            MSR_KERNEL_GS_BASE => state.msrs.kernel_gs_base = msr_value,
            MSR_STAR => state.msrs.star = msr_value,
            MSR_LSTAR => state.msrs.lstar = msr_value,
            MSR_CSTAR => state.msrs.cstar = msr_value,
            MSR_SYSCALL_MASK => state.msrs.syscall_mask = msr_value,
            MSR_IA32_TSC_DEADLINE => {
                state.msrs.tsc_deadline = msr_value;
                drop(state);
                vcpu.handle_lapic_timer_write_effect(
                    crate::emulate::apic::LapicWriteEffect::StartTimerDeadline,
                )?;
                return Ok(());
            }
            _ => {
                log::warn!("rustshyper: unrecognized WRMSR #{:#x}", msr_index);
                inject_gp_fault(&mut state.exception);
                let mut perf = [0u32; 32];
                inject_pending_exception(&mut state.exception, &mut perf)?;
                return Ok(());
            }
        }

        update_guest_msr_vmcs(&state.msrs)?;
        return Ok(());
    }

    let msr_value = match msr_index {
        MSR_IA32_TSC => state.tsc.tsc_physical,
        MSR_IA32_APIC_BASE => state.msrs.apic_base,
        MSR_EFER => state.msrs.efer,
        MSR_IA32_PAT => state.msrs.pat,
        MSR_FS_BASE => state.msrs.fs_base,
        MSR_GS_BASE => state.msrs.gs_base,
        MSR_KERNEL_GS_BASE => state.msrs.kernel_gs_base,
        MSR_STAR => state.msrs.star,
        MSR_LSTAR => state.msrs.lstar,
        MSR_CSTAR => state.msrs.cstar,
        MSR_SYSCALL_MASK => state.msrs.syscall_mask,
        MSR_AMD64_DE_CFG => 0,
        MSR_IA32_TSC_DEADLINE => state.msrs.tsc_deadline,
        _ => {
            log::warn!("rustshyper: unrecognized RDMSR #{:#x}", msr_index);
            inject_gp_fault(&mut state.exception);
            let mut perf = [0u32; 32];
            inject_pending_exception(&mut state.exception, &mut perf)?;
            return Ok(());
        }
    };

    state.regs.rax = msr_value as u32 as u64;
    state.regs.rdx = msr_value >> 32;
    Ok(())
}

fn update_guest_msr_vmcs(msrs: &GuestMsrState) -> Result<()> {
    VmcsGuest64::IA32_PAT.write(msrs.pat).map_err(Error::from)?;
    VmcsGuest64::IA32_EFER
        .write(msrs.efer)
        .map_err(Error::from)?;
    VmcsGuestNW::FS_BASE
        .write(msrs.fs_base as usize)
        .map_err(Error::from)?;
    VmcsGuestNW::GS_BASE
        .write(msrs.gs_base as usize)
        .map_err(Error::from)?;
    Ok(())
}

fn sanitize_apic_base(value: u64) -> u64 {
    const APIC_BASE_BSP: u64 = 1 << 8;
    const APIC_BASE_ENABLE: u64 = 1 << 11;

    LAPIC_BASE | APIC_BASE_ENABLE | (value & APIC_BASE_BSP)
}

fn queue_gp_fault(vcpu: &Vcpu) -> Result<()> {
    let mut state = vcpu.state.lock();
    inject_gp_fault(&mut state.exception);
    let mut perf = [0u32; 32];
    inject_pending_exception(&mut state.exception, &mut perf)?;
    Ok(())
}

fn read_gpr(regs: &VcpuRegs, index: u8, size: u8) -> u64 {
    let raw = match index {
        0 => regs.rax,
        1 => regs.rcx,
        2 => regs.rdx,
        3 => regs.rbx,
        4 => regs.rsp,
        5 => regs.rbp,
        6 => regs.rsi,
        7 => regs.rdi,
        8 => regs.r8,
        9 => regs.r9,
        10 => regs.r10,
        11 => regs.r11,
        12 => regs.r12,
        13 => regs.r13,
        14 => regs.r14,
        15 => regs.r15,
        _ => 0,
    };

    match size {
        1 => raw & 0xff,
        2 => raw & 0xffff,
        4 => raw & 0xffff_ffff,
        _ => raw,
    }
}

fn write_gpr(regs: &mut VcpuRegs, index: u8, size: u8, value: u64) {
    let slot = match index {
        0 => &mut regs.rax,
        1 => &mut regs.rcx,
        2 => &mut regs.rdx,
        3 => &mut regs.rbx,
        4 => &mut regs.rsp,
        5 => &mut regs.rbp,
        6 => &mut regs.rsi,
        7 => &mut regs.rdi,
        8 => &mut regs.r8,
        9 => &mut regs.r9,
        10 => &mut regs.r10,
        11 => &mut regs.r11,
        12 => &mut regs.r12,
        13 => &mut regs.r13,
        14 => &mut regs.r14,
        15 => &mut regs.r15,
        _ => return,
    };

    *slot = match size {
        1 => (*slot & !0xff) | (value & 0xff),
        2 => (*slot & !0xffff) | (value & 0xffff),
        4 => value & 0xffff_ffff,
        _ => value,
    };
}

fn emulate_apic_mmio(vcpu: &Vcpu, fault_gpa: u64) -> Result<bool> {
    let is_lapic = (LAPIC_BASE..(LAPIC_BASE + LAPIC_SIZE)).contains(&fault_gpa);
    let is_ioapic = (IOAPIC_BASE..(IOAPIC_BASE + IOAPIC_SIZE)).contains(&fault_gpa);
    if !is_lapic && !is_ioapic {
        return Ok(false);
    }

    let guest_rip = VmcsGuestNW::RIP.read().map_err(Error::from)? as u64;
    let mut insn_bytes = [0_u8; MAX_INSN_LENGTH];
    vcpu.read_guest_memory(guest_rip, &mut insn_bytes)?;
    let Some(insn) = decode_mmio_instruction(&insn_bytes) else {
        return Ok(false);
    };

    if is_lapic {
        if !emulate_lapic_mmio(vcpu, fault_gpa, insn)? {
            return Ok(false);
        }
    } else {
        if !emulate_ioapic_mmio(vcpu, fault_gpa, insn)? {
            return Ok(false);
        }
    }

    advance_guest_rip()?;
    Ok(true)
}

fn emulate_lapic_mmio(vcpu: &Vcpu, fault_gpa: u64, insn: MmioInstruction) -> Result<bool> {
    let offset = fault_gpa - LAPIC_BASE;
    if insn.is_read {
        let mut state = vcpu.state.lock();
        let (value, ok) = emulate_lapic_read(&state.lapic, &state.apic_timer, &state.tsc, offset);
        if !ok {
            return Ok(false);
        }
        write_gpr(&mut state.regs, insn.reg, insn.size, value);
        return Ok(true);
    }

    let value = {
        let state = vcpu.state.lock();
        read_gpr(&state.regs, insn.reg, insn.size)
    };
    let vm = vcpu
        .vm
        .upgrade()
        .ok_or_else(|| Error::with_message(Errno::NotFound, "vm not found"))?;
    let mut ioapic = vm.ioapic.lock();
    let effect = {
        let mut state = vcpu.state.lock();
        let (effect, ok) = emulate_lapic_write(
            &mut state.lapic,
            &mut state.apic_timer,
            &mut ioapic,
            offset,
            value,
        );
        if !ok {
            return Ok(false);
        }
        effect
    };
    drop(ioapic);
    vcpu.handle_lapic_timer_write_effect(effect)?;
    Ok(true)
}

fn emulate_ioapic_mmio(vcpu: &Vcpu, fault_gpa: u64, insn: MmioInstruction) -> Result<bool> {
    let offset = fault_gpa - IOAPIC_BASE;
    let vm = vcpu
        .vm
        .upgrade()
        .ok_or_else(|| Error::with_message(Errno::NotFound, "vm not found"))?;
    let mut ioapic = vm.ioapic.lock();

    if insn.is_read {
        let (value, ok) = emulate_ioapic_read(&ioapic, offset);
        if !ok {
            return Ok(false);
        }
        let mut state = vcpu.state.lock();
        write_gpr(&mut state.regs, insn.reg, insn.size, value);
        return Ok(true);
    }

    let value = {
        let state = vcpu.state.lock();
        read_gpr(&state.regs, insn.reg, insn.size)
    };
    if !emulate_ioapic_write(&mut ioapic, offset, value) {
        return Ok(false);
    }
    Ok(true)
}

fn decode_mmio_instruction(bytes: &[u8; MAX_INSN_LENGTH]) -> Option<MmioInstruction> {
    let mut ptr = 0usize;
    let mut op_size_16 = false;
    let mut rex = 0u8;
    let mut rex_w = false;

    while ptr < bytes.len() {
        let byte = bytes[ptr];
        if byte == 0x66 {
            op_size_16 = true;
            ptr += 1;
        } else if (byte & 0xf0) == 0x40 {
            rex = byte;
            rex_w = (byte & 0x08) != 0;
            ptr += 1;
        } else {
            break;
        }
    }

    let opcode = *bytes.get(ptr)?;
    ptr += 1;

    let (is_read, size) = match opcode {
        0x88 => (false, 1),
        0x8a => (true, 1),
        0x89 => (
            false,
            if rex_w {
                8
            } else if op_size_16 {
                2
            } else {
                4
            },
        ),
        0x8b => (
            true,
            if rex_w {
                8
            } else if op_size_16 {
                2
            } else {
                4
            },
        ),
        _ => return None,
    };

    let modrm = *bytes.get(ptr)?;
    let mut reg = (modrm >> 3) & 0x7;
    if (rex & 0x04) != 0 {
        reg |= 0x8;
    }

    Some(MmioInstruction { is_read, size, reg })
}
