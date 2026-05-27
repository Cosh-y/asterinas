use core::fmt;

use ostd::arch::virt::*;
use x86_64::registers::control::Cr0Flags;

use super::{
    emulate::apic::{
        emulate_ioapic_read, emulate_ioapic_write, emulate_lapic_read, emulate_lapic_write,
        LapicWriteEffect, Icr, ioapic_eoi, IOAPIC_BASE, IOAPIC_SIZE, LAPIC_BASE, LAPIC_SIZE,
    },
    emulate::cr::emulate_cr_access,
    emulate::msr::emulate_msrrw,
    emulate::timer::{start_apic_timer_deadline_locked, start_apic_timer_locked},
    error::*,
    vcpu::{Vcpu, VcpuState, VcpuMsrs},
};
use crate::interrupt::{
    clear_interrupt_shadow_after_hlt, handle_external_interrupt, handle_interrupt_window,
    inject_gp_fault, inject_pending_exception,
};

const MAX_INSN_LENGTH: usize = 15;
const PAUSE_INSN_LENGTH: usize = 2;

const VMX_EXIT_REASON_IO_INSTRUCTION: u32 = 30;
const VMX_EXIT_REASON_EPT_VIOLATION: u32 = 48;

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
    len: usize,
}

struct HexOption<T>(Option<T>);

impl<T> fmt::Display for HexOption<T>
where
    T: fmt::LowerHex + Copy,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(value) => write!(formatter, "Some({value:#x})"),
            None => formatter.write_str("None"),
        }
    }
}

fn hex_option<T>(value: Option<T>) -> HexOption<T>
where
    T: fmt::LowerHex + Copy,
{
    HexOption(value)
}

fn log_vmentry_guest_state(vcpu: &Vcpu, exit_info: &VmxExitInfo) {
    let vm_instruction_error = VmcsReadOnly32::VM_INSTRUCTION_ERROR.read().ok();
    let guest_rsp = VmcsGuestNW::RSP.read().ok();
    let guest_rflags = VmcsGuestNW::RFLAGS.read().ok();
    let guest_cr0 = VmcsGuestNW::CR0.read().ok();
    let guest_cr3 = VmcsGuestNW::CR3.read().ok();
    let guest_cr4 = VmcsGuestNW::CR4.read().ok();
    let guest_efer = VmcsGuest64::IA32_EFER.read().ok();
    let cs_selector = VmcsGuest16::CS_SELECTOR.read().ok();
    let ss_selector = VmcsGuest16::SS_SELECTOR.read().ok();
    let tr_selector = VmcsGuest16::TR_SELECTOR.read().ok();
    let ldtr_selector = VmcsGuest16::LDTR_SELECTOR.read().ok();
    let cs_ar = VmcsGuest32::CS_ACCESS_RIGHTS.read().ok();
    let ss_ar = VmcsGuest32::SS_ACCESS_RIGHTS.read().ok();
    let tr_ar = VmcsGuest32::TR_ACCESS_RIGHTS.read().ok();
    let ldtr_ar = VmcsGuest32::LDTR_ACCESS_RIGHTS.read().ok();
    let exit_reason_name = VmxExitReason::try_from(exit_info.exit_reason).ok();

    log::error!(
        "rustshyper: VM-entry failure for vcpu {}: exit_reason={:#x} ({:?}), vm_instruction_error={}",
        vcpu.id(),
        exit_info.exit_reason,
        exit_reason_name,
        hex_option(vm_instruction_error),
    );
    log::error!(
        "rustshyper:   entry: rip={:#x}, rsp={}, rflags={}, qualification={:#x}",
        exit_info.guest_rip,
        hex_option(guest_rsp),
        hex_option(guest_rflags),
        exit_info.exit_qualification
    );
    log::error!(
        "rustshyper:   control: cr0={}, cr3={}, cr4={}, efer={}",
        hex_option(guest_cr0),
        hex_option(guest_cr3),
        hex_option(guest_cr4),
        hex_option(guest_efer),
    );
    log::error!(
        "rustshyper:   segments: cs={}/{}, ss={}/{}, tr={}/{}, ldtr={}/{}",
        hex_option(cs_selector),
        hex_option(cs_ar),
        hex_option(ss_selector),
        hex_option(ss_ar),
        hex_option(tr_selector),
        hex_option(tr_ar),
        hex_option(ldtr_selector),
        hex_option(ldtr_ar),
    );
}

pub fn vmexit_handler(vcpu: &Vcpu, exit_info: &VmxExitInfo) -> Result<Option<RunStateMessage>> {
    if exit_info.entry_failure {
        log_vmentry_guest_state(vcpu, exit_info);
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
        Ok(VmxExitReason::TRIPLE_FAULT) => Ok(Some(build_run_state(exit_info))),
        Ok(VmxExitReason::HLT) => {
            if vcpu.wait_for_hlt_wakeup()? {
                // log::error!("Guest HLT. But wake up from host kernel due to event injection or interrupt window");
                clear_interrupt_shadow_after_hlt()?;
                advance_guest_rip(vcpu)?;
                Ok(None)
            } else {
                // log::error!("Guest HLT. Can't wake up from host kernel. Returning to userspace to wait for wakeup event.");
                Ok(Some(build_run_state(exit_info)))
            }
        }
        Ok(VmxExitReason::INTERRUPT_WINDOW) => {
            handle_interrupt_window()?;
            Ok(None)
        }
        Ok(VmxExitReason::CPUID) => {
            vcpu.emulate_cpuid()?;
            advance_guest_rip(vcpu)?;
            Ok(None)
        }
        Ok(VmxExitReason::VMCALL) => Ok(Some(build_run_state(exit_info))),
        Ok(VmxExitReason::CR_ACCESS) => {
            emulate_cr_access(vcpu)?;
            advance_guest_rip(vcpu)?;
            Ok(None)
        }
        Ok(VmxExitReason::MSR_READ) => {
            emulate_msrrw(vcpu, false)?;
            advance_guest_rip(vcpu)?;
            Ok(None)
        }
        Ok(VmxExitReason::MSR_WRITE) => {
            emulate_msrrw(vcpu, true)?;
            advance_guest_rip(vcpu)?;
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
            Ok(Some(build_run_state(exit_info)))
        }
        Ok(VmxExitReason::PAUSE_INSTRUCTION) => {
            // advance_guest_rip(vcpu)?;
            vcpu.state.lock().arch_mut().advance_rip(PAUSE_INSN_LENGTH as u64);
            Ok(Some(build_run_state(exit_info)))
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

fn instruction_len() -> core::result::Result<u32, ostd::Error> {
    VmcsReadOnly32::VMEXIT_INSTRUCTION_LEN.read()
}

fn queue_gp_fault(vcpu: &Vcpu) -> Result<()> {
    let mut state = vcpu.state.lock();
    inject_gp_fault(&mut state.exception);
    let mut perf = [0u32; 32];
    inject_pending_exception(&mut state.exception, &mut perf)?;
    Ok(())
}

// TODO: why return bool here?
fn emulate_apic_mmio(vcpu: &Vcpu, fault_gpa: u64) -> Result<bool> {
    // log::error!("Guest access to APIC MMIO at GPA {:#x}", fault_gpa);
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

    vcpu.state.lock().arch_mut().advance_rip(insn.len as u64);
    Ok(true)
}

fn emulate_lapic_mmio(vcpu: &Vcpu, fault_gpa: u64, insn: MmioInstruction) -> Result<bool> {
    // log::error!("Guest access to LAPIC MMIO at GPA {:#x}", fault_gpa);
    let offset = fault_gpa - LAPIC_BASE;
    if insn.is_read {
        let mut state = vcpu.state.lock();
        let (value, ok) = emulate_lapic_read(&state.lapic, &state.apic_timer, &state.tsc, offset);
        if !ok {
            return Ok(false);
        }
        
        let gpr_index = map_instruction_gpr_index_to_common_gpr_index(insn.reg);
        state.arch_mut().set_gpr(gpr_index, insn.size, value);

        return Ok(true);
    }

    let value = {
        let mut state = vcpu.state.lock();
        let gpr_index = map_instruction_gpr_index_to_common_gpr_index(insn.reg);
        state.arch_mut().gpr(gpr_index)
    };
    
    let vm = vcpu
        .vm
        .upgrade()
        .ok_or_else(|| Error::with_message(Errno::NotFound, "vm not found"))?;
    
    let effect = {
        let mut state = vcpu.state.lock();
        let VcpuState { lapic, apic_timer, .. } = &mut *state;
        emulate_lapic_write(lapic, apic_timer, offset, value)
    };
    
    match effect {
        Some(LapicWriteEffect::Eoi { isr_vec }) => {
            ioapic_eoi(&mut vm.ioapic.lock(), isr_vec);
        }
        Some(LapicWriteEffect::Icr(icr)) => {
            vm.deliver_lapic_icr(vcpu.id(), icr)?;
        },
        Some(LapicWriteEffect::StartTimer) => {
            let mut state = vcpu.state.lock();
            start_apic_timer_locked(&mut state);
        }
        Some(LapicWriteEffect::StartTimerDeadline) => {
            let mut state = vcpu.state.lock();
            start_apic_timer_deadline_locked(&mut state);
        }
        None => {}
    }
    Ok(true)
}

fn emulate_ioapic_mmio(vcpu: &Vcpu, fault_gpa: u64, insn: MmioInstruction) -> Result<bool> {
    // log::error!("Guest access to IOAPIC MMIO at GPA {:#x}", fault_gpa);
    let offset = fault_gpa - IOAPIC_BASE;
    // log::error!("IOAPIC MMIO access with offset {:#x}, IOAPIC_BASE {:#x}", offset, IOAPIC_BASE);
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
        let gpr_index = map_instruction_gpr_index_to_common_gpr_index(insn.reg);
        state.arch_mut().set_gpr(gpr_index, insn.size, value);
        
        return Ok(true);
    }

    let value = {
        let state = vcpu.state.lock();
        let gpr_index = map_instruction_gpr_index_to_common_gpr_index(insn.reg);
        state.arch().gpr(gpr_index)
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
        match byte {
            0x66 => op_size_16 = true,
            0x67 | 0x2e | 0x36 | 0x3e | 0x26 | 0x64 | 0x65 | 0xf0 | 0xf2 | 0xf3 => {}
            b if (b & 0xf0) == 0x40 => {
                rex = b;
                rex_w = (b & 0x08) != 0;
            }
            _ => break,
        }
        ptr += 1;
    }

    let opcode = *bytes.get(ptr)?;
    ptr += 1;

    // MOV Opcode
    // Volume 2, 4.3 Instructions(M-U)
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

    // modrm
    // bit: 7 6 | 5 4 3 | 2 1 0
    //      mod |  reg  |  r/m
    // mod: r/m is reg or memory
    //      11    for reg
    //      other for memory
    // Volume 2, 2.1 Instruction Format...
    let modrm = *bytes.get(ptr)?;
    ptr += 1;

    let mode = modrm >> 6;
    let rm = modrm & 0x7;
    if mode == 0b11 {
        return None;
    }
    if rm == 0x4 {
        let sib = *bytes.get(ptr)?;
        ptr += 1;
        let base = sib & 0x7;
        if mode == 0 && base == 0x5 {
            ptr += 4;
        }
    } else if mode == 0 && rm == 0x5 {
        ptr += 4;
    }
    match mode {
        0 => {}
        1 => ptr += 1,
        2 => ptr += 4,
        _ => return None,
    }
    if ptr > MAX_INSN_LENGTH {
        return None;
    }

    let mut reg = (modrm >> 3) & 0x7;
    if (rex & 0x04) != 0 {
        reg |= 0x8;
    }

    Some(MmioInstruction {
        is_read,
        size,
        reg,
        len: ptr,
    })
}

fn map_instruction_gpr_index_to_common_gpr_index(index: u8) -> u8 {
    // The encode method of gpr index in ModRM:
    // ???
    match index {
        0 => 0,
        1 => 2,
        2 => 3,
        3 => 1,
        4 => 7,
        5 => 6,
        6 => 4,
        7 => 5,
        other => other,
    }
}

pub(crate) fn advance_guest_rip(vcpu: &Vcpu) -> Result<()> {
    let len = instruction_len().map_err(Error::from)? as usize;
    vcpu.state.lock().arch_mut().advance_rip(len as u64);
    Ok(())
}
