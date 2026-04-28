use core::fmt;

use ostd::arch::virt::*;

use super::{
    emulate::apic::{
        emulate_ioapic_read, emulate_ioapic_write, emulate_lapic_read, emulate_lapic_write,
        LapicWriteEffect, IOAPIC_BASE, IOAPIC_SIZE, LAPIC_BASE, LAPIC_SIZE,
    },
    error::*,
    vm::{
        guest_cr4_read_shadow, guest_ia32e_mode_active, sanitize_guest_cr0, sanitize_guest_cr4,
        sanitize_guest_efer, GuestMsrState, Vcpu,
    },
};
use crate::interrupt::{
    clear_interrupt_shadow_after_hlt, handle_external_interrupt, handle_interrupt_window,
    inject_gp_fault, inject_pending_exception,
};

const MAX_INSN_LENGTH: usize = 15;
const PAUSE_INSN_LENGTH: usize = 2;

const VMX_EXIT_REASON_IO_INSTRUCTION: u32 = 30;
const VMX_EXIT_REASON_EPT_VIOLATION: u32 = 48;

const X86_CR0_PE: u64 = 1 << 0;
const X86_CR0_PG: u64 = 1 << 31;
const MSR_IA32_TSC: u32 = 0x0000_0010;
const MSR_IA32_APIC_BASE: u32 = 0x0000_001B;
const MSR_IA32_BIOS_SIGN_ID: u32 = 0x0000_008B;
const MSR_IA32_XAPIC_DISABLE_STATUS: u32 = 0x0000_00BD;
const MSR_IA32_MTRRCAP: u32 = 0x0000_00FE;
const MSR_IA32_ARCH_CAPABILITIES: u32 = 0x0000_010A;
const MSR_IA32_MCG_CAP: u32 = 0x0000_0179;
const MSR_IA32_SYSENTER_CS: u32 = 0x0000_0174;
const MSR_IA32_SYSENTER_ESP: u32 = 0x0000_0175;
const MSR_IA32_SYSENTER_EIP: u32 = 0x0000_0176;
const MSR_IA32_MISC_ENABLE: u32 = 0x0000_01A0;
const MSR_IA32_PAT: u32 = 0x0000_0277;
const MSR_IA32_MTRR_DEF_TYPE: u32 = 0x0000_02FF;
const MSR_EFER: u32 = 0xC000_0080;
const MSR_STAR: u32 = 0xC000_0081;
const MSR_LSTAR: u32 = 0xC000_0082;
const MSR_CSTAR: u32 = 0xC000_0083;
const MSR_SYSCALL_MASK: u32 = 0xC000_0084;
const MSR_FS_BASE: u32 = 0xC000_0100;
const MSR_GS_BASE: u32 = 0xC000_0101;
const MSR_KERNEL_GS_BASE: u32 = 0xC000_0102;
const MSR_IA32_TSC_AUX: u32 = 0xC000_0103;
const MSR_IA32_TSC_DEADLINE: u32 = 0x0000_06E0;
const MSR_X2APIC_BASE: u32 = 0x0000_0800;
const MSR_X2APIC_END: u32 = 0x0000_08FF;
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
            if vcpu.poll_expired_lapic_timer()? {
                clear_interrupt_shadow_after_hlt()?;
                advance_guest_rip()?;
                Ok(None)
            } else {
                Ok(Some(build_run_state(exit_info)))
            }
        }
        Ok(VmxExitReason::INTERRUPT_WINDOW) => {
            handle_interrupt_window()?;
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
            Ok(Some(build_run_state(exit_info)))
        }
        Ok(VmxExitReason::PAUSE_INSTRUCTION) => {
            advance_rip(PAUSE_INSN_LENGTH).map_err(Error::from)?;
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

pub(crate) fn advance_guest_rip() -> Result<()> {
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
                2 => vcpu.set_guest_cr2(value),
                3 => VmcsGuestNW::CR3.write(value as usize).map_err(Error::from)?,
                4 => emulate_cr4_write(value)?,
                other => log::warn!("rustshyper: ignoring guest write to CR{}", other),
            }
        }
        1 => {
            let value = match cr_index {
                0 => VmcsControlNW::CR0_READ_SHADOW.read().map_err(Error::from)? as u64,
                2 => vcpu.guest_cr2(),
                3 => VmcsGuestNW::CR3.read().map_err(Error::from)? as u64,
                4 => VmcsControlNW::CR4_READ_SHADOW.read().map_err(Error::from)? as u64,
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
    let actual_value = sanitize_guest_cr0(value);
    let guest_efer = vcpu.state.lock().msrs.efer;

    VmcsControlNW::CR0_READ_SHADOW
        .write(shadow_value as usize)
        .map_err(Error::from)?;
    VmcsGuestNW::CR0
        .write(actual_value as usize)
        .map_err(Error::from)?;
    sync_guest_efer_state(guest_efer, value)?;
    Ok(())
}

fn emulate_cr4_write(value: u64) -> Result<()> {
    let actual_value = sanitize_guest_cr4(value);
    let shadow_value = guest_cr4_read_shadow(value);

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
    let guest_rip = VmcsGuestNW::RIP.read().map_err(Error::from)? as u64;

    if is_write {
        let msr_value = (state.regs.rax as u32 as u64) | ((state.regs.rdx as u32 as u64) << 32);
        if is_x2apic_msr(msr_index) {
            drop(state);
            emulate_x2apic_msr_write(vcpu, msr_index, msr_value)?;
            return Ok(());
        }

        match msr_index {
            MSR_IA32_TSC => {
                state.tsc.tsc_physical = msr_value;
                state.tsc.tsc_offset = msr_value.wrapping_sub(ostd::arch::read_tsc());
            }
            MSR_IA32_APIC_BASE => {
                state.msrs.apic_base = sanitize_apic_base(msr_value);
            }
            MSR_IA32_BIOS_SIGN_ID => {}
            MSR_IA32_MISC_ENABLE => {}
            MSR_IA32_SYSENTER_CS => state.msrs.sysenter_cs = msr_value,
            MSR_IA32_SYSENTER_ESP => state.msrs.sysenter_esp = msr_value,
            MSR_IA32_SYSENTER_EIP => state.msrs.sysenter_eip = msr_value,
            MSR_EFER => state.msrs.efer = msr_value,
            MSR_IA32_PAT => state.msrs.pat = msr_value,
            MSR_FS_BASE => state.msrs.fs_base = msr_value,
            MSR_GS_BASE => state.msrs.gs_base = msr_value,
            MSR_KERNEL_GS_BASE => state.msrs.kernel_gs_base = msr_value,
            MSR_IA32_TSC_AUX => state.msrs.tsc_aux = msr_value,
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
                log::warn!(
                    "rustshyper: unrecognized WRMSR rip={:#x} msr={:#x} value={:#x}",
                    guest_rip,
                    msr_index,
                    msr_value
                );
                inject_gp_fault(&mut state.exception);
                let mut perf = [0u32; 32];
                inject_pending_exception(&mut state.exception, &mut perf)?;
                return Ok(());
            }
        }

        update_guest_msr_vmcs(&state.msrs)?;
        return Ok(());
    }

    if is_x2apic_msr(msr_index) {
        drop(state);
        let Some(msr_value) = emulate_x2apic_msr_read(vcpu, msr_index) else {
            log::warn!(
                "rustshyper: unsupported x2APIC RDMSR rip={:#x} msr={:#x}",
                guest_rip,
                msr_index,
            );
            queue_gp_fault(vcpu)?;
            return Ok(());
        };
        let mut state = vcpu.state.lock();
        state.regs.rax = msr_value as u32 as u64;
        state.regs.rdx = msr_value >> 32;
        return Ok(());
    }

    let msr_value = match msr_index {
        MSR_IA32_TSC => state.tsc.tsc_physical,
        MSR_IA32_APIC_BASE => state.msrs.apic_base,
        MSR_IA32_BIOS_SIGN_ID => 0,
        MSR_IA32_XAPIC_DISABLE_STATUS => 0,
        MSR_IA32_MTRRCAP => 0,
        MSR_IA32_ARCH_CAPABILITIES => 0,
        MSR_IA32_MCG_CAP => 0,
        MSR_IA32_MISC_ENABLE => 0,
        MSR_IA32_SYSENTER_CS => state.msrs.sysenter_cs,
        MSR_IA32_SYSENTER_ESP => state.msrs.sysenter_esp,
        MSR_IA32_SYSENTER_EIP => state.msrs.sysenter_eip,
        MSR_EFER => state.msrs.efer,
        MSR_IA32_PAT => state.msrs.pat,
        MSR_IA32_MTRR_DEF_TYPE => 0,
        MSR_FS_BASE => state.msrs.fs_base,
        MSR_GS_BASE => state.msrs.gs_base,
        MSR_KERNEL_GS_BASE => state.msrs.kernel_gs_base,
        MSR_IA32_TSC_AUX => state.msrs.tsc_aux,
        MSR_STAR => state.msrs.star,
        MSR_LSTAR => state.msrs.lstar,
        MSR_CSTAR => state.msrs.cstar,
        MSR_SYSCALL_MASK => state.msrs.syscall_mask,
        MSR_AMD64_DE_CFG => 0,
        MSR_IA32_TSC_DEADLINE => state.msrs.tsc_deadline,
        _ => {
            log::warn!(
                "rustshyper: unrecognized RDMSR rip={:#x} msr={:#x} rax={:#x} rdx={:#x}",
                guest_rip,
                msr_index,
                state.regs.rax,
                state.regs.rdx
            );
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
    let guest_cr0 = VmcsControlNW::CR0_READ_SHADOW.read().map_err(Error::from)? as u64;

    VmcsGuest64::IA32_PAT.write(msrs.pat).map_err(Error::from)?;
    VmcsGuest64::IA32_EFER
        .write(sanitize_guest_efer(msrs.efer, guest_cr0))
        .map_err(Error::from)?;
    VmcsGuestNW::FS_BASE
        .write(msrs.fs_base as usize)
        .map_err(Error::from)?;
    VmcsGuestNW::GS_BASE
        .write(msrs.gs_base as usize)
        .map_err(Error::from)?;
    VmcsGuest32::IA32_SYSENTER_CS
        .write(msrs.sysenter_cs as u32)
        .map_err(Error::from)?;
    VmcsGuestNW::IA32_SYSENTER_ESP
        .write(msrs.sysenter_esp as usize)
        .map_err(Error::from)?;
    VmcsGuestNW::IA32_SYSENTER_EIP
        .write(msrs.sysenter_eip as usize)
        .map_err(Error::from)?;
    sync_guest_ia32e_mode_control(msrs.efer, guest_cr0)?;
    Ok(())
}

fn sync_guest_efer_state(guest_efer: u64, guest_cr0: u64) -> Result<()> {
    VmcsGuest64::IA32_EFER
        .write(sanitize_guest_efer(guest_efer, guest_cr0))
        .map_err(Error::from)?;
    sync_guest_ia32e_mode_control(guest_efer, guest_cr0)
}

fn sync_guest_ia32e_mode_control(guest_efer: u64, guest_cr0: u64) -> Result<()> {
    let mut entry = VmcsControl32::VMENTRY_CONTROLS
        .read()
        .map_err(Error::from)?;
    let ia32e_mode_guest = 1 << 9;

    if guest_ia32e_mode_active(guest_efer, guest_cr0) {
        entry |= ia32e_mode_guest;
    } else {
        entry &= !ia32e_mode_guest;
    }

    VmcsControl32::VMENTRY_CONTROLS
        .write(entry)
        .map_err(Error::from)?;
    Ok(())
}

fn sanitize_apic_base(value: u64) -> u64 {
    const APIC_BASE_BSP: u64 = 1 << 8;
    const APIC_BASE_X2APIC: u64 = 1 << 10;
    const APIC_BASE_ENABLE: u64 = 1 << 11;

    LAPIC_BASE | APIC_BASE_ENABLE | (value & (APIC_BASE_BSP | APIC_BASE_X2APIC))
}

fn is_x2apic_msr(msr_index: u32) -> bool {
    (MSR_X2APIC_BASE..=MSR_X2APIC_END).contains(&msr_index)
}

fn x2apic_msr_offset(msr_index: u32) -> u64 {
    u64::from(msr_index - MSR_X2APIC_BASE) << 4
}

fn emulate_x2apic_msr_read(vcpu: &Vcpu, msr_index: u32) -> Option<u64> {
    let offset = x2apic_msr_offset(msr_index);
    let state = vcpu.state.lock();

    if offset == 0x20 {
        return Some(state.lapic.id as u64);
    }

    let (value, ok) = emulate_lapic_read(&state.lapic, &state.apic_timer, &state.tsc, offset);
    ok.then_some(value)
}

fn emulate_x2apic_msr_write(vcpu: &Vcpu, msr_index: u32, value: u64) -> Result<()> {
    let offset = x2apic_msr_offset(msr_index);
    let vm = vcpu
        .vm
        .upgrade()
        .ok_or_else(|| Error::with_message(Errno::NotFound, "vm not found"))?;
    let mut ioapic = vm.ioapic.lock();

    let effect = {
        let mut state = vcpu.state.lock();
        let (effect, ok) = {
            let super::vm::VcpuState {
                lapic, apic_timer, ..
            } = &mut *state;
            emulate_lapic_write(lapic, apic_timer, &mut ioapic, offset, value)
        };
        if !ok {
            inject_gp_fault(&mut state.exception);
            let mut perf = [0u32; 32];
            inject_pending_exception(&mut state.exception, &mut perf)?;
            return Ok(());
        }
        effect
    };

    drop(ioapic);
    match effect {
        LapicWriteEffect::StartTimer | LapicWriteEffect::StartTimerDeadline => {
            vcpu.handle_lapic_timer_write_effect(effect)?;
        }
        LapicWriteEffect::None => {}
    }
    Ok(())
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

    advance_rip(insn.len).map_err(Error::from)?;
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
        let super::vm::VcpuState {
            lapic, apic_timer, ..
        } = &mut *state;
        let (effect, ok) = emulate_lapic_write(lapic, apic_timer, &mut ioapic, offset, value);
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
