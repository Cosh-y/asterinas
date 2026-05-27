use x86::msr::*;
use x86_64::registers::control::Cr0Flags;
use x86_64::registers::model_specific::EferFlags;

use ostd::arch::virt::*;

use crate::emulate::apic::LAPIC_BASE;
use crate::emulate::timer::start_apic_timer_deadline_locked;
use crate::error::*;
use crate::vcpu::{Vcpu, VcpuState, VcpuArchState};

pub(crate) fn emulate_msrrw(vcpu: &Vcpu, is_write: bool) -> Result<()> {
    let mut state = vcpu.state.lock();
    let msr_index = state.arch().gpr(2) as u32;
    let guest_rip = VmcsGuestNW::RIP.read().map_err(Error::from)? as u64;

    if is_write {
        let msr_value = (state.arch().gpr(0) as u32 as u64) | ((state.arch().gpr(3) as u32 as u64) << 32);

        match msr_index {
            TSC => {
                let raw_tsc = ostd::arch::read_tsc();
                state.tsc.tsc_physical = msr_value;
                state.tsc.tsc_offset = msr_value.wrapping_sub(raw_tsc);
            }
            IA32_TSC_ADJUST => {
                let delta = msr_value.wrapping_sub(state.arch().msr(IA32_TSC_ADJUST));
                state.arch_mut().set_msr(IA32_TSC_ADJUST, msr_value);
                state.tsc.tsc_physical = state.tsc.tsc_physical.wrapping_add(delta);
                state.tsc.tsc_offset = state.tsc.tsc_offset.wrapping_add(delta);
            }
            IA32_APIC_BASE => {
                state.arch_mut().set_msr(IA32_APIC_BASE, sanitize_apic_base(msr_value));
            }
            IA32_EFER => {
                let mut guest_efer = msr_value;
                let guest_cr0 = state.arch().cr0();
                if (guest_efer & EferFlags::LONG_MODE_ENABLE.bits()) != 0
                    && (guest_cr0 & Cr0Flags::PAGING.bits()) != 0 {
                    guest_efer |= EferFlags::LONG_MODE_ACTIVE.bits();
                } else {
                    guest_efer &= !EferFlags::LONG_MODE_ACTIVE.bits();
                }
                state.arch_mut().set_msr(IA32_EFER, guest_efer);
            }
            IA32_BIOS_SIGN_ID => {}
            IA32_MISC_ENABLE => {}
            IA32_TSC_DEADLINE => {
                state.arch_mut().set_msr(IA32_TSC_DEADLINE, msr_value);
                start_apic_timer_deadline_locked(&mut state);
                return Ok(());
            }
            _ => {
                state.arch_mut().set_msr(msr_index, msr_value);
            }
        }

        return Ok(());
    }

    // is read
    let msr_value = match msr_index {
        TSC => state.tsc.tsc_physical,
        _   => state.arch().msr(msr_index),
    };

    state.arch_mut().set_gpr(0, 8, msr_value as u32 as u64);
    state.arch_mut().set_gpr(3, 8, msr_value >> 32);
    Ok(())
}

fn sanitize_apic_base(value: u64) -> u64 {
    const APIC_BASE_BSP: u64 = 1 << 8;
    const APIC_BASE_ENABLE: u64 = 1 << 11;

    LAPIC_BASE | APIC_BASE_ENABLE | (value & APIC_BASE_BSP)
}
