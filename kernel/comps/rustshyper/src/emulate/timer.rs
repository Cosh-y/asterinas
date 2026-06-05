use ostd::arch::{read_tsc, tsc_freq};

use crate::emulate::apic::{lapic_set_irr, ApicTimer, TscState};
use crate::emulate::apic::lapic_check_pending_vector;
use crate::error::*;
use crate::vcpu::{Vcpu, VcpuState};

pub const VMX_PREEMPTION_TIMER_MULTIPLIER_FALLBACK: u8 = 0;
pub const VMX_PREEMPTION_TIMER_POLL_VALUE: u32 = 50_000;

pub(crate) fn handle_preemption_timer_expire(vcpu: &Vcpu) -> Result<()> {
    let mut state = vcpu.state.lock();
    if !state.tsc.activated {
        return Ok(());
    }
    if state.tsc.ddl_physical > state.tsc.tsc_physical {
        return Ok(());
    }

    expire_lapic_timer_locked(&mut state);
    Ok(())
}

/// Try waiting in the kernel for an interrupt that can break out of hlt.
pub(crate) fn wait_for_hlt_wakeup(vcpu: &Vcpu) -> Result<bool> {
    {
        vcpu.refresh_guest_tsc();
        let mut state = vcpu.state.lock();
        if state.interrupt.pending || lapic_check_pending_vector(&state.lapic).is_some() {
            return Ok(true);
        }
        if !state.tsc.activated {
            return Ok(false);
        }
        if state.tsc.ddl_physical <= state.tsc.tsc_physical {
            expire_lapic_timer_locked(&mut state);
            return Ok(true);
        }
    }

    let wait_started_tsc = read_tsc();
    let max_wait_ticks = hlt_wait_max_ticks();
    loop {
        vcpu.refresh_guest_tsc();
        let mut state = vcpu.state.lock();
        if state.interrupt.pending || lapic_check_pending_vector(&state.lapic).is_some() {
            return Ok(true);
        }
        if state.tsc.activated && state.tsc.ddl_physical <= state.tsc.tsc_physical {
            expire_lapic_timer_locked(&mut state);
            return Ok(true);
        }
        debug_assert!(state.tsc.activated);
        let raw_tsc = read_tsc();
        if raw_tsc.saturating_sub(wait_started_tsc) >= max_wait_ticks {
            return Ok(false);
        }
        drop(state);

        core::hint::spin_loop();
    }
}

/// Compute the next deadline for one-shot / periodic APIC timer modes.
pub(crate) fn lapic_timer_deadline(timer: &ApicTimer, tsc: &TscState) -> Option<u64> {
    if timer.initial_count == 0 || is_deadline_mode(timer) {
        return None;
    }
    Some(
        tsc.tsc_physical
            .wrapping_add((timer.initial_count as u64) << timer.divide_shift),
    )
}

/// Compute the deadline for TSC-deadline mode.
pub(crate) fn lapic_timer_deadline_tsc(timer: &ApicTimer, tsc_deadline: u64) -> Option<u64> {
    if tsc_deadline == 0 || !is_deadline_mode(timer) {
        return None;
    }
    Some(tsc_deadline)
}

/// 在 tsc_physical 超过 ddl_physical 时调用
/// 通过 lapic_set_irr 设置中断，并根据 timer mode 再次设置 timer
pub(crate) fn expire_lapic_timer_locked(state: &mut VcpuState) {
    let tsc = state.tsc;
    let lapic = &mut state.lapic;
    let timer = &state.apic_timer;
    if (timer.lvt_timer_bits & (1 << 16)) == 0 {
        let vector = (timer.lvt_timer_bits & 0xFF) as u8;
        lapic_set_irr(lapic, vector);
    }

    if is_periodic_mode(timer) {
        let next_deadline = tsc
            .tsc_physical
            .wrapping_add((timer.initial_count as u64) << timer.divide_shift);
        timer_activate_locked(state, next_deadline);
    } else {
        timer_deactivate_locked(state);
    }
}

/// 关闭时钟中断计时器
pub(crate) fn timer_deactivate_locked(state: &mut VcpuState) {
    state.tsc.activated = false;
    state.tsc.ddl_physical = 0;
}

/// 以 One-shot/Periodic 模式设置 lapic timer
pub(crate) fn start_apic_timer_locked(state: &mut VcpuState) {
    if let Some(deadline) = lapic_timer_deadline(&state.apic_timer, &state.tsc) {
        timer_activate_locked(state, deadline);
    } else {
        timer_deactivate_locked(state);
    }
}

/// 以 tsc-deadline 模式设置 lapic timer
/// tsc-deadline 模式即以 IA32_TSC_DEADLINE MSR 作为触发时钟中断的绝对时间
pub(crate) fn start_apic_timer_deadline_locked(state: &mut VcpuState) {
    use x86::msr::*;
    if let Some(deadline) = lapic_timer_deadline_tsc(&state.apic_timer, state.arch().msr(IA32_TSC_DEADLINE)) {
        timer_activate_locked(state, deadline);
    } else {
        timer_deactivate_locked(state);
    }
}

/// 根据 tsc_physical 到 ddl_physical 的差距计算应写入 preemption timer 中的值
/// 即到下次发生 preemption timer exit 的时间间隔
pub(crate) fn compute_preemption_timer_value(state: &VcpuState) -> u32 {
    if !state.tsc.activated {
        return VMX_PREEMPTION_TIMER_POLL_VALUE;
    }

    let ticks = state
        .tsc
        .ddl_physical
        .saturating_sub(state.tsc.tsc_physical);

    if state.tsc.multiplier == VMX_PREEMPTION_TIMER_MULTIPLIER_FALLBACK {
        return ticks.min(u64::from(VMX_PREEMPTION_TIMER_POLL_VALUE)).max(1) as u32;
    }

    let shifted = ticks >> state.tsc.multiplier;
    let shifted = if shifted == 0 { 1 } else { shifted };
    shifted.min(u64::from(VMX_PREEMPTION_TIMER_POLL_VALUE)) as u32
}


const HLT_WAIT_MAX_TSC_FREQ_DIVISOR: u64 = 100;
const HLT_WAIT_MAX_FALLBACK_TICKS: u64 = 25_000_000;

fn hlt_wait_max_ticks() -> u64 {
    match tsc_freq() {
        0 => HLT_WAIT_MAX_FALLBACK_TICKS,
        freq => (freq / HLT_WAIT_MAX_TSC_FREQ_DIVISOR).max(1),
    }
}

fn is_periodic_mode(timer: &ApicTimer) -> bool {
    ((timer.lvt_timer_bits >> 17) & 0b11) == 1
}

fn is_deadline_mode(timer: &ApicTimer) -> bool {
    ((timer.lvt_timer_bits >> 17) & 0b11) == 2
}

/// 查看当前 tsc_physical 是否小于 deadline，若是，则设置 deadline 到 ddl_physical 中
/// 否则调用 expire_lapic_timer_locked
fn timer_activate_locked(state: &mut VcpuState, deadline_ticks: u64) {
    state.tsc.activated = true;
    if deadline_ticks > state.tsc.tsc_physical {
        state.tsc.ddl_physical = deadline_ticks;
    } else {
        expire_lapic_timer_locked(state);
    }
}
