//! Emulated LAPIC and IOAPIC device for guest VMs.
//!
//! Use pure software emulate for now.
use ostd::arch::vm::{GuestInterrupt, GuestTimerInstant};

use super::ioctl::LapicState;
use crate::prelude::*;

pub const IOAPIC_NUM_PINS: usize = 24;

const APIC_MODE_EXTINT: u32 = 0x7;
const APIC_LVT_VECTOR_MASK: u32 = 0xFF;
const APIC_LVT_DELIVERY_MODE_MASK: u32 = 0x700;
const APIC_LVT_SEND_PENDING: u32 = 1 << 12;
const APIC_LVT_INPUT_POLARITY: u32 = 1 << 13;
const APIC_LVT_REMOTE_IRR: u32 = 1 << 14;
const APIC_LVT_LEVEL_TRIGGER: u32 = 1 << 15;
const APIC_LVT_MASKED: u32 = 1 << 16;
const APIC_LINT_MASK: u32 = APIC_LVT_VECTOR_MASK
    | APIC_LVT_DELIVERY_MODE_MASK
    | APIC_LVT_SEND_PENDING
    | APIC_LVT_INPUT_POLARITY
    | APIC_LVT_REMOTE_IRR
    | APIC_LVT_LEVEL_TRIGGER
    | APIC_LVT_MASKED;
const APIC_TIMER_MODE_ONESHOT: u32 = 0b00;
const APIC_TIMER_MODE_PERIODIC: u32 = 0b01;
const APIC_TIMER_MODE_TSC_DEADLINE: u32 = 0b10;

/// Local APIC state.
#[derive(Debug)]
pub struct Lapic {
    pub id: u32,
    pub ldr: u32, // Logical Destination Register

    /// Task Priority Register: 7:4 = priority threshold
    pub tpr: u8,
    pub ppr: u8, // Processor Priority (derived)

    /// Interrupt Request Register: containes pending interrupts that have not yet been dispatched to the processor
    pub irr: [u32; 8],
    /// In-Service Register: contains interrupts that have been dispatched to the processor but not yet EOIed
    pub isr: [u32; 8],
    pub icr: [u32; 2], // Interrupt Command Register, 64 bits
    pub tmr: [u32; 8], // Trigger Mode Register

    pub lvt_lint0: u32,
    pub lvt_lint1: u32,

    pub timer: ApicTimer,
}

impl Default for Lapic {
    fn default() -> Self {
        Self {
            id: 0,
            ldr: 0,
            tpr: 0,
            ppr: 0,
            irr: [0; 8],
            isr: [0; 8],
            icr: [0; 2],
            tmr: [0; 8],
            lvt_lint0: APIC_LVT_MASKED,
            lvt_lint1: APIC_LVT_MASKED,
            timer: ApicTimer::default(),
        }
    }
}

/// Intel® 64 and IA-32 Architectures Software Developer’s Manual.
/// 12.5.4. APIC Timer.
/// APIC timer state.
#[derive(Debug, Default)]
pub struct ApicTimer {
    pub lvt_timer: u32, // LVT(Local Vector Table) Timer Register
    /// divide configuration register
    /// timer count rate = virtual crystal frequency / divide.
    pub divide: u32,
    pub initial_count: u32,
    pub current_count: u32,
    pub tsc_deadline_msr: u64,
    pub deadline_tsc: Option<u64>,
}

impl ApicTimer {
    pub fn divide_shift(&self) -> u32 {
        let shift = (self.divide & 0b11) | ((self.divide & 0b1000) >> 1);
        (shift + 1) & 0b111
    }

    pub fn count_to_tsc_cycles(&self, count: u64) -> u64 {
        let divide = 1_u64 << self.divide_shift();
        let cycles = u128::from(count)
            .saturating_mul(u128::from(divide))
            .saturating_mul(u128::from(tsc_freq()))
            / u128::from(VIRTUAL_TSC_CRYSTAL_HZ);
        cycles.min(u128::from(u64::MAX)) as u64
    }

    fn is_masked(&self) -> bool {
        (self.lvt_timer & APIC_LVT_MASKED) != 0
    }

    fn mode(&self) -> u32 {
        (self.lvt_timer >> 17) & 0b11
    }

    fn write_lvt_timer(&mut self, value: u32) {
        let old_mode = self.mode();
        self.lvt_timer = value;
        let new_mode = self.mode();

        if new_mode == APIC_TIMER_MODE_TSC_DEADLINE {
            self.deadline_tsc = (self.tsc_deadline_msr != 0).then_some(self.tsc_deadline_msr);
        } else if old_mode == APIC_TIMER_MODE_TSC_DEADLINE {
            self.deadline_tsc = None;
        }
    }

    pub fn read_tsc_deadline_msr(&self) -> u64 {
        self.tsc_deadline_msr
    }

    pub fn write_tsc_deadline_msr(&mut self, value: u64) {
        self.tsc_deadline_msr = value;
        if self.mode() == APIC_TIMER_MODE_TSC_DEADLINE {
            self.deadline_tsc = (value != 0).then_some(value);
        }
    }

    fn vector(&self) -> u8 {
        (self.lvt_timer & 0xff) as u8
    }

    fn period_tsc_cycles(&self) -> Option<u64> {
        if self.initial_count == 0 {
            return None;
        }
        Some(
            self.count_to_tsc_cycles(u64::from(self.initial_count))
                .max(1),
        )
    }

    fn arm(&mut self, current_tsc: u64, initial_count: u64) {
        self.initial_count = initial_count as u32;
        self.current_count = self.initial_count;
        if self.mode() == APIC_TIMER_MODE_TSC_DEADLINE {
            self.deadline_tsc = (self.tsc_deadline_msr != 0).then_some(self.tsc_deadline_msr);
            return;
        }

        self.deadline_tsc = self
            .period_tsc_cycles()
            .map(|period| current_tsc.saturating_add(period));
    }

    fn stop(&mut self) {
        self.current_count = 0;
        self.deadline_tsc = None;
    }

    fn current_count(&self, current_tsc: u64) -> u64 {
        if self.mode() == APIC_TIMER_MODE_TSC_DEADLINE {
            return 0;
        }

        let Some(deadline_tsc) = self.deadline_tsc else {
            return 0;
        };
        if current_tsc >= deadline_tsc {
            return 0;
        }
        (deadline_tsc - current_tsc) / self.count_to_tsc_cycles(1).max(1)
    }
}

/// A single I/O APIC redirection table entry.
#[derive(Debug, Default, Clone, Copy)]
pub struct IoapicRedent {
    pub vector: u8,
    /// Delivery mode (3 bits): 000 = Fixed
    pub delivery_mode: u8,
    /// Destination mode: 0 = Physical, 1 = Logical
    pub dest_mode: u8,
    /// Valid for level-triggered interrupts
    /// 1 = Level-triggered interrupt has been sent and not yet EOIed.
    pub remote_irr: bool,
    /// Trigger mode: 0 = Edge, 1 = Level
    pub trigger_mode: TriggerMode,
    pub mask: bool,
    /// Target LAPIC ID
    pub dest_id: u8,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TriggerMode {
    #[default]
    Edge,
    Level,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IrqLineLevel {
    Deasserted,
    Asserted,
}

/// Packed 64-bit redirection table entry (fields + bits view).
#[derive(Debug, Default, Clone, Copy)]
pub struct IoapicRedtbl {
    pub bits: u64,
}

/// I/O APIC state.
#[derive(Debug)]
pub struct Ioapic {
    // reference to 82093AA I/O ADVANCED PROGRAMMABLE INTERRUPT CONTROLLER
    // IOAPIC registers
    // 3.1. IOREGSEL and IOWIN is memory mapped registers
    pub ioregsel: u32,
    // 3.2. IOAPICID and others
    pub id: u32,
    // 3.2.4. IOREDTBL[23:0] -- I/O REDIRECTION TABLE REGISTERS
    pub redtbl: [IoapicRedtbl; IOAPIC_NUM_PINS],
    pin_levels: u32,
}

impl Default for Ioapic {
    fn default() -> Self {
        Self {
            ioregsel: 0,
            id: 1,
            redtbl: [IoapicRedtbl::default(); IOAPIC_NUM_PINS],
            pin_levels: 0,
        }
    }
}

impl Lapic {
    pub fn to_kvm_state(&self) -> LapicState {
        let mut state = LapicState::default();

        write_apic_reg(&mut state.regs, XLAPIC_RW_ID, self.id << 24);
        write_apic_reg(&mut state.regs, XLAPIC_RO_VER, (6 << 16) | 0x14);
        write_apic_reg(&mut state.regs, XLAPIC_RW_TPR, self.tpr as u32);
        write_apic_reg(&mut state.regs, XLAPIC_RO_PPR, self.ppr as u32);
        write_apic_reg(&mut state.regs, XLAPIC_RW_LDR, self.ldr);
        write_apic_reg(&mut state.regs, XLAPIC_RW_DFR, 0xFFFF_FFFF);
        write_apic_reg(&mut state.regs, XLAPIC_RW_SIVR, 0x1FF);
        write_apic_reg(&mut state.regs, XLAPIC_RW_LVT_CMCI, APIC_LVT_MASKED);
        write_apic_reg(&mut state.regs, XLAPIC_RW_LVT_THERM, APIC_LVT_MASKED);
        write_apic_reg(&mut state.regs, XLAPIC_RW_LVT_PERF, APIC_LVT_MASKED);
        write_apic_reg(&mut state.regs, XLAPIC_RW_LVT_LINT0, self.lvt_lint0);
        write_apic_reg(&mut state.regs, XLAPIC_RW_LVT_LINT1, self.lvt_lint1);
        write_apic_reg(&mut state.regs, XLAPIC_RW_LVT_ERROR, APIC_LVT_MASKED);
        write_apic_reg(&mut state.regs, XLAPIC_RW_LVT_TIMER, self.timer.lvt_timer);
        write_apic_reg(
            &mut state.regs,
            XLAPIC_RW_TIMER_INIT,
            self.timer.initial_count,
        );
        write_apic_reg(
            &mut state.regs,
            XLAPIC_RO_TIMER_CURR,
            self.timer.current_count,
        );
        write_apic_reg(&mut state.regs, XLAPIC_RW_TIMER_DIVI, self.timer.divide);
        write_apic_reg(&mut state.regs, XLAPIC_RW_ICR_LOW, self.icr[0]);
        write_apic_reg(&mut state.regs, XLAPIC_RW_ICR_HIGH, self.icr[1]);

        for index in 0..8 {
            write_apic_reg(
                &mut state.regs,
                apic_reg_array_offset(XLAPIC_RO_ISR_BASE, index),
                self.isr[index],
            );
            write_apic_reg(
                &mut state.regs,
                apic_reg_array_offset(XLAPIC_RO_TMR_BASE, index),
                self.tmr[index],
            );
            write_apic_reg(
                &mut state.regs,
                apic_reg_array_offset(XLAPIC_RO_IRR_BASE, index),
                self.irr[index],
            );
        }

        state
    }

    pub fn set_from_kvm_state(&mut self, state: &LapicState) {
        self.id = (read_apic_reg(&state.regs, XLAPIC_RW_ID) >> 24) & 0xFF;
        self.tpr = (read_apic_reg(&state.regs, XLAPIC_RW_TPR) & 0xFF) as u8;
        self.ldr = read_apic_reg(&state.regs, XLAPIC_RW_LDR) & 0xFF00_0000;
        self.lvt_lint0 = sanitize_lvt_lint(read_apic_reg(&state.regs, XLAPIC_RW_LVT_LINT0));
        self.lvt_lint1 = sanitize_lvt_lint(read_apic_reg(&state.regs, XLAPIC_RW_LVT_LINT1));
        self.timer
            .write_lvt_timer(read_apic_reg(&state.regs, XLAPIC_RW_LVT_TIMER));
        self.timer.initial_count = read_apic_reg(&state.regs, XLAPIC_RW_TIMER_INIT);
        self.timer.current_count = read_apic_reg(&state.regs, XLAPIC_RO_TIMER_CURR);
        self.timer.divide = read_apic_reg(&state.regs, XLAPIC_RW_TIMER_DIVI);
        self.icr[0] = read_apic_reg(&state.regs, XLAPIC_RW_ICR_LOW);
        self.icr[1] = read_apic_reg(&state.regs, XLAPIC_RW_ICR_HIGH);

        for index in 0..8 {
            self.isr[index] = read_apic_reg(
                &state.regs,
                apic_reg_array_offset(XLAPIC_RO_ISR_BASE, index),
            );
            self.tmr[index] = read_apic_reg(
                &state.regs,
                apic_reg_array_offset(XLAPIC_RO_TMR_BASE, index),
            );
            self.irr[index] = read_apic_reg(
                &state.regs,
                apic_reg_array_offset(XLAPIC_RO_IRR_BASE, index),
            );
        }

        self.update_ppr();
    }

    pub fn read_tsc_deadline_msr(&self) -> u64 {
        self.timer.read_tsc_deadline_msr()
    }

    pub fn write_tsc_deadline_msr(&mut self, value: u64) {
        self.timer.write_tsc_deadline_msr(value);
    }

    pub fn add_pending_interrupt(&mut self, vec: u8, trigger_mode: TriggerMode) {
        match trigger_mode {
            TriggerMode::Edge => Self::clear_bit(&mut self.tmr, vec),
            TriggerMode::Level => Self::set_bit(&mut self.tmr, vec),
        }
        // Set the corresponding bit in IRR.
        Self::set_bit(&mut self.irr, vec);
    }

    fn complete_interrupt(&mut self) -> Option<(u8, TriggerMode)> {
        if let Some(isr_vec) = Self::find_highest(&self.isr) {
            let trigger_mode = if Self::test_bit(&self.tmr, isr_vec) {
                TriggerMode::Level
            } else {
                TriggerMode::Edge
            };
            Self::clear_bit(&mut self.isr, isr_vec);
            self.update_ppr();
            Some((isr_vec, trigger_mode))
        } else {
            None
        }
    }

    fn update_ppr(&mut self) {
        let isr_prio = Self::find_highest(&self.isr).map(|v| v & 0xF0).unwrap_or(0) as u8;
        self.ppr = self.tpr.max(isr_prio);
    }

    fn set_bit(val: &mut [u32; 8], vec: u8) {
        val[(vec / 32) as usize] |= 1u32 << (vec % 32);
    }

    fn clear_bit(val: &mut [u32; 8], vec: u8) {
        val[(vec / 32) as usize] &= !(1u32 << (vec % 32));
    }

    fn test_bit(val: &[u32; 8], vec: u8) -> bool {
        val[(vec / 32) as usize] & (1u32 << (vec % 32)) != 0
    }

    fn find_highest(val: &[u32; 8]) -> Option<u8> {
        for i in (0..8usize).rev() {
            let v = val[i];
            if v != 0 {
                let bit = 31 - v.leading_zeros();
                return Some((i as u32 * 32 + bit) as u8);
            }
        }
        None
    }
}

fn sanitize_lvt_lint(value: u32) -> u32 {
    value & APIC_LINT_MASK
}

fn apic_reg_array_offset(base: u64, index: usize) -> u64 {
    base + (index as u64) * 0x10
}

fn read_apic_reg(regs: &[u8], offset: u64) -> u32 {
    let offset = offset as usize;
    let mut bytes = [0; 4];
    bytes.copy_from_slice(&regs[offset..offset + 4]);
    u32::from_le_bytes(bytes)
}

fn write_apic_reg(regs: &mut [u8], offset: u64, value: u32) {
    let offset = offset as usize;
    regs[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

use ostd::{
    arch::tsc_freq,
    vm::{GuestInterruptPort, GuestTimerPort},
};

use super::cpuid::VIRTUAL_TSC_CRYSTAL_HZ;

pub(super) struct LapicPort {
    inner: SpinLock<Lapic>,
}

impl LapicPort {
    pub fn new(lapic: Lapic) -> Self {
        Self {
            inner: SpinLock::new(lapic),
        }
    }

    pub fn lock(&self) -> SpinLockGuard<'_, Lapic, ostd::sync::PreemptDisabled> {
        self.inner.lock()
    }
}

impl GuestInterruptPort for LapicPort {
    fn check_pending_interrupt(&self) -> Option<GuestInterrupt> {
        if let Some(vector) = self.lock().check_pending_interrupt() {
            Some(GuestInterrupt { vector })
        } else {
            None
        }
    }

    fn accept_interrupt(&self, interrupt: GuestInterrupt) {
        self.lock().accept_interrupt(interrupt.vector);
    }
}

impl GuestTimerPort for LapicPort {
    fn poll_deadline(&self, current: GuestTimerInstant) -> Option<GuestTimerInstant> {
        if let Some(tsc) = self.lock().poll_deadline(current.tsc) {
            Some(GuestTimerInstant { tsc })
        } else {
            None
        }
    }
}

impl Lapic {
    pub(crate) fn check_pending_interrupt(&self) -> Option<u8> {
        let pending_vector = Self::find_highest(&self.irr)?;

        // interrupt vector 的高四位表示优先级，低四位表示具体的中断号。同一优先级的中断由低四位决定先后顺序。
        let pending_prio = pending_vector >> 4;
        let tpr_prio = self.tpr >> 4;
        let isr_prio = Self::find_highest(&self.isr)
            .map(|vector| vector >> 4)
            .unwrap_or(0);

        if pending_prio > tpr_prio && pending_prio > isr_prio {
            Some(pending_vector)
        } else {
            None
        }
    }

    fn accept_interrupt(&mut self, vector: u8) {
        Self::set_bit(&mut self.isr, vector);
        Self::clear_bit(&mut self.irr, vector);
        self.update_ppr();
    }

    pub(crate) fn poll_deadline(&mut self, current_tsc: u64) -> Option<u64> {
        let deadline_tsc = self.timer.deadline_tsc?;
        if current_tsc < deadline_tsc {
            return (!self.timer.is_masked()).then_some(deadline_tsc);
        }

        let vector = self.timer.vector();
        let mode = self.timer.mode();
        let masked = self.timer.is_masked();
        if !masked {
            self.add_pending_interrupt(vector, TriggerMode::Edge);
        }

        let next_deadline = match mode {
            APIC_TIMER_MODE_PERIODIC => {
                let Some(period) = self.timer.period_tsc_cycles() else {
                    self.timer.stop();
                    return None;
                };
                let elapsed_periods = current_tsc
                    .saturating_sub(deadline_tsc)
                    .checked_div(period)
                    .unwrap_or(0)
                    .saturating_add(1);
                Some(deadline_tsc.saturating_add(period.saturating_mul(elapsed_periods)))
            }
            APIC_TIMER_MODE_ONESHOT | APIC_TIMER_MODE_TSC_DEADLINE => None,
            _ => None,
        };
        self.timer.deadline_tsc = next_deadline;
        next_deadline
    }
}

impl Ioapic {
    /// Updates an I/O APIC input pin and delivers a newly active interrupt.
    pub fn set_pin_level<'a, I>(&mut self, lapics: I, irq: usize, line_level: IrqLineLevel) -> bool
    where
        I: IntoIterator<Item = &'a mut Lapic>,
    {
        if irq >= IOAPIC_NUM_PINS {
            return false;
        }

        let was_asserted = self.pin_is_asserted(irq);
        match line_level {
            IrqLineLevel::Deasserted => {
                self.pin_levels &= !(1 << irq);
                return false;
            }
            IrqLineLevel::Asserted => self.pin_levels |= 1 << irq,
        }

        let entry = self.redtbl[irq].fields();
        if (entry.trigger_mode == TriggerMode::Edge && was_asserted) ||
           (entry.trigger_mode == TriggerMode::Level && entry.remote_irr) {
            return false;
        }

        self.inject_irq_line(lapics, irq)
    }

    /// Delivers an IRQ from the I/O APIC to the appropriate vCPU LAPICs.
    pub fn inject_irq_line<'a, I>(&mut self, lapics: I, irq: usize) -> bool
    where
        I: IntoIterator<Item = &'a mut Lapic>,
    {
        if irq >= IOAPIC_NUM_PINS {
            return false;
        }

        let entry = self.redtbl[irq].fields();

        if entry.mask {
            return false;
        }

        if entry.trigger_mode == TriggerMode::Level && entry.remote_irr {
            return false;
        }

        let vec = entry.vector;
        if vec < 0x10 || vec > 0xFE {
            return false;
        }

        if entry.delivery_mode != 0b000 {
            warn!(
                "vIOAPIC: Unhandled delivery mode: {:03b}, ignore",
                entry.delivery_mode
            );
        }

        let destination = entry.dest_id;
        let mut delivered = false;
        if entry.dest_mode == 0 {
            // Physical mode: send to a specific LAPIC.
            for lapic in lapics {
                if lapic.id as u8 == destination {
                    lapic.add_pending_interrupt(vec, entry.trigger_mode);
                    delivered = true;
                    break;
                }
            }
        } else {
            // Logical mode (flat model): send to a group of LAPICs.
            for lapic in lapics {
                if ((lapic.ldr >> 24) as u8) & destination != 0 {
                    lapic.add_pending_interrupt(vec, entry.trigger_mode);
                    delivered = true;
                }
            }
        }

        if delivered && entry.trigger_mode == TriggerMode::Level {
            self.redtbl[irq].set_remote_irr(true);
        }
        delivered
    }

    /// Completes a level interrupt and returns asserted pins that need redelivery.
    pub fn complete_level_interrupt(&mut self, vec: u8) -> u32 {
        let mut redelivery_pins = 0;
        for irq in 0..IOAPIC_NUM_PINS {
            let entry = self.redtbl[irq].fields();
            if entry.vector != vec || entry.trigger_mode != TriggerMode::Level {
                continue;
            }

            self.redtbl[irq].set_remote_irr(false);
            if self.pin_is_asserted(irq) {
                redelivery_pins |= 1 << irq;
            }
        }
        redelivery_pins
    }

    fn pin_is_asserted(&self, irq: usize) -> bool {
        self.pin_levels & (1 << irq) != 0
    }
}

impl IoapicRedtbl {
    pub fn fields(&self) -> IoapicRedent {
        IoapicRedent {
            vector: (self.bits & 0xFF) as u8,
            delivery_mode: ((self.bits >> 8) & 0x7) as u8,
            dest_mode: ((self.bits >> 11) & 0x1) as u8,
            remote_irr: ((self.bits >> 14) & 0x1) != 0,
            trigger_mode: if ((self.bits >> 15) & 0x1) != 0 {
                TriggerMode::Level
            } else {
                TriggerMode::Edge
            },
            mask: ((self.bits >> 16) & 0x1) != 0,
            dest_id: ((self.bits >> 56) & 0xFF) as u8,
        }
    }

    pub fn set_remote_irr(&mut self, val: bool) {
        if val {
            self.bits |= 1 << 14;
        } else {
            self.bits &= !(1 << 14);
        }
    }
}

pub const LAPIC_BASE: u64 = 0xFEE0_0000;
pub const LAPIC_SIZE: u64 = 0x400;

pub const IOAPIC_BASE: u64 = 0xFEC0_0000;
pub const IOAPIC_SIZE: u64 = 0x20;

const XLAPIC_RW_ID: u64 = 0x020;
const XLAPIC_RO_VER: u64 = 0x030;
const XLAPIC_RW_TPR: u64 = 0x080;
const XLAPIC_RO_APR: u64 = 0x090;
const XLAPIC_RO_PPR: u64 = 0x0A0;
const XLAPIC_WO_EOI: u64 = 0x0B0;
const XLAPIC_RO_RRD: u64 = 0x0C0;
const XLAPIC_RW_LDR: u64 = 0x0D0;
const XLAPIC_RW_DFR: u64 = 0x0E0;
const XLAPIC_RW_SIVR: u64 = 0x0F0;
const XLAPIC_RO_ISR_BASE: u64 = 0x100;
const XLAPIC_RO_ISR_SIZE: u64 = 0x080; // 0x180 - 0x100
const XLAPIC_RO_TMR_BASE: u64 = 0x180;
const XLAPIC_RO_TMR_SIZE: u64 = 0x080; // 0x200 - 0x180
const XLAPIC_RO_IRR_BASE: u64 = 0x200;
const XLAPIC_RO_IRR_SIZE: u64 = 0x080; // 0x280 - 0x200
const XLAPIC_RW_ESR: u64 = 0x280;
const XLAPIC_RW_LVT_CMCI: u64 = 0x2F0;
const XLAPIC_RW_ICR_LOW: u64 = 0x300;
const XLAPIC_RW_ICR_HIGH: u64 = 0x310;
const XLAPIC_RW_LVT_TIMER: u64 = 0x320;
const XLAPIC_RW_LVT_THERM: u64 = 0x330;
const XLAPIC_RW_LVT_PERF: u64 = 0x340;
const XLAPIC_RW_LVT_LINT0: u64 = 0x350;
const XLAPIC_RW_LVT_LINT1: u64 = 0x360;
const XLAPIC_RW_LVT_ERROR: u64 = 0x370;
const XLAPIC_RW_TIMER_INIT: u64 = 0x380;
const XLAPIC_RO_TIMER_CURR: u64 = 0x390;
const XLAPIC_WO_SELF_IPI: u64 = 0x3F0;
const XLAPIC_RW_TIMER_DIVI: u64 = 0x3E0;

use super::{
    mmio::{decode_current_mmio_instruction, MmioDirection, MmioInstruction},
    vcpu::Vcpu,
};

/// Emulate a guest access to APIC MMIO region.
/// Returns `Ok(true)` if the access is successfully emulated.
///         `Ok(false)` if the access is not to APIC MMIO or is unsupported.
///         `Err` if an error occurs during emulation.
pub(super) fn emulate_apic_mmio(vcpu: Arc<Vcpu>, fault_gpa: u64) -> Result<bool> {
    // log::error!("Guest access to APIC MMIO at GPA {:#x}", fault_gpa);
    let is_lapic = (LAPIC_BASE..(LAPIC_BASE + LAPIC_SIZE)).contains(&fault_gpa);
    let is_ioapic = (IOAPIC_BASE..(IOAPIC_BASE + IOAPIC_SIZE)).contains(&fault_gpa);
    if !is_lapic && !is_ioapic {
        return Ok(false);
    }

    let Some(insn) = decode_current_mmio_instruction(&vcpu)? else {
        return Ok(false);
    };

    if is_lapic {
        if !emulate_lapic_mmio(vcpu.clone(), fault_gpa, insn)? {
            return Ok(false);
        }
    } else {
        if !emulate_ioapic_mmio(vcpu.clone(), fault_gpa, insn)? {
            return Ok(false);
        }
    }

    vcpu.guest_context().advance_rip(u64::from(insn.len()));
    Ok(true)
}

fn emulate_lapic_mmio(vcpu: Arc<Vcpu>, fault_gpa: u64, insn: MmioInstruction) -> Result<bool> {
    // log::error!("Guest access to LAPIC MMIO at GPA {:#x}", fault_gpa);
    let offset = fault_gpa - LAPIC_BASE;
    // read
    if insn.direction() == MmioDirection::Read {
        let (value, ok) = emulate_lapic_read(vcpu.clone(), offset);
        if !ok {
            return Ok(false);
        }

        insn.complete_read(&mut vcpu.guest_context(), value)?;

        return Ok(true);
    }

    // write
    let Some(value) = insn.write_value(&vcpu.guest_context()) else {
        return Ok(false);
    };

    let vm = vcpu.vm()?;

    match emulate_lapic_write(vcpu.clone(), offset, value) {
        Some(LapicWriteEffect::Eoi {
            isr_vec,
            trigger_mode,
        }) => {
            if trigger_mode == TriggerMode::Level {
                vm.complete_ioapic_interrupt(isr_vec);
            }
        }
        Some(LapicWriteEffect::DeliverIcr { icr }) => {
            vm.inject_ipi(icr)?;
        }
        None => {}
    }
    Ok(true)
}

fn emulate_ioapic_mmio(vcpu: Arc<Vcpu>, fault_gpa: u64, insn: MmioInstruction) -> Result<bool> {
    // log::error!("Guest access to IOAPIC MMIO at GPA {:#x}", fault_gpa);
    let offset = fault_gpa - IOAPIC_BASE;
    // log::error!("IOAPIC MMIO access with offset {:#x}, IOAPIC_BASE {:#x}", offset, IOAPIC_BASE);
    let vm = vcpu.vm()?;
    let mut ioapic = vm.ioapic();

    // read
    if insn.direction() == MmioDirection::Read {
        let (value, ok) = emulate_ioapic_read(&ioapic, offset);
        if !ok {
            return Ok(false);
        }
        insn.complete_read(&mut vcpu.guest_context(), value)?;

        return Ok(true);
    }

    // write
    let Some(value) = insn.write_value(&vcpu.guest_context()) else {
        return Ok(false);
    };
    if !emulate_ioapic_write(&mut ioapic, offset, value) {
        return Ok(false);
    }
    Ok(true)
}

/// Emulate a LAPIC MMIO read.
///
/// Returns `(value, ok)` where `ok` is false if the offset is unsupported.
pub fn emulate_lapic_read(vcpu: Arc<Vcpu>, offset: u64) -> (u64, bool) {
    let lapic = vcpu.lapic();
    let value = match offset {
        XLAPIC_RW_ID => (lapic.id as u64) << 24,
        // Not support EOI-broadcast; Max LVT Number is 6; Version is 'Integrated APIC'
        XLAPIC_RO_VER => (0u64 << 24) | (6u64 << 16) | 0x14,
        XLAPIC_RW_TPR => lapic.tpr as u64,
        XLAPIC_RO_APR => 0,
        XLAPIC_RO_PPR => lapic.ppr as u64,
        XLAPIC_RO_RRD => 0,
        XLAPIC_RW_LDR => lapic.ldr as u64,
        XLAPIC_RW_DFR => 0xFFFF_FFFF,
        XLAPIC_RW_SIVR => 0x1FF,
        XLAPIC_RW_LVT_CMCI => 1u64 << 16,
        XLAPIC_RW_LVT_THERM | XLAPIC_RW_LVT_PERF | XLAPIC_RW_LVT_ERROR => 0x10000,
        XLAPIC_RW_LVT_LINT0 => lapic.lvt_lint0 as u64,
        XLAPIC_RW_LVT_LINT1 => lapic.lvt_lint1 as u64,
        XLAPIC_RW_LVT_TIMER => lapic.timer.lvt_timer as u64,
        XLAPIC_RW_TIMER_INIT => lapic.timer.initial_count as u64,
        XLAPIC_RW_TIMER_DIVI => lapic.timer.divide as u64,
        XLAPIC_RO_TIMER_CURR => {
            // read the timer ticks remaining until the next timer interrupt.
            let context = vcpu.guest_context();
            lapic.timer.current_count(context.guest_tsc())
        }
        XLAPIC_RW_ESR => 0,
        o if o >= XLAPIC_RO_ISR_BASE && o < XLAPIC_RO_ISR_BASE + XLAPIC_RO_ISR_SIZE => {
            lapic.isr[((o - XLAPIC_RO_ISR_BASE) / 16) as usize] as u64
        }
        o if o >= XLAPIC_RO_IRR_BASE && o < XLAPIC_RO_IRR_BASE + XLAPIC_RO_IRR_SIZE => {
            lapic.irr[((o - XLAPIC_RO_IRR_BASE) / 16) as usize] as u64
        }
        XLAPIC_RW_ICR_HIGH => lapic.icr[1] as u64,
        XLAPIC_RW_ICR_LOW => lapic.icr[0] as u64,
        o if o >= XLAPIC_RO_TMR_BASE && o < XLAPIC_RO_TMR_BASE + XLAPIC_RO_TMR_SIZE => {
            lapic.tmr[((o - XLAPIC_RO_TMR_BASE) / 16) as usize] as u64
        }
        _ => {
            warn!("MMIO.xLAPIC: Read at offset {:#05x} not supported", offset);
            return (0, false);
        }
    };
    (value, true)
}

pub struct Icr {
    pub delivery_mode: u8,
    pub dest_mode: u8,
    pub dest_shorthand: u8,
    pub dest_id: u8,
    pub src_id: u8,
    pub vector: u8,
}

/// Result of a LAPIC write that may require a timer action.
pub enum LapicWriteEffect {
    Eoi {
        isr_vec: u8,
        trigger_mode: TriggerMode,
    },
    DeliverIcr {
        icr: Icr,
    },
}

/// Emulate a LAPIC MMIO write.
///
/// Returns the side-effect the caller must act on, and `ok` (false = unsupported offset).
pub fn emulate_lapic_write(vcpu: Arc<Vcpu>, offset: u64, value: u64) -> Option<LapicWriteEffect> {
    let mut lapic = vcpu.lapic();
    match offset {
        XLAPIC_RW_ID => {
            let new_apic_id = ((value >> 24) & 0xFF) as u32;
            lapic.id = new_apic_id;
        }
        XLAPIC_RW_TPR => {
            lapic.tpr = (value & 0xFF) as u8;
        }
        XLAPIC_WO_EOI => {
            // Find highest in-service vector and complete it
            if let Some((isr_vec, trigger_mode)) = lapic.complete_interrupt() {
                return Some(LapicWriteEffect::Eoi {
                    isr_vec,
                    trigger_mode,
                });
            }
        }
        XLAPIC_RW_LDR => {
            let new = (value as u32) & 0xFF00_0000;
            // Accept only a single-bit logical ID
            if new != 0 && (new & new.wrapping_sub(1 << 24)) == 0 {
                lapic.ldr = new;
            }
        }
        XLAPIC_RW_DFR => {
            if ((value >> 28) & 0xF) != 0xF {
                warn!("vLAPIC: Unsupported cluster model, ignore");
            }
        }
        XLAPIC_RW_LVT_TIMER => {
            lapic.timer.write_lvt_timer(value as u32);
        }
        XLAPIC_RW_TIMER_INIT => {
            let current_tsc = vcpu.guest_context().guest_tsc();
            lapic.timer.arm(current_tsc, value);
        }
        XLAPIC_RW_TIMER_DIVI => {
            lapic.timer.divide = value as u32;
        }
        XLAPIC_WO_SELF_IPI => {
            lapic.add_pending_interrupt((value & 0xFF) as u8, TriggerMode::Edge);
        }
        XLAPIC_RW_ESR => {
            if value != 0 {
                warn!(
                    "MMIO.xLAPIC: Write to xLAPIC_RW_ESR with non-zero value {:#018x}",
                    value
                );
            }
        }
        XLAPIC_RW_ICR_LOW => {
            lapic.icr[0] = value as u32;
            // TODO: make sure
            if value >> 32 != 0 {
                lapic.icr[1] = (value >> 32) as u32;
            }
            let icr = Icr {
                vector: (value & 0xFF) as u8,
                delivery_mode: ((value >> 8) & 0x7) as u8,
                dest_mode: ((value >> 11) & 0x1) as u8,
                dest_shorthand: ((value >> 18) & 0b11) as u8,
                dest_id: ((lapic.icr[1] >> 24) & 0xFF) as u8,
                src_id: lapic.id as u8,
            };
            return Some(LapicWriteEffect::DeliverIcr { icr });
        }
        XLAPIC_RW_ICR_HIGH => {
            lapic.icr[1] = value as u32;
        }
        XLAPIC_RW_LVT_LINT0 => {
            lapic.lvt_lint0 = sanitize_lvt_lint(value as u32);
        }
        XLAPIC_RW_LVT_LINT1 => {
            lapic.lvt_lint1 = sanitize_lvt_lint(value as u32);
        }
        XLAPIC_RW_SIVR | XLAPIC_RW_LVT_CMCI | XLAPIC_RW_LVT_THERM | XLAPIC_RW_LVT_PERF
        | XLAPIC_RW_LVT_ERROR => { /* silently ignored */ }
        _ => {
            warn!(
                "MMIO.xLAPIC: Write at offset {:#05x} not supported, value is {:#018x}",
                offset, value
            );
            return None;
        }
    }
    None
}

/// Emulate an IOAPIC MMIO read.
pub fn emulate_ioapic_read(ioapic: &Ioapic, offset: u64) -> (u64, bool) {
    if offset == 0x00 {
        return (ioapic.ioregsel as u64, true);
    } else if offset == 0x10 {
        let index = ioapic.ioregsel;
        let value = match index {
            0x00 => (ioapic.id as u64) << 24,
            0x01 => {
                // Bits 0-7: Version (0x11 for 82093AA)
                // Bits 16-23: Max Redirection Entry (N-1); 24 entries -> 23
                0x0017_0011
            }
            i if (0x10..=0x3F).contains(&i) => {
                let pin = ((i - 0x10) / 2) as usize;
                let value = ioapic.redtbl[pin].bits;
                if i & 1 != 0 {
                    value >> 32
                } else {
                    value & 0x0000_0000_FFFF_FFFF
                }
            }
            _ => 0,
        };
        return (value, true);
    }
    warn!("IOAPIC: Read invalid offset {:#x}", offset);
    (0, false)
}

/// Emulate an IOAPIC MMIO write.
pub fn emulate_ioapic_write(ioapic: &mut Ioapic, offset: u64, value: u64) -> bool {
    if offset == 0x00 {
        ioapic.ioregsel = (value & 0xFF) as u32;
        return true;
    } else if offset == 0x10 {
        let index = ioapic.ioregsel;
        if (0x10..=0x3F).contains(&index) {
            let pin = ((index - 0x10) / 2) as usize;
            if index & 1 != 0 {
                ioapic.redtbl[pin].bits &= 0x0000_0000_FFFF_FFFF;
                ioapic.redtbl[pin].bits |= value << 32;
            } else {
                let remote_irr = ioapic.redtbl[pin].fields().remote_irr;
                ioapic.redtbl[pin].bits &= 0xFFFF_FFFF_0000_0000;
                ioapic.redtbl[pin].bits |= value & 0xFFFF_FFFF;
                if ioapic.redtbl[pin].fields().trigger_mode == TriggerMode::Level {
                    ioapic.redtbl[pin].set_remote_irr(remote_irr);
                } else {
                    ioapic.redtbl[pin].set_remote_irr(false);
                }
            }
        }
        return true;
    }
    warn!("IOAPIC: Write invalid offset {:#x}", offset);
    false
}

pub fn default_lapic_ldr(vcpu_id: u32) -> u32 {
    (1_u32.checked_shl(vcpu_id).unwrap_or(0)) << 24
}

pub fn default_lapic_lvt_lint0(vcpu_id: u32) -> u32 {
    if vcpu_id == 0 {
        APIC_MODE_EXTINT << 8
    } else {
        APIC_LVT_MASKED
    }
}

const APIC_ICR_DESTINATION_MODE_LOGICAL: u8 = 1;
const APIC_ICR_SHORTHAND_NONE: u8 = 0;
const APIC_ICR_SHORTHAND_SELF: u8 = 1;
const APIC_ICR_SHORTHAND_ALL_INCLUDING_SELF: u8 = 2;
const APIC_ICR_SHORTHAND_ALL_EXCLUDING_SELF: u8 = 3;

pub fn icr_matches_destination(target_lapic: &Lapic, source_icr: &Icr) -> bool {
    match source_icr.dest_shorthand {
        APIC_ICR_SHORTHAND_NONE => {
            if source_icr.dest_mode == APIC_ICR_DESTINATION_MODE_LOGICAL {
                ((target_lapic.ldr >> 24) as u8) & source_icr.dest_id != 0
            } else {
                target_lapic.id as u8 == source_icr.dest_id
            }
        }
        APIC_ICR_SHORTHAND_SELF => target_lapic.id as u8 == source_icr.src_id,
        APIC_ICR_SHORTHAND_ALL_INCLUDING_SELF => true,
        APIC_ICR_SHORTHAND_ALL_EXCLUDING_SELF => target_lapic.id as u8 != source_icr.src_id,
        _ => false,
    }
}
