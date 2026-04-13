//! Emulated LAPIC and IOAPIC device for guest VMs.
//!
//! Use pure software emulate for now.

use alloc::sync::Arc;

use ostd::sync::Mutex;

use super::super::vm::{Vcpu, Vm};

// ===== LAPIC MMIO Register Offsets =====
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
const XLAPIC_RW_ICR_BASE: u64 = 0x300;
const XLAPIC_RW_ICR_SIZE: u64 = 0x020; // 0x320 - 0x300
const XLAPIC_RW_LVT_TIMER: u64 = 0x320;
const XLAPIC_RW_LVT_THERM: u64 = 0x330;
const XLAPIC_RW_LVT_PERF: u64 = 0x340;
const XLAPIC_RW_LVT_LINT0: u64 = 0x350;
const XLAPIC_RW_LVT_LINT1: u64 = 0x360;
const XLAPIC_RW_LVT_ERROR: u64 = 0x370;
const XLAPIC_RW_TIMER_INIT: u64 = 0x380;
const XLAPIC_RO_TIMER_CURR: u64 = 0x390;
const XLAPIC_RW_TIMER_DIVI: u64 = 0x3E0;

pub const IOAPIC_NUM_PINS: usize = 24;

// ===== Data Structures =====

/// Local APIC state.
#[derive(Debug, Default)]
pub struct Lapic {
    pub id: u32,
    pub ldr: u32, // Logical Destination Register

    pub tpr: u8, // Task Priority
    pub ppr: u8, // Processor Priority (derived)

    pub irr: [u32; 8], // Interrupt Request Register
    pub isr: [u32; 8], // In-Service Register
    pub icr: [u32; 8], // Interrupt Command Register
    pub tmr: [u32; 8], // Trigger Mode Register
}

/// APIC timer state.
#[derive(Debug, Default)]
pub struct ApicTimer {
    pub lvt_timer_bits: u32,
    pub divide_shift: u8,
    pub initial_count: u32,
}

/// TSC (time-stamp counter) tracking for APIC timer.
#[derive(Debug, Default)]
pub struct TscState {
    pub activated: bool,
    pub tsc_physical: u64,
    pub ddl_physical: u64,
}

/// A single I/O APIC redirection table entry.
#[derive(Debug, Default, Clone, Copy)]
pub struct IoapicRedent {
    pub vector: u8,
    /// Delivery mode (3 bits): 000 = Fixed
    pub delivery_mode: u8,
    /// Destination mode: 0 = Physical, 1 = Logical
    pub dest_mode: u8,
    pub delivery_status: u8,
    pub polarity: u8,
    pub remote_irr: bool,
    /// Trigger mode: 0 = Edge, 1 = Level
    pub trigger_mode: bool,
    pub mask: bool,
    /// Target LAPIC ID
    pub dest_id: u8,
}

/// Packed 64-bit redirection table entry (fields + bits view).
#[derive(Debug, Default, Clone, Copy)]
pub struct IoapicRedtbl {
    pub bits: u64,
}

impl IoapicRedtbl {
    pub fn fields(&self) -> IoapicRedent {
        IoapicRedent {
            vector: (self.bits & 0xFF) as u8,
            delivery_mode: ((self.bits >> 8) & 0x7) as u8,
            dest_mode: ((self.bits >> 11) & 0x1) as u8,
            delivery_status: ((self.bits >> 12) & 0x1) as u8,
            polarity: ((self.bits >> 13) & 0x1) as u8,
            remote_irr: ((self.bits >> 14) & 0x1) != 0,
            trigger_mode: ((self.bits >> 15) & 0x1) != 0,
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

/// I/O APIC state.
#[derive(Debug)]
pub struct Ioapic {
    pub ioregsel: u32,
    pub id: u32,
    pub redtbl: [IoapicRedtbl; IOAPIC_NUM_PINS],
}

impl Default for Ioapic {
    fn default() -> Self {
        Self {
            ioregsel: 0,
            id: 0,
            redtbl: [IoapicRedtbl::default(); IOAPIC_NUM_PINS],
        }
    }
}

// ===== Bit manipulation helpers =====

fn lapic_set_bit(val: &mut [u32; 8], vec: u8) {
    val[(vec / 32) as usize] |= 1u32 << (vec % 32);
}

fn lapic_clear_bit(val: &mut [u32; 8], vec: u8) {
    val[(vec / 32) as usize] &= !(1u32 << (vec % 32));
}

fn lapic_find_highest(val: &[u32; 8]) -> Option<u8> {
    for i in (0..8usize).rev() {
        let v = val[i];
        if v != 0 {
            let bit = 31 - v.leading_zeros();
            return Some((i as u32 * 32 + bit) as u8);
        }
    }
    None
}

// ===== LAPIC operations =====

pub fn lapic_find_highest_isr(lapic: &Lapic) -> Option<u8> {
    lapic_find_highest(&lapic.isr)
}

pub fn lapic_find_highest_irr(lapic: &Lapic) -> Option<u8> {
    lapic_find_highest(&lapic.irr)
}

fn lapic_update_ppr(lapic: &mut Lapic) {
    let isr_prio = lapic_find_highest_isr(lapic).map(|v| v & 0xF0).unwrap_or(0) as u8;
    lapic.ppr = lapic.tpr.max(isr_prio);
}

pub fn lapic_set_isr(lapic: &mut Lapic, vec: u8) {
    lapic_set_bit(&mut lapic.isr, vec);
}

pub fn lapic_clear_isr(lapic: &mut Lapic, vec: u8) {
    lapic_clear_bit(&mut lapic.isr, vec);
}

pub fn lapic_set_irr(lapic: &mut Lapic, vec: u8) {
    lapic_set_bit(&mut lapic.irr, vec);
}

pub fn lapic_clear_irr(lapic: &mut Lapic, vec: u8) {
    lapic_clear_bit(&mut lapic.irr, vec);
}

pub fn lapic_set_tmr(lapic: &mut Lapic, vec: u8) {
    lapic_set_bit(&mut lapic.tmr, vec);
}

pub fn lapic_clear_tmr(lapic: &mut Lapic, vec: u8) {
    lapic_clear_bit(&mut lapic.tmr, vec);
}

/// Move an IRR vector into service (set ISR, clear IRR, update PPR).
pub fn lapic_kick_to_service(lapic: &mut Lapic, vec: u8) {
    lapic_set_isr(lapic, vec);
    lapic_clear_irr(lapic, vec);
    lapic_update_ppr(lapic);
}

/// Return the highest-priority pending vector that is deliverable, or `None`.
pub fn lapic_check_pending_vector(lapic: &Lapic) -> Option<u8> {
    let pending_vector = lapic_find_highest_irr(lapic)?;

    let isr_vector = lapic_find_highest_isr(lapic);

    let pending_prio = pending_vector >> 4;
    let tpr_prio = lapic.tpr >> 4;
    let isr_prio = isr_vector.map(|v| v >> 4).unwrap_or(0);

    if pending_prio > tpr_prio && pending_prio > isr_prio {
        Some(pending_vector)
    } else {
        None
    }
}

// ===== APIC timer helpers =====

fn is_periodic_mode(timer: &ApicTimer) -> bool {
    ((timer.lvt_timer_bits >> 17) & 0b11) == 1
}

fn is_deadline_mode(timer: &ApicTimer) -> bool {
    ((timer.lvt_timer_bits >> 17) & 0b11) == 2
}

/// Called by the timer subsystem when the APIC timer deadline is reached.
///
/// Sets the IRR bit for the configured timer vector and, for periodic mode,
/// re-arms the timer.  The caller is responsible for actually re-arming /
/// deactivating the hardware timer (`timer_activate` / `timer_deactivate`).
pub fn lapic_on_timer_expire(
    lapic: &mut Lapic,
    timer: &mut ApicTimer,
    tsc: &TscState,
) -> TimerExpireAction {
    if (timer.lvt_timer_bits & (1 << 16)) == 0 {
        let vector = (timer.lvt_timer_bits & 0xFF) as u8;
        if vector < 32 {
            log::warn!("xLAPIC: Find a timer vector triggered below 32: {}", vector);
        }
        lapic_set_irr(lapic, vector);
    }

    if is_periodic_mode(timer) {
        let next_deadline = tsc
            .tsc_physical
            .wrapping_add((timer.initial_count as u64) << timer.divide_shift);
        TimerExpireAction::Rearm(next_deadline)
    } else {
        TimerExpireAction::Deactivate
    }
}

/// What the caller should do with the hardware timer after `lapic_on_timer_expire`.
pub enum TimerExpireAction {
    /// Re-arm the timer at the given TSC deadline.
    Rearm(u64),
    /// Deactivate the timer.
    Deactivate,
}

// ===== LAPIC MMIO emulation =====

/// Emulate a LAPIC MMIO read.
///
/// Returns `(value, ok)` where `ok` is false if the offset is unsupported.
pub fn emulate_lapic_read(
    lapic: &Lapic,
    timer: &ApicTimer,
    tsc: &TscState,
    offset: u64,
) -> (u64, bool) {
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
        XLAPIC_RW_LVT_CMCI => (1u64 << 16),
        XLAPIC_RW_LVT_THERM | XLAPIC_RW_LVT_PERF | XLAPIC_RW_LVT_LINT0 | XLAPIC_RW_LVT_LINT1
        | XLAPIC_RW_LVT_ERROR => 0x10000,
        XLAPIC_RW_LVT_TIMER => timer.lvt_timer_bits as u64,
        XLAPIC_RW_TIMER_INIT => timer.initial_count as u64,
        XLAPIC_RW_TIMER_DIVI => {
            let dcr = ((timer.divide_shift as u64).wrapping_sub(1)) & 0b0111;
            (dcr & 0b0011) | ((dcr & 0b0100) << 1)
        }
        XLAPIC_RO_TIMER_CURR => {
            if tsc.activated && tsc.ddl_physical > tsc.tsc_physical {
                tsc.ddl_physical - tsc.tsc_physical
            } else {
                0
            }
        }
        XLAPIC_RW_ESR => 0,
        o if o >= XLAPIC_RO_ISR_BASE && o < XLAPIC_RO_ISR_BASE + XLAPIC_RO_ISR_SIZE => {
            lapic.isr[((o - XLAPIC_RO_ISR_BASE) / 16) as usize] as u64
        }
        o if o >= XLAPIC_RO_IRR_BASE && o < XLAPIC_RO_IRR_BASE + XLAPIC_RO_IRR_SIZE => {
            lapic.irr[((o - XLAPIC_RO_IRR_BASE) / 16) as usize] as u64
        }
        o if o >= XLAPIC_RW_ICR_BASE && o < XLAPIC_RW_ICR_BASE + XLAPIC_RW_ICR_SIZE => {
            lapic.icr[((o - XLAPIC_RW_ICR_BASE) / 16) as usize] as u64
        }
        o if o >= XLAPIC_RO_TMR_BASE && o < XLAPIC_RO_TMR_BASE + XLAPIC_RO_TMR_SIZE => {
            lapic.icr[((o - XLAPIC_RO_TMR_BASE) / 16) as usize] as u64
        }
        _ => {
            log::warn!("MMIO.xLAPIC: Read at offset {:#05x} not supported", offset);
            return (0, false);
        }
    };
    (value, true)
}

/// Result of a LAPIC write that may require a timer action.
pub enum LapicWriteEffect {
    None,
    StartTimer,
    StartTimerDeadline,
}

/// Emulate a LAPIC MMIO write.
///
/// Returns the side-effect the caller must act on, and `ok` (false = unsupported offset).
pub fn emulate_lapic_write(
    lapic: &mut Lapic,
    timer: &mut ApicTimer,
    ioapic: &mut Ioapic,
    offset: u64,
    value: u64,
) -> (LapicWriteEffect, bool) {
    match offset {
        XLAPIC_RW_ID => {
            let new_apic_id = ((value >> 24) & 0xFF) as u32;
            log::info!(
                "MMIO.xLAPIC: Guest requests to overwrite apic_id {} to {}",
                lapic.id,
                new_apic_id
            );
            lapic.id = new_apic_id;
        }
        XLAPIC_RW_TPR => {
            lapic.tpr = (value & 0xFF) as u8;
        }
        XLAPIC_WO_EOI => {
            // Find highest in-service vector and complete it
            if let Some(isr_vec) = lapic_find_highest_isr(lapic) {
                lapic_clear_isr(lapic, isr_vec);
                lapic_update_ppr(lapic);

                // Re-assert any level-triggered IRQ that is still pending
                if let Some(irr_vec) = lapic_find_highest_irr(lapic) {
                    if (irr_vec & 0xF0) > lapic.ppr {
                        lapic_clear_irr(lapic, irr_vec);
                        lapic_set_irr(lapic, irr_vec);
                        lapic_update_ppr(lapic);
                    }
                }

                ioapic_eoi(ioapic, isr_vec);
            }
        }
        XLAPIC_RW_LDR => {
            let new = (value as u32) & 0xFF00_0000;
            // Accept only a single-bit logical ID
            if new != 0 && (new & new.wrapping_sub(1 << 24)) == 0 {
                lapic.ldr = new;
                log::debug!(
                    "vLAPIC: LDR set to {:#010x}, LAPIC.ID={:#010x}",
                    lapic.ldr,
                    lapic.id >> 24
                );
            }
        }
        XLAPIC_RW_DFR => {
            if ((value >> 28) & 0xF) != 0xF {
                log::warn!("vLAPIC: Unsupported cluster model, ignore");
            }
        }
        XLAPIC_RW_SIVR | XLAPIC_RW_LVT_CMCI | XLAPIC_RW_LVT_THERM | XLAPIC_RW_LVT_PERF
        | XLAPIC_RW_LVT_LINT0 | XLAPIC_RW_LVT_LINT1 | XLAPIC_RW_LVT_ERROR => { /* silently ignored */
        }
        XLAPIC_RW_LVT_TIMER => {
            let timer_mode = (value >> 17) & 0b11;
            timer.lvt_timer_bits = value as u32;
            let effect = match timer_mode {
                0 | 1 => LapicWriteEffect::StartTimer, // One-shot or Periodic
                2 => LapicWriteEffect::StartTimerDeadline, // TSC-Deadline
                _ => {
                    log::warn!("MMIO.xLAPIC: Write a reserved timer mode");
                    LapicWriteEffect::None
                }
            };
            return (effect, true);
        }
        XLAPIC_RW_TIMER_INIT => {
            timer.initial_count = value as u32;
            return (LapicWriteEffect::StartTimer, true);
        }
        XLAPIC_RW_TIMER_DIVI => {
            let shift = (value & 0b11) | ((value & 0b1000) >> 1);
            timer.divide_shift = (((shift + 1) & 0b111) as u8);
            return (LapicWriteEffect::StartTimer, true);
        }
        XLAPIC_RW_ESR => {
            if value != 0 {
                log::warn!(
                    "MMIO.xLAPIC: Write to xLAPIC_RW_ESR with non-zero value {:#018x}",
                    value
                );
            }
        }
        o if o >= XLAPIC_RW_ICR_BASE && o < XLAPIC_RW_ICR_BASE + XLAPIC_RW_ICR_SIZE => {
            lapic.icr[((o - XLAPIC_RW_ICR_BASE) / 16) as usize] = value as u32;
        }
        _ => {
            log::warn!(
                "MMIO.xLAPIC: Write at offset {:#05x} not supported, value is {:#018x}",
                offset,
                value
            );
            return (LapicWriteEffect::None, false);
        }
    }
    (LapicWriteEffect::None, true)
}

// ===== I/O APIC emulation =====

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
    log::warn!("IOAPIC: Read invalid offset {:#x}", offset);
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
                ioapic.redtbl[pin].bits &= 0xFFFF_FFFF_0000_0000;
                ioapic.redtbl[pin].bits |= value & 0xFFFF_FFFF;
            }
        }
        return true;
    }
    log::warn!("IOAPIC: Write invalid offset {:#x}", offset);
    false
}

/// Deliver an IRQ from the I/O APIC to the appropriate vCPU's LAPIC.
///
/// Mirrors `ioapic_kick_irq` from the C implementation.
/// `lapics` is a slice of (`lapic_id`, mutable `Lapic`) pairs, one per vCPU.
pub fn ioapic_kick_irq(ioapic: &mut Ioapic, lapics: &mut [(&u32, &mut Lapic)], irq: usize) {
    if irq >= IOAPIC_NUM_PINS {
        return;
    }

    let entry = ioapic.redtbl[irq].fields();

    if entry.mask {
        return;
    }
    if entry.remote_irr {
        return;
    }

    let vec = entry.vector;
    if vec < 0x10 || vec > 0xFE {
        return;
    }

    if entry.delivery_mode != 0b000 {
        log::warn!(
            "vIOAPIC: Unhandled delivery mode: {:03b}, ignore",
            entry.delivery_mode
        );
    }

    let destination = entry.dest_id;
    if entry.dest_mode == 0 {
        // Physical mode: send to a specific LAPIC
        for (lapic_id, lapic) in lapics.iter_mut() {
            if (**lapic_id >> 24) as u8 == destination {
                lapic_set_irr(lapic, vec);
                break;
            }
        }
    } else {
        // Logical mode (flat model): send to a group of LAPICs
        for (_, lapic) in lapics.iter_mut() {
            if ((lapic.ldr >> 24) as u8) & destination != 0 {
                lapic_set_irr(lapic, vec);
            }
        }
    }

    if entry.trigger_mode {
        ioapic.redtbl[irq].set_remote_irr(true);
    }
}

/// Clear the `remote_irr` flag for any redirection entry whose vector matches `vec`.
///
/// Mirrors `ioapic_eoi` from the C implementation.
pub fn ioapic_eoi(ioapic: &mut Ioapic, vec: u8) {
    for irq in 0..IOAPIC_NUM_PINS {
        if ioapic.redtbl[irq].fields().vector == vec {
            ioapic.redtbl[irq].set_remote_irr(false);
        }
    }
}
