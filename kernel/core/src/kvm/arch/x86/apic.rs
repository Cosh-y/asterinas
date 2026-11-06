use alloc::sync::Weak;
use aster_input::event_type_codes::KeyCode::{M, N};

use crate::{
    kvm::vm::Vm,
    prelude::*,
};

pub(super) struct Pic8259 {

}

const IOAPIC_NUM_PINS: usize = 24;

pub(super) struct Ioapic {
    // Gpaddr.
    base_address: usize,
    id: u8,
    ioregsel: u8,

    redirtbl: [RedirectEntry; IOAPIC_NUM_PINS],

    // pins' bitmap.
    irr: u32,

    vm: Weak<Vm>,
}

struct RedirectEntry(u64);

impl RedirectEntry {
    fn vector(&self) -> u8 {
        self.0 as u8
    }

    fn destination(&self) -> u8 {
        (self.0 >> 56) as u8
    }

    fn is_masked(&self) -> bool {
        self.0 & (1 << 16) != 0
    }

    fn is_level_triggered(&self) -> bool {
        self.0 & (1 << 15) != 0
    }

    // 电平触发中断已经投递，正在等待 EOI
    fn is_remote_irr(&self) -> bool {
        self.0 & (1 << 14) != 0
    }

    fn set_remote_irr(&mut self, set: bool) {
        let mask = 1 << 14;
        if set {
            self.0 |= mask
        } else {
            self.0 &= !mask
        }
    }
}

impl Ioapic {
    pub(super) fn new(vm: Weak<Vm>) -> Self {
        const IOAPIC_BASE_ADDRESS: usize = 0xFEC0_0000;

        Self {
            base_address: IOAPIC_BASE_ADDRESS,
            id: 1,
            ioregsel: 0,
            redirtbl: [const { RedirectEntry(1 << 16) }; IOAPIC_NUM_PINS],
            irr: 0,
            vm,
        }
    }

    // receive irq
    pub(super) fn set_irq_line(&mut self, pin: usize, is_asserted: bool) -> Result<()> {
        let mask = 1u32 << pin;
        let was_asserted = self.irr & mask != 0;

        if is_asserted {
            self.irr |= mask;
        } else {
            self.irr &= !mask;
            return Ok(());
        }

        let entry = &mut self.redirtbl[pin];

        if entry.is_masked() {
            return Ok(());
        }

        if entry.is_level_triggered() && entry.is_remote_irr() {
            return Ok(());
        }

        if !entry.is_level_triggered() && was_asserted {
            return Ok(());
        }

        // Inject irq to lapic.
        let apic_id = entry.destination() as u32;
        let vm = self.vm.upgrade().unwrap();
        let vcpu = vm
            .vcpus()
            .iter()
            .find(|(vcpu_id, &vcpu)| {
                vcpu.upgrade().unwrap().arch().lapic().apic_id == apic_id
            })
            .map(|(vcpu_id, &vcpu)| {
                vcpu.upgrade().unwrap()
            })
            .ok_or(Error::new(Errno::EFAULT))?;

        let vector = entry.vector();
        let is_level_triggered = entry.is_level_triggered();
        vcpu.arch().lapic().receive_irq(vector, is_level_triggered);

        // 电平触发还需设置 entry 的 is_remote_irr
        if is_level_triggered {
            entry.set_remote_irr(true);
        }

        vcpu.notify();

        Ok(())
    }

    pub(super) fn receive_eoi(&mut self, vector: u8) -> Result<()> {
        for (pin, entry) in self.redirtbl.iter_mut().enumerate() {
            if entry.vector() == vector && entry.is_level_triggered() {
                entry.set_remote_irr(false);

                if self.irr & (1 << pin) != 0 && !entry.is_masked() {
                    let apic_id = entry.destination() as u32;
                    let vm = self.vm.upgrade().unwrap();
                    let vcpu = vm
                        .vcpus()
                        .iter()
                        .find(|(vcpu_id, &vcpu)| {
                            vcpu.upgrade().unwrap().arch().lapic().apic_id == apic_id
                        })
                        .map(|(vcpu_id, &vcpu)| {
                            vcpu.upgrade().unwrap()
                        })
                        .ok_or(Error::new(Errno::EFAULT))?;

                    vcpu.arch().lapic().receive_irq(vector, true);
                    entry.set_remote_irr(true);

                    vcpu.notify();
                }
            }
        }

        Ok(())
    }
}

pub(super) struct Lapic {
    apic_id: u32,

    // Lapic 的寄存器在 xAPIC 下 MMIO；x2APIC 下通过 MSR 访问

    // 256 位寄存器，每一位对应一个 vector
    // irr: 记录 LAPIC 已经接收、但尚未交给 CPU 处理的中断
    // isr: 记录已经交给 CPU、但尚未通过 EOI 宣告完成的中断
    // tmr: 记录对应中断向量的触发方式：边沿触发或电平触发
    //（1 是电平触发，在接收中断时由硬件设置，记录他主要用于判断后续 EOI 时是否要通知 I/O APIC）
    // 他们对软件都是 read only 的
    irr: [u32; 8],
    isr: [u32; 8],
    tmr: [u32; 8],

    // vector 是 8 bit 的数字，高 4 位表示优先级，数字越大优先级越高
    // tpr 的高 4 位是优先级门槛，只有 vector 优先级 > tpr 才能投递
    // ppr: u8  // 只读，取 isr 中最高优先级和 tpr 的大值，最终决定 vector 是否可以投递
    tpr: u8,

    timer: LapicTimer,
}

pub(super) struct LapicTimer {
    /// CPU crystal clock frequency.
    /// 
    /// The divisor selected by `divide_config` determines the rate at which
    /// `current_count` decrements. CPUID leaf 0x15 reports the crystal clock frequency
    /// and the TSC-to-crystal clock ratio.
    apic_freq: u32,

    // Local Vector Table (of timer). 设置 local interrupt 的 “APIC 信息”
    lvt_timer: u32,
    initial_count: u32,
    // current_count: u32,
    // 分频，由 guest 设置
    divide_config: u32,

    deadline_tsc: u64,
}

impl Lapic {
    /// Creates a local APIC with no pending interrupts and a masked, stopped timer.
    pub(super) fn new(apic_id: u32) -> Self {
        Self {
            apic_id,
            irr: [0; 8],
            isr: [0; 8],
            tmr: [0; 8],
            tpr: 0,
            timer: LapicTimer::new(),
        }
    }

    fn receive_irq(&mut self, vector: u8, is_level_triggered: bool) {
        let index = (vector / 32) as usize;
        let mask = 1u32 << (vector % 32);
        self.irr[index] |= mask;
        if is_level_triggered {
            self.tmr[index] |= mask;
        } else {
            self.tmr[index] &= !mask;
        }
    }

    pub(super) fn quiry_irq(&self) -> Option<u8> {
        // Calculate PPR.
        let highest_isr = find_highest_bit(&self.isr).map_or(0, |v| v & 0xf0);
        let ppr = if self.tpr > highest_isr {
            self.tpr
        } else {
            highest_isr
        };

        let highest_irr = find_highest_bit(&self.irr)?;
        if highest_irr & 0xf0 > ppr {
            return Some(highest_irr);
        }

        return None;
    }

    pub(super) fn inject_irq(&mut self) -> Option<u8> {
        // Calculates PPR.
        let highest_isr = find_highest_bit(&self.isr).map_or(0, |v| v & 0xf0);
        let ppr = if self.tpr > highest_isr {
            self.tpr
        } else {
            highest_isr
        };

        let highest_irr = find_highest_bit(&self.irr)?;
        if highest_irr & 0xf0 > ppr {
            let index = (highest_irr / 32) as usize;
            let mask = 1u32 << (highest_irr % 32);
            self.isr[index] |= mask;
            self.irr[index] &= !mask;

            return Some(highest_irr);
        }

        return None;
    }

    pub(super) fn emulate_msr_write(
        &mut self,
        msr: u32,
        value: u64,
        cur_guest_tsc: u64
    ) -> Result<Option<LapicEvent>> {
        match msr {
            // TPR
            0x808 => {
                self.tpr = value as u8;
                Ok(None)
            }
            // Timer
            0x832..=0x83E => {
                self.timer.emulate_msr_write(msr, value, cur_guest_tsc)?;
                Ok(None)
            }
            // EOI
            0x80B => {
                // TODO: Check if `value` is 0?
                // clean highest bit in isr.
                let highest_irr = find_highest_bit(&self.isr)?;
                let index = (highest_irr / 32) as usize;
                let mask = 1u32 << (highest_irr % 32);
                self.isr[index] &= !mask;
                // check tmr, eoi IOAPIC.
                if self.tmr[index] & mask != 0 {
                    return Ok(Some(LapicEvent::Eoi2Ioapic { vector: highest_irr }))
                }
                // inject_irq() will be called in pre_guest_run(), so we don't need to call it here.
                Ok(None)
            }
        }
    }

    pub(super) fn emulate_msr_read(&mut self, msr: u32, cur_guest_tsc: u64) -> Result<u64> {
        match msr {
            // TPR
            0x808 => {
                Ok(self.tpr as _)
            }
            // Timer
            0x832..=0x83E => {
                self.timer.emulate_msr_read(msr, cur_guest_tsc)
            }
            // EOI
            0x80B => {
                // TODO: How to inject #GP(0)?
                // 设计一种返回值表达要注入异常
                // 在 GuestContext 中记录要注入异常的动作
                // 在 pre_guest_run 中注入异常（考虑下是否需要）
                Ok(0)
            }
        }
    }

    /// Actives timer interrupt in lapic and re-calculates next deadline tsc.
    pub(super) fn process_timer_expire(&mut self, cur_guest_tsc: u64) {
        if self.timer.deadline_tsc == 0 || cur_guest_tsc < self.timer.deadline_tsc {
            return;
        }

        if !self.timer.is_masked() {
            self.receive_irq(self.timer.vector(), false);
        }

        if self.timer.is_period_mode() {
            let tsc_freq = ostd::arch::tsc_freq();
            let timer = &mut self.timer;
            let numerator = timer.initial_count as u128 * timer.divide_factor() as u128 * tsc_freq as u128;
            let delta_tsc = numerator.div_ceil(timer.apic_freq as u128);
            let delta_tsc = u64::try_from(delta_tsc).unwrap();
            timer.deadline_tsc += ((cur_guest_tsc - timer.deadline_tsc) / delta_tsc + 1) * delta_tsc;
        } else {
            self.timer.deadline_tsc = 0;
        }
    }

    pub(super) fn timer(&self) -> &LapicTimer {
        &self.timer
    }
}

impl LapicTimer {
    fn new() -> Self {
        const APIC_TIMER_FREQ_HZ: u32 = 1_000_000_000;
        const LVT_MASKED: u32 = 1 << 16;

        Self {
            apic_freq: APIC_TIMER_FREQ_HZ,
            lvt_timer: LVT_MASKED,
            initial_count: 0,
            // A zero divide configuration encodes division by two.
            divide_config: 0,
            deadline_tsc: 0,
        }
    }

    fn emulate_msr_write(&mut self, msr: u32, value: u64, cur_guest_tsc: u64) -> Result<()> {
        match msr {
            // LVT timer.
            0x832 => {
                const WRITABLE_MASK: u64 = 0xff | (1 << 16) | (1 << 17);

                self.lvt_timer = (value & WRITABLE_MASK) as u32;
                Ok(())
            }
            // Initial count
            0x838 => {
                let value = value as u32;
                if value == 0 {
                    self.deadline_tsc = 0;
                    self.initial_count = 0;
                    return Ok(())
                }

                // Currently, guest's tsc freq is equal to host's tsc freq.
                let tsc_freq = ostd::arch::tsc_freq();
                let numerator = value as u128 * self.divide_factor() as u128 * tsc_freq as u128;
                let delta_tsc = numerator.div_ceil(self.apic_freq as u128);
                let delta_tsc = u64::try_from(delta_tsc)
                    .map_err(|_| { Error::with_message(Errno::EFAULT, "bad lapic timer initial count") })?;
                let deadline = cur_guest_tsc.checked_add(delta_tsc)
                    .ok_or(Error::with_message(Errno::EFAULT, "bad lapic timer initial count"))?;
                self.deadline_tsc = deadline;
                self.initial_count = value as u32;
                Ok(())
            }
            // Current count
            0x839 => {
                // writable?
                // TODO: inject #GP(0)
                Ok(())
            }
            // Divide config
            0x83E => {
                const WRITABLE_MASK: u64 = 0xB;

                let old_factor = self.divide_factor();
                self.divide_config = (value & WRITABLE_MASK) as u32;
                let new_factor = self.divide_factor();

                if self.deadline_tsc != 0 && new_factor != old_factor {
                    let Some(gap_tsc) = self.deadline_tsc.checked_sub(cur_guest_tsc) else {
                        return Ok(());
                    };
                    let new_gap_tsc = gap_tsc / old_factor as u64 * new_factor as u64;
                    self.deadline_tsc = cur_guest_tsc + new_gap_tsc;
                }
                Ok(())
            }
        }
    }

    fn emulate_msr_read(&self, msr: u32, cur_guest_tsc: u64) -> Result<u64> {
        match msr {
            // LVT timer.
            0x832 => Ok(self.lvt_timer as u64),
            // Initial count
            0x838 => Ok(self.initial_count as u64),
            // Current count
            0x839 => {
                let Some(gap_tsc) = self.deadline_tsc.checked_sub(cur_guest_tsc) else {
                    return Ok(0);
                };

                let tsc_freq = ostd::arch::tsc_freq();
                let numerator = gap_tsc as u128 * self.apic_freq as u128;
                let count = numerator / self.divide_factor() as u128 / tsc_freq as u128;
                let count = u32::try_from(count)
                    .map_err(|_| { Error::with_message(Errno::EFAULT, "current_count calculate error") })?;
                Ok(count as u64)
            }
            // Divide config
            0x83E => Ok(self.divide_config as u64),
        }
    }

    pub(super) fn quiry_deadline_tsc(&self) -> Option<u64> {
        if self.deadline_tsc != 0 {
            return Some(self.deadline_tsc);
        } else {
            return None;
        }
    }

    fn is_period_mode(&self) -> bool {
        const TIMER_MODE_MASK: u32 = 0x3 << 17;
        const PERIOD_MODE: u32 = 0x1 << 17;
        
        self.lvt_timer & TIMER_MODE_MASK == PERIOD_MODE
    }

    fn is_masked(&self) -> bool {
        self.lvt_timer & (1 << 16) != 0
    }

    fn vector(&self) -> u8 {
        (self.lvt_timer & 0xff) as u8
    }

    fn divide_factor(&self) -> u32 {
        let index =
            (self.divide_config & 0b11)
            | ((self.divide_config >> 1) & 0b100);

        const DIVISORS: [u32; 8] = [2, 4, 8, 16, 32, 64, 128, 1];
        DIVISORS[index as usize]
    }
}

pub(super) enum LapicEvent {
    Eoi2Ioapic {
        vector: u8,
    },
}

fn find_highest_bit(bitmap: &[u32; 8]) -> Option<u8> {
    for (index, &word) in bitmap.iter().enumerate().rev() {
        if word == 0 {
            continue;
        }

        let bit = 31 - word.leading_zeros();
        let vector = index as u32 * 32 + bit;

        return Some(vector as u8)
    }

    None
}
