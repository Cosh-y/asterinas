use alloc::sync::Weak;
use aster_input::event_type_codes::KeyCode::{M, N};
use ostd::vm::Gpaddr;

use crate::{
    kvm::vm::Vm,
    prelude::*,
};

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

    pub(super) fn mmio_range(&self) -> Range<usize> {
        self.base_address..self.base_address + 0x20
    }

    pub(super) fn emulate_mmio_read(&self, gpaddr: Gpaddr) -> Result<u32> {
        //
    }

    pub(super) fn emulate_mmio_write(&mut self, gpaddr: Gpaddr, value: u32) -> Result<()> {
        //
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
            .values()
            .filter_map(Weak::upgrade)
            .find(|vcpu| vcpu.arch().lapic().apic_id == apic_id)
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
                        .values()
                        .filter_map(Weak::upgrade)
                        .find(|vcpu| vcpu.arch().lapic().apic_id == apic_id)
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
    apic_base: u64,

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

    lvt_lint0: u32,

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

    deadline_tsc: Option<u64>,
}

#[repr(u32)]
#[derive(TryFromInt, PartialEq, Eq)]
pub(super) enum DeliveryMode {
    Fixed = 0,
    LowestPriority = 1,
    SMI = 2,
    NMI = 4,
    INIT = 5,
    Startup = 6,
    ExtInt = 7,
}

impl Lapic {
    /// Creates a local APIC with no pending interrupts and a masked, stopped timer.
    pub(super) fn new(apic_id: u32) -> Self {
        Self {
            apic_id,
            apic_base: 0,
            irr: [0; 8],
            isr: [0; 8],
            tmr: [0; 8],
            tpr: 0,
            lvt_lint0: 0,
            timer: LapicTimer::new(),
        }
    }

    pub(super) fn set_apic_base(&mut self, apic_base: u64) {
        self.apic_base = apic_base;
    }


    pub(super) fn can_inject_external_interrupt(&self) -> bool {
        (!self.is_lint0_masked()
            && self.lint0_delivery_mode() == DeliveryMode::ExtInt)
            || self.apic_base & APIC_ENABLE == 0
    }

    fn is_lint0_masked(&self) -> bool {
        self.lvt_lint0 & (1 << 16) != 0
    }

    fn lint0_delivery_mode(&self) -> DeliveryMode {
        DeliveryMode::try_from((self.lvt_lint0 >> 8) & 0x7).unwrap()
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
            0x832 | 0x838..=0x83E => {
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
            // LINT0
            0x835 => {
                const WRITABLE_MASK: u64 = 0x7ff | (1 << 13) | (1 << 15) | (1 << 16);

                self.lvt_lint0 = (value & WRITABLE_MASK) as u32;
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
            0x832 | 0x838..=0x83E => {
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
            // LINT0
            0x835 => {
                Ok(self.lvt_lint0 as _)
            }
        }
    }

    /// Actives timer interrupt in lapic and re-calculates next deadline tsc.
    pub(super) fn process_timer_expire(&mut self, cur_guest_tsc: u64) {
        let Some(deadline) = self.timer.deadline_tsc else {
            return;
        };

        if tsc_delta(deadline, cur_guest_tsc) > 0 {
            return;
        }

        if !self.timer.is_masked() {
            self.receive_irq(self.timer.vector(), false);
        }

        if self.timer.is_period_mode() {
            let tsc_freq = ostd::arch::tsc_freq();
            let timer = &mut self.timer;
            let numerator = timer.initial_count as u128 * timer.divide_factor() as u128 * tsc_freq as u128;
            let period = numerator.div_ceil(timer.apic_freq as u128) as u64;
            let gap = cur_guest_tsc.wrapping_sub(timer.deadline_tsc.unwrap());
            let until_next = period - gap % period;
            timer.deadline_tsc = Some(cur_guest_tsc.wrapping_add(until_next));
        } else {
            self.timer.deadline_tsc = None;
        }
    }

    pub(super) fn timer(&self) -> &LapicTimer {
        &self.timer
    }

    pub(super) fn timer_mut(&mut self) -> &mut LapicTimer {
        &mut self.timer
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
            deadline_tsc: None,
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
                    self.deadline_tsc = None;
                    self.initial_count = 0;
                    return Ok(())
                }

                // Currently, guest's tsc freq is equal to host's tsc freq.
                let tsc_freq = ostd::arch::tsc_freq();
                let numerator = value as u128 * self.divide_factor() as u128 * tsc_freq as u128;
                let delta_tsc = numerator.div_ceil(self.apic_freq as u128);
                let delta_tsc = u64::try_from(delta_tsc)
                    .map_err(|_| { Error::with_message(Errno::EFAULT, "bad lapic timer initial count") })?;
                self.deadline_tsc = Some(cur_guest_tsc.wrapping_add(delta_tsc));
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

                if self.deadline_tsc.is_some() && new_factor != old_factor {
                    let gap_tsc = tsc_delta(self.deadline_tsc.unwrap(), cur_guest_tsc).max(0) as u64;
                    let new_gap_tsc = (gap_tsc as u128 * new_factor as u128).div_ceil(old_factor as u128) as u64;
                    self.deadline_tsc = Some(cur_guest_tsc.wrapping_add(new_gap_tsc));
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
                let Some(deadline_tsc) = self.deadline_tsc else {
                    return Ok(0);
                };

                let gap_tsc = tsc_delta(deadline, cur_guest_tsc).max(0) as u64;

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
        self.deadline_tsc
    }

    pub(super) fn shift_deadline_tsc(&mut self, shift: u64) {
        if let Some(deadline_tsc) = self.deadline_tsc.as_mut() {
            *deadline_tsc = deadline_tsc.wrapping_add(shift);
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

// Assume the delta between `deadline` and `current` is less then 
// half of `u64::MAX`. So the deadline is in futrue when return 
// value is positive.
fn tsc_delta(deadline: u64, current: u64) -> i64 {
    deadline.wrapping_sub(current) as i64
}

pub(super) struct Pic8259 {
    // master
    // slave INT 连接 master IR2 pin
    // master INT 连接 Lapic LINT0 pin
    // 连接到每一个 Lapic 的 LINT0，只在 Lapic 被关闭 或 LINT0 未屏蔽且为 ExtINT 时注入
    master: Pic,

    // slave
    slave: Pic,
}

#[derive(Default)]
pub(super) struct Pic {
    // IR0-IR7's bitmap.
    ir: u8,
    vector_base: u8,
    
    irr: u8,
    isr: u8,
    elcr: u8, // 1 for level triggered.
    imr: u8,

    // 0x20 0x21
    // 0xA0 0xA1
    phase: PicPhase,
    read_isr: bool,
    
    // Note: 当前实现按固定配置处理，不记录 Guest 的配置（除了 vector_base）
    // 固定的配置包括 EOI 模式、是否 cascade、连接主从的 IR pin、ICW4 等
    // icw1: u8,
    // icw2: u8,
    // icw3: u8,
    // icw4: u8,

    // INT 中断请求输出
}

#[expect(non_camel_case_types)]
enum PicPhase {
    EXPECT_ICW2,
    EXPECT_ICW3,
    EXPECT_ICW4,
    WORK,
}

impl Pic8259 {
    pub(super) fn new() -> Self {
        Self {
            master: Pic::default(),
            slave: Pic::default(),
        }
    }

    // receive irq
    pub(super) fn set_ir(&mut self, pin: usize, is_asserted: bool) {
        debug_assert!(pin < 16);

        let pic = if pin < 8 {
            &mut self.master
        } else {
            &mut self.slave
        };
        let mask = 1 << (pin % 8);

        let was_asserted = pic.ir & mask != 0;
        let is_level_triggered = pic.elcr & mask != 0;
        if is_asserted {
            pic.ir |= mask;
        } else {
            pic.ir &= !mask;
        }

        if is_level_triggered {
            if is_asserted {
                pic.irr |= mask;
            } else {
                pic.irr &= !mask;
            }
        } else if !was_asserted && is_asserted {
            pic.irr |= mask;
        }

        if pin >= 8 {
            self.sync_master();
        }
    }

    fn sync_master(&mut self) {
        if self.slave.quiry_irq().is_some() {
            self.set_ir(2, true);
            self.set_ir(2, false);
        }
    }

    pub(super) fn quiry_irq(&self) -> Option<u8> {
        let Some(master_vector) = self.master.quiry_irq() else {
            return None;
        };
        if master_vector == self.master.vector_base + 2 {
            return self.slave.quiry_irq().or(Some(self.slave.vector_base + 7));
        }
        Some(master_vector)
    }

    pub(super) fn inject_irq(&mut self) -> Option<u8> {
        // irr -> isr
        let Some(vector) = self.master.inject_irq() else {
            return None;
        };
        if vector == self.master.vector_base + 2 {
            return self.slave.inject_irq().or(Some(self.slave.vector_base + 7));
        }
        Some(vector)
    }

    pub(super) fn receive_eoi_master(&mut self, pin: Option<usize>) {
        self.master.receive_eoi(pin);
        self.sync_master();
    }

    pub(super) fn receive_eoi_slave(&mut self, pin: Option<usize>) {
        self.slave.receive_eoi(pin);
        self.sync_master();
    }

    pub(super) fn emulate_io_write(&mut self, port: u16, value: u8) {
        match port {
            0x20 => self.master.write_command(value),
            0x21 => self.master.write_data(value),
            0xa0 => self.slave.write_command(value),
            0xa1 => self.slave.write_data(value),
            0x4d0 => self.master.elcr = value & 0b1111_1000,
            0x4d1 => self.slave.elcr = value & 0b1101_1110,
            _ => return,
        }
        self.sync_master();
    }

    pub(super) fn emulate_io_read(&self, port: u16) -> u8 {
        match port {
            0x20 => self.master.read_command(),
            0x21 => self.master.read_data(),
            0xa0 => self.slave.read_command(),
            0xa1 => self.slave.read_data(),
            0x4d0 => self.master.elcr,
            0x4d1 => self.slave.elcr,
            _ => 0,
        }
    }
}

impl Pic {
    fn quiry_irq(&self) -> Option<u8> {
        let candidates = self.irr & !self.imr;
        if candidates != 0 {
            if self.isr == 0 {
                return Some(self.vector_base + candidates.trailing_zeros() as u8);
            } else {
                let highest_isr = self.isr.trailing_zeros() as u8;
                let highest_irr = candidates.trailing_zeros() as u8;
                if highest_irr < highest_isr {
                    return Some(self.vector_base + highest_irr);
                }
            }
        }
        None
    }

    fn inject_irq(&mut self) -> Option<u8> {
        let vector = self.quiry_irq()?;
        let pin = (vector - self.vector_base) as usize;
        let mask = 1 << pin;
        if self.elcr & mask == 0 {
            self.irr &= !mask;
        }
        self.isr |= mask;
        Some(vector)
    }

    fn receive_eoi(&mut self, pin: Option<usize>) {
        let pin = pin.unwrap_or_else(|| self.isr.trailing_zeros() as usize);
        self.isr &= !(1 << pin);
    }

    fn write_command(&mut self, value: u8) {
        if value & (1 << 4) != 0 { // ICW1.
            self.phase = PicPhase::EXPECT_ICW2;
        } else if value & (1 << 3) == 0 { // OCW2 EOI.
            match value >> 5 {
                0b001 => self.receive_eoi(None),
                0b011 => self.receive_eoi(Some((value & 0x07) as usize)),
                _ => {}
            }
        } else if value & (1 << 3) != 0 { // OCW3 IRR or ISR.
            if value & (1 << 1) != 0 {
                self.read_isr = value & 1 != 0;
            }
        }
    }

    fn write_data(&mut self, value: u8) {
        match self.phase {
            PicPhase::EXPECT_ICW2 => { // ICW2, vector base.
                self.vector_base = value & 0b1111_1000;
                self.phase = PicPhase::EXPECT_ICW3;
            }
            PicPhase::EXPECT_ICW3 => { // ICW3, cascade IR.
                self.phase = PicPhase::EXPECT_ICW4;
            }
            PicPhase::EXPECT_ICW4 => { // ICW4, mode.
                self.phase = PicPhase::WORK;
            }
            PicPhase::WORK => {
                self.imr = value;
            }
        }
    }

    fn read_command(&self) -> u8 {
        if self.read_isr {
            self.isr
        } else {
            self.irr
        }
    }

    fn read_data(&self) -> u8 {
        self.imr
    }
}

const PIT_FREQ: u64 = 1_193_182;

pub(super) struct Pit {
    vm: Weak<Vm>,
    is_speaker_dummy: bool,
    // channel 0-2
    // channel 0 -> GSI 0 固定
    channels: [PitChannel; 3],
}

enum AccessMode {
    LowOnly,
    HighOnly,
    LowThenHigh,
}

enum BytePhase {
    Low,
    High,
}

struct PitChannel {
    cr: u16,
    // TODO: Is this field necessary?
    host_tsc_at_load: u64,
    // TODO:
    // This is host tsc but `deadline_tsc` is guest tsc in `LapicTimer`.
    deadline_tsc: Option<u64>,

    status: u16,

    access_mode: AccessMode,
    pending_low_byte: Option<u8>,
    write_byte_phase: BytePhase,
    read_byte_phase: BytePhase,

    latched_read_byte_phase: BytePhase,
    latched_count: Option<u16>,
    latched_status: Option<u16>,

    // For mode 0 only.
    frozen_count: Option<u16>,
}

impl Pit {
    pub(super) fn new(vm: Weak<Vm>, is_speaker_dummy: bool) -> Self {
        Self {
            vm,
            is_speaker_dummy,
            channels: [
                PitChannel::new(),
                PitChannel::new(),
                PitChannel::new(),
            ],
        }
    }

    pub(super) fn emulate_io_write(&mut self, port: u16, value: u8) {
        match port {
            // Channel 0-2
            0x40..=0x42 => {
                self.channels[(port - 0x40) as usize].write_count(value);
            }
            // Control Word.
            0x43 => self.write_control(value),
            // Speaker Control (specific to channel 2).
            0x61 => {}
            _ => {}
        }
    }

    pub(super) fn emulate_io_read(&mut self, port: u16) -> u8 {
        match port {
            0x40..=0x42 => {
                self.channels[(port - 0x40) as usize].read_count()
            }
            0x61 => 0,
            _ => 0,
        }
    }

    pub(super) fn process_timer_expire(&mut self) {
        if self.channels[0].deadline_tsc.is_none() {
            return;
        }

        if self.channels[0].deadline_tsc.unwrap() > current_tsc() {
            return;
        }
        
        let vm = self.vm.upgrade().unwrap();
        vm.set_gsi_level(0, 1).unwrap();
        vm.set_gsi_level(0, 0).unwrap();

        let timer = &mut self.channels[0];
        if timer.mode() == 2 {
            let tsc_freq = ostd::arch::tsc_freq();
            let cur_tsc = ostd::arch::read_tsc();
            let period = (timer.initial_count() as u128 * tsc_freq as u128).div_ceil(PIT_FREQ as u128) as u64;
            let gap = cur_tsc.wrapping_sub(timer.deadline_tsc.unwrap());
            let until_next = period - gap % period;
            timer.deadline_tsc = Some(cur_tsc.wrapping_add(until_next));
        } else {
            timer.deadline_tsc = None;
        }
    }

    pub(super) fn quiry_deadline_tsc(&self) -> Option<u64> {
        self.channels[0].deadline_tsc
    }

    fn write_control(&mut self, value: u8) {
        let channel_index = (value >> 6) as usize;

        // Read-back.
        if channel_index == 3 {
            for (index, channel) in self.channels.iter_mut().enumerate() {
                if value & (1 << (index + 1)) == 0 {
                    continue;
                }

                if value & (1 << 5) == 0 {
                    channel.latch_count();
                }
                if value & (1 << 4) == 0 {
                    channel.latch_status();
                }
            }
            return;
        }

        let channel = &mut self.channels[channel_index];
        let access_mode = match (value >> 4) & 3 {
            0 => {
                channel.latch_count();
                return;
            }
            1 => AccessMode::LowOnly,
            2 => AccessMode::HighOnly,
            _ => AccessMode::LowThenHigh,
        };

        channel.program(value, access_mode);
    }
}

impl PitChannel {
    const NULL_COUNT: u16 = 1 << 6;

    fn new() -> Self {
        Self {
            access_mode: AccessMode::LowThenHigh,
            pending_low_byte: None,
            write_byte_phase: BytePhase::Low,
            read_byte_phase: BytePhase::Low,
            latched_read_byte_phase: BytePhase::Low,
            latched_count: None,
            latched_status: None,
            frozen_count: None,
            cr: 0,
            status: Self::NULL_COUNT,
            host_tsc_at_load: 0,
            deadline_tsc: None,
        }
    }

    fn program(&mut self, value: u8, access_mode: AccessMode) {
        self.access_mode = access_mode;

        self.pending_low_byte = None;
        self.write_byte_phase = BytePhase::Low;
        self.read_byte_phase = BytePhase::Low;
        self.latched_read_byte_phase = BytePhase::Low;

        self.latched_count = None;
        self.latched_status = None;

        self.frozen_count = None;

        self.cr = 0;

        // mode 0: OUT is initially low. OUT goes high and remains high when the count expires.
        // mode 2: OUT is initially high. OUT goes low for one clock cycle when the count expires, then returns high.
        //         The counter works periodically.
        // mode 4: OUT is initially high. OUT goes low for one clock cycle when the count expires, then returns high.
        //         The counter works one-shot.
        self.status = u16::from(value & 0x3f) | Self::NULL_COUNT;

        // Initialize OUT.
        match self.mode() {
            0 => {
                self.status &= !0x80;
            }
            2 | 4 => {
                self.status |= 0x80;
            }
            _ => {}
        }
    }

    fn write_count(&mut self, value: u8) {
        match self.access_mode {
            AccessMode::LowOnly => {
                self.write_count_register(u16::from(value));
            }
            AccessMode::HighOnly => {
                self.write_count_register(u16::from(value) << 8);
            }
            AccessMode::LowThenHigh => match self.write_byte_phase {
                BytePhase::Low => {
                    self.pending_low_byte = Some(value);
                    self.write_byte_phase = BytePhase::High;

                    if self.mode() == 0 {
                        self.frozen_count = Some(self.current_count());
                    }
                }
                BytePhase::High => {
                    self.frozen_count = None;
                    let low = self.pending_low_byte.take().unwrap();
                    let count = u16::from_le_bytes([low, value]);

                    self.write_byte_phase = BytePhase::Low;
                    self.write_count_register(count);
                }
            },
        }
    }

    fn write_count_register(&mut self, count: u16) {
        self.cr = count;

        self.status &= !Self::NULL_COUNT;
        self.host_tsc_at_load = ostd::arch::read_tsc();
    }

    fn latch_count(&mut self) {
        if self.latched_count.is_some() {
            return;
        }

        self.latched_count = Some(self.current_count());
        self.latched_read_byte_phase = BytePhase::Low;
    }

    fn latch_status(&mut self) {
        if self.latched_status.is_some() {
            return;
        }

        self.latched_status = Some(self.current_status());
    }

    fn read_count(&mut self) -> u8 {
        if let Some(status) = self.latched_status.take() {
            return status as u8;
        }

        if let Some(count) = self.latched_count {
            let (value, finished) = Self::read_count_byte(
                &self.access_mode,
                &mut self.latched_read_byte_phase,
                count,
            );

            if finished {
                self.latched_count = None;
            }

            return value;
        }

        let count = self.current_count();
        Self::read_count_byte(
            &self.access_mode,
            &mut self.read_byte_phase,
            count,
        )
        .0
    }

    // Return the byte currently read and whether the count has been read in full.
    fn read_count_byte(
        access_mode: &AccessMode,
        phase: &mut BytePhase,
        count: u16,
    ) -> (u8, bool) {
        match access_mode {
            AccessMode::LowOnly => (count as u8, true),
            AccessMode::HighOnly => ((count >> 8) as u8, true),
            AccessMode::LowThenHigh => match phase {
                BytePhase::Low => {
                    *phase = BytePhase::High;
                    (count as u8, false)
                }
                BytePhase::High => {
                    *phase = BytePhase::Low;
                    ((count >> 8) as u8, true)
                }
            },
        }
    }

    fn current_count(&self) -> u16 {
        if let Some(frozen_count) = self.frozen_count {
            return frozen_count;
        }

        let elapsed_tsc = ostd::arch::read_tsc() - self.host_tsc_at_load;
        let elapsed = (elapsed_tsc as u128 * PIT_FREQ as u128 / ostd::arch::tsc_freq() as u128) as u64;
        let initial_count = self.initial_count();
        match self.mode() {
            0 | 4 => self.cr.wrapping_sub(elapsed as u16),
            2 => (initial_count - (elapsed % initial_count)) as u16,
            _ => self.cr,
        }
    }

    fn current_status(&self) -> u16 {
        if self.frozen_count.is_some() {
            return self.status;
        }

        if self.status & Self::NULL_COUNT != 0 {
            return self.status;
        }

        let elapsed_tsc = ostd::arch::read_tsc() - self.host_tsc_at_load;
        let elapsed = (elapsed_tsc as u128 * PIT_FREQ as u128 / ostd::arch::tsc_freq() as u128) as u64;
        let initial_count = self.initial_count();
        let out = match self.mode() {
            0 => (elapsed >= initial_count) as u16,
            2 => (((elapsed % initial_count) as u16) != ((initial_count - 1) as u16)) as u16,
            4 => (elapsed != initial_count) as u16,
            _ => 0,
        };
        (self.status & !0x80) | (out << 7)
    }

    fn initial_count(&self) -> u64 {
        if self.cr == 0 {
            0x10000
        } else {
            self.cr as u64
        }
    }

    fn mode(&self) -> u16 {
        (self.status & 0x0e) >> 1
    }
}
