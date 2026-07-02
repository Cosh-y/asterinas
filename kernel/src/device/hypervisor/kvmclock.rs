use ostd::vm::GuestPhysMemSpace;

use super::vm::{monotonic_nanos, realtime_nanos};
use crate::prelude::*;

pub(super) const MSR_KVM_WALL_CLOCK: u32 = 0x11;
pub(super) const MSR_KVM_SYSTEM_TIME: u32 = 0x12;
pub(super) const MSR_KVM_WALL_CLOCK_NEW: u32 = 0x4b56_4d00;
pub(super) const MSR_KVM_SYSTEM_TIME_NEW: u32 = 0x4b56_4d01;

const KVM_MSR_ENABLED: u64 = 1;
const NSEC_PER_SEC: u64 = 1_000_000_000;
const PVCLOCK_FLAGS_NONE: u8 = 0;

const KVM_CLOCK_MSR_INDEXES: [u32; 4] = [
    MSR_KVM_WALL_CLOCK,
    MSR_KVM_SYSTEM_TIME,
    MSR_KVM_WALL_CLOCK_NEW,
    MSR_KVM_SYSTEM_TIME_NEW,
];

#[derive(Debug, Default)]
pub(super) struct KvmClock {
    old_wall_clock_msr: u64,
    old_system_time_msr: u64,
    new_wall_clock_msr: u64,
    new_system_time_msr: u64,
    wall_clock_version: u32,
    system_time_version: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
struct PvclockVcpuTimeInfo {
    version: u32,
    pad0: u32,
    tsc_timestamp: u64,
    system_time: u64,
    tsc_to_system_mul: u32,
    tsc_shift: i8,
    flags: u8,
    pad: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
struct PvclockWallClock {
    version: u32,
    sec: u32,
    nsec: u32,
}

#[derive(Clone, Copy, Debug)]
enum KvmClockMsr {
    OldWallClock,
    OldSystemTime,
    NewWallClock,
    NewSystemTime,
}

impl KvmClock {
    pub(super) fn read_msr(&self, index: u32) -> Option<u64> {
        let msr = KvmClockMsr::from_index(index)?;
        Some(self.msr_value(msr))
    }

    pub(super) fn write_msr(
        &mut self,
        index: u32,
        value: u64,
        guest_mem: &GuestPhysMemSpace,
        guest_tsc: u64,
    ) -> Result<bool> {
        let Some(msr) = KvmClockMsr::from_index(index) else {
            return Ok(false);
        };

        *self.msr_value_mut(msr) = value;
        match msr {
            KvmClockMsr::OldWallClock | KvmClockMsr::NewWallClock => {
                if value != 0 {
                    self.write_wall_clock(guest_mem, value)?;
                }
            }
            KvmClockMsr::OldSystemTime | KvmClockMsr::NewSystemTime => {
                if value & KVM_MSR_ENABLED != 0 {
                    self.write_system_time(guest_mem, value & !KVM_MSR_ENABLED, guest_tsc)?;
                }
            }
        }

        Ok(true)
    }

    fn msr_value(&self, msr: KvmClockMsr) -> u64 {
        match msr {
            KvmClockMsr::OldWallClock => self.old_wall_clock_msr,
            KvmClockMsr::OldSystemTime => self.old_system_time_msr,
            KvmClockMsr::NewWallClock => self.new_wall_clock_msr,
            KvmClockMsr::NewSystemTime => self.new_system_time_msr,
        }
    }

    fn msr_value_mut(&mut self, msr: KvmClockMsr) -> &mut u64 {
        match msr {
            KvmClockMsr::OldWallClock => &mut self.old_wall_clock_msr,
            KvmClockMsr::OldSystemTime => &mut self.old_system_time_msr,
            KvmClockMsr::NewWallClock => &mut self.new_wall_clock_msr,
            KvmClockMsr::NewSystemTime => &mut self.new_system_time_msr,
        }
    }

    fn write_system_time(
        &mut self,
        guest_mem: &GuestPhysMemSpace,
        gpa: u64,
        guest_tsc: u64,
    ) -> Result<()> {
        let (mul, shift) = tsc_to_system_scale()?;
        let odd_version = next_odd_version(self.system_time_version);
        let even_version = next_even_version(odd_version);
        write_guest_val(guest_mem, gpa, &odd_version)?;

        let time_info = PvclockVcpuTimeInfo {
            version: even_version,
            pad0: 0,
            tsc_timestamp: guest_tsc,
            system_time: monotonic_nanos(),
            tsc_to_system_mul: mul,
            tsc_shift: shift,
            flags: PVCLOCK_FLAGS_NONE,
            pad: [0; 2],
        };
        write_guest_val(guest_mem, gpa, &time_info)?;
        self.system_time_version = even_version;
        Ok(())
    }

    fn write_wall_clock(&mut self, guest_mem: &GuestPhysMemSpace, gpa: u64) -> Result<()> {
        let monotonic = monotonic_nanos();
        let wall_clock_nanos = realtime_nanos()?.saturating_sub(monotonic);
        let sec = (wall_clock_nanos / NSEC_PER_SEC).min(u64::from(u32::MAX)) as u32;
        let nsec = (wall_clock_nanos % NSEC_PER_SEC) as u32;

        let odd_version = next_odd_version(self.wall_clock_version);
        let even_version = next_even_version(odd_version);
        write_guest_val(guest_mem, gpa, &odd_version)?;

        let wall_clock = PvclockWallClock {
            version: even_version,
            sec,
            nsec,
        };
        write_guest_val(guest_mem, gpa, &wall_clock)?;
        self.wall_clock_version = even_version;
        Ok(())
    }
}

impl KvmClockMsr {
    fn from_index(index: u32) -> Option<Self> {
        match index {
            MSR_KVM_WALL_CLOCK => Some(Self::OldWallClock),
            MSR_KVM_SYSTEM_TIME => Some(Self::OldSystemTime),
            MSR_KVM_WALL_CLOCK_NEW => Some(Self::NewWallClock),
            MSR_KVM_SYSTEM_TIME_NEW => Some(Self::NewSystemTime),
            _ => None,
        }
    }
}

pub(super) fn is_kvmclock_msr(index: u32) -> bool {
    KvmClockMsr::from_index(index).is_some()
}

pub(super) fn msr_indices() -> &'static [u32] {
    &KVM_CLOCK_MSR_INDEXES
}

fn write_guest_val<T: Pod>(guest_mem: &GuestPhysMemSpace, gpa: u64, value: &T) -> Result<()> {
    let gpa = usize::try_from(gpa)?;
    let mut writer = guest_mem.writer(gpa, size_of::<T>())?;
    writer.write_val(value)?;
    Ok(())
}

fn tsc_to_system_scale() -> Result<(u32, i8)> {
    let freq_hz = ostd::arch::tsc_freq();
    if freq_hz == 0 {
        return_errno_with_message!(Errno::EINVAL, "TSC frequency is not available");
    }

    let numerator = u128::from(NSEC_PER_SEC) << 32;
    let mut shift = 0_u32;
    loop {
        let denominator = u128::from(freq_hz) << shift;
        let mul = numerator / denominator;
        if mul != 0 && mul <= u128::from(u32::MAX) {
            return Ok((mul as u32, shift as i8));
        }
        if shift >= i8::MAX as u32 {
            return_errno_with_message!(Errno::EINVAL, "TSC frequency cannot be scaled");
        }
        shift += 1;
    }
}

fn next_odd_version(version: u32) -> u32 {
    version.wrapping_add(1) | 1
}

fn next_even_version(version: u32) -> u32 {
    version.wrapping_add(1) & !1
}
