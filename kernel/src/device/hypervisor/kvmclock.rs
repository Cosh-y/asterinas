use super::{
    msr::{
        MSR_KVM_SYSTEM_TIME, MSR_KVM_SYSTEM_TIME_NEW, MSR_KVM_WALL_CLOCK, MSR_KVM_WALL_CLOCK_NEW,
    },
    vm::{Vm, realtime_nanos},
};
use crate::prelude::*;

const KVM_MSR_ENABLED: u64 = 1;
const NSEC_PER_SEC: u64 = 1_000_000_000;
const PVCLOCK_FLAGS_NONE: u8 = 0;

#[derive(Debug, Default)]
pub(super) struct KvmClock {
    old_wall_clock_msr: u64,
    old_system_time_msr: u64,
    new_wall_clock_msr: u64,
    new_system_time_msr: u64,
    wall_clock_version: u32,
    system_time_version: u32,
}

/// Pvclock structure per-vCPU.
///
/// It's GPA is written to MSR_KVM_SYSTEM_TIME_NEW by the guest,
/// and the hypervisor writes the current time to it.
///
/// To calculate the current kvmclock time, the guest does the following:
/// ```
/// delta = rdtsc() - tsc_timestamp;
/// if (tsc_shift >= 0)
///     delta <<= tsc_shift;
/// else
///     delta >>= -tsc_shift;
/// elapsed_ns = (delta * tsc_to_system_mul) >> 32;
///
/// current_system_time = system_time + elapsed_ns;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
struct PvclockVcpuTimeInfo {
    // Version of current time.
    // Guest reads the version before and after reading the time info.
    // Only if the version is same and even, the time info is valid.
    version: u32,
    pad0: u32,
    // Guest TSC at baseline time.
    tsc_timestamp: u64,
    // Guest kvmclock time at baseline time. Unit in nanoseconds.
    system_time: u64,
    tsc_to_system_mul: u32,
    tsc_shift: i8,
    flags: u8,
    pad: [u8; 2],
}

/// Pvclock structure per-VM for wall clock time.
///
/// It's GPA is written to MSR_KVM_WALL_CLOCK_NEW by the guest,
/// and the hypervisor updates it when MSR_KVM_WALL_CLOCK_NEW is written
/// by the guest.
///
/// Record the actual calendar time when the system_time of the Guest is
/// equal to 0.
///
/// To calculate the current calender time, the guest does the following:
/// ```
/// current_wall_clock_time = kvmclock_wall_clock() + current_system_time;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
struct PvclockWallClock {
    version: u32,
    sec: u32,
    nsec: u32,
}

impl KvmClock {
    pub(super) fn read_msr(&self, index: u32) -> u64 {
        match index {
            MSR_KVM_WALL_CLOCK => self.old_wall_clock_msr,
            MSR_KVM_SYSTEM_TIME => self.old_system_time_msr,
            MSR_KVM_WALL_CLOCK_NEW => self.new_wall_clock_msr,
            MSR_KVM_SYSTEM_TIME_NEW => self.new_system_time_msr,
            _ => {
                error!(
                    "hypervisor: ignoring guest read from unknown KVM clock MSR {:#x}",
                    index
                );
                0
            }
        }
    }

    pub(super) fn write_msr(
        &mut self,
        index: u32,
        value: u64,
        vm: &Vm,
        guest_tsc: u64,
    ) -> Result<()> {
        match index {
            MSR_KVM_WALL_CLOCK => {
                self.old_wall_clock_msr = value;
                if value != 0 {
                    self.write_wall_clock(vm, value)?;
                }
            }
            MSR_KVM_SYSTEM_TIME => {
                self.old_system_time_msr = value;
                if value & KVM_MSR_ENABLED != 0 {
                    self.write_system_time(vm, value & !KVM_MSR_ENABLED, guest_tsc)?;
                }
            }
            MSR_KVM_WALL_CLOCK_NEW => {
                self.new_wall_clock_msr = value;
                if value != 0 {
                    self.write_wall_clock(vm, value)?;
                }
            }
            MSR_KVM_SYSTEM_TIME_NEW => {
                self.new_system_time_msr = value;
                if value & KVM_MSR_ENABLED != 0 {
                    self.write_system_time(vm, value & !KVM_MSR_ENABLED, guest_tsc)?;
                }
            }
            _ => error!(
                "hypervisor: ignoring guest write to unknown KVM clock MSR {:#x}",
                index
            ),
        }

        Ok(())
    }

    fn write_system_time(&mut self, vm: &Vm, gpa: u64, guest_tsc: u64) -> Result<()> {
        let (mul, shift) = tsc_to_system_scale()?;
        let odd_version = next_odd_version(self.system_time_version);
        let even_version = next_even_version(odd_version);
        vm.memory().write_val(gpa as usize, &odd_version)?;

        let time_info = PvclockVcpuTimeInfo {
            version: even_version,
            pad0: 0,
            tsc_timestamp: guest_tsc,
            system_time: vm.kvmclock_nanos(),
            tsc_to_system_mul: mul,
            tsc_shift: shift,
            flags: PVCLOCK_FLAGS_NONE,
            pad: [0; 2],
        };
        vm.memory().write_val(gpa as usize, &time_info)?;
        self.system_time_version = even_version;
        Ok(())
    }

    fn write_wall_clock(&mut self, vm: &Vm, gpa: u64) -> Result<()> {
        let monotonic = vm.kvmclock_nanos();
        let wall_clock_nanos = realtime_nanos()?.saturating_sub(monotonic);
        let sec = (wall_clock_nanos / NSEC_PER_SEC).min(u64::from(u32::MAX)) as u32;
        let nsec = (wall_clock_nanos % NSEC_PER_SEC) as u32;

        let odd_version = next_odd_version(self.wall_clock_version);
        let even_version = next_even_version(odd_version);
        vm.memory().write_val(gpa as usize, &odd_version)?;

        let wall_clock = PvclockWallClock {
            version: even_version,
            sec,
            nsec,
        };
        vm.memory().write_val(gpa as usize, &wall_clock)?;
        self.wall_clock_version = even_version;
        Ok(())
    }

    pub(super) fn update_system_time(&mut self, vm: &Vm, guest_tsc: u64) -> Result<()> {
        let old_system_time_msr = self.old_system_time_msr;
        if old_system_time_msr & KVM_MSR_ENABLED != 0 {
            self.write_system_time(vm, old_system_time_msr & !KVM_MSR_ENABLED, guest_tsc)?;
        }

        let new_system_time_msr = self.new_system_time_msr;
        if new_system_time_msr & KVM_MSR_ENABLED != 0 {
            self.write_system_time(vm, new_system_time_msr & !KVM_MSR_ENABLED, guest_tsc)?;
        }
        Ok(())
    }
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
