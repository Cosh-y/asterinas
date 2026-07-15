// SPDX-License-Identifier: MPL-2.0

//! Emulates guest MSR accesses in the kernel hypervisor layer.

use ostd::arch::{
    read_tsc,
    vm::{GuestExitInfo, X86GprIndex},
};

use super::{cr, ioctl::IA32_TSC_DEADLINE, vcpu::Vcpu};
use crate::prelude::*;

const IA32_TSC: u32 = 0x10;
const IA32_APIC_BASE: u32 = 0x1b;
const IA32_TSC_ADJUST: u32 = 0x3b;
const IA32_BIOS_SIGN_ID: u32 = 0x8b;
const IA32_SYSENTER_CS: u32 = 0x174;
const IA32_SYSENTER_ESP: u32 = 0x175;
const IA32_SYSENTER_EIP: u32 = 0x176;
const IA32_MISC_ENABLE: u32 = 0x1a0;
const IA32_PAT: u32 = 0x277;
const IA32_EFER: u32 = 0xc000_0080;
const IA32_STAR: u32 = 0xc000_0081;
const IA32_LSTAR: u32 = 0xc000_0082;
const IA32_CSTAR: u32 = 0xc000_0083;
const IA32_FMASK: u32 = 0xc000_0084;
const IA32_FS_BASE: u32 = 0xc000_0100;
const IA32_GS_BASE: u32 = 0xc000_0101;
const IA32_KERNEL_GSBASE: u32 = 0xc000_0102;
const IA32_TSC_AUX: u32 = 0xc000_0103;
// KVM-specific MSRs for paravirtualized clock support. See:
// https://www.kernel.org/doc/html/latest/virt/kvm/x86/msr.html
pub(super) const MSR_KVM_WALL_CLOCK: u32 = 0x11;
pub(super) const MSR_KVM_SYSTEM_TIME: u32 = 0x12;
pub(super) const MSR_KVM_WALL_CLOCK_NEW: u32 = 0x4b56_4d00;
pub(super) const MSR_KVM_SYSTEM_TIME_NEW: u32 = 0x4b56_4d01;

#[derive(Clone, Copy, Debug)]
pub(super) enum MsrAccess {
    Read,
    Write,
}

/// Returns the MSR indexes emulated by the kernel hypervisor layer.
pub(super) fn msr_indices() -> Vec<u32> {
    Vec::from([
        IA32_TSC,
        IA32_APIC_BASE,
        IA32_TSC_ADJUST,
        IA32_BIOS_SIGN_ID,
        IA32_SYSENTER_CS,
        IA32_SYSENTER_ESP,
        IA32_SYSENTER_EIP,
        IA32_MISC_ENABLE,
        IA32_PAT,
        IA32_TSC_DEADLINE,
        IA32_EFER,
        IA32_STAR,
        IA32_LSTAR,
        IA32_CSTAR,
        IA32_FMASK,
        IA32_FS_BASE,
        IA32_GS_BASE,
        IA32_KERNEL_GSBASE,
        IA32_TSC_AUX,
        MSR_KVM_WALL_CLOCK,
        MSR_KVM_SYSTEM_TIME,
        MSR_KVM_WALL_CLOCK_NEW,
        MSR_KVM_SYSTEM_TIME_NEW,
    ])
}

/// Emulates a VM-exit caused by RDMSR or WRMSR.
pub(super) fn emulate_msr(vcpu: &Vcpu, exit_info: &GuestExitInfo, access: MsrAccess) -> Result<()> {
    let (msr_index, msr_value) = {
        let context = vcpu.guest_context();
        let msr_index = context.gpr(X86GprIndex::Rcx) as u32;
        let msr_value = (context.gpr(X86GprIndex::Rax) as u32 as u64)
            | ((context.gpr(X86GprIndex::Rdx) as u32 as u64) << 32);
        (msr_index, msr_value)
    };

    match access {
        MsrAccess::Read => {
            let value = read_msr(vcpu, msr_index);
            let mut context = vcpu.guest_context();
            context.set_gpr(X86GprIndex::Rax, 8, value as u32 as u64);
            context.set_gpr(X86GprIndex::Rdx, 8, value >> 32);
        }
        MsrAccess::Write => {
            write_msr(vcpu, msr_index, msr_value)?;
        }
    }

    vcpu.guest_context()
        .advance_rip(u64::from(exit_info.instruction_len));
    Ok(())
}

/// Reads an emulated MSR for ioctl or guest instruction handling.
pub(super) fn read_msr(vcpu: &Vcpu, index: u32) -> u64 {
    let context = vcpu.guest_context();
    match index {
        MSR_KVM_WALL_CLOCK
        | MSR_KVM_SYSTEM_TIME
        | MSR_KVM_WALL_CLOCK_NEW
        | MSR_KVM_SYSTEM_TIME_NEW => vcpu.read_kvmclock_msr(index),
        IA32_TSC_DEADLINE => vcpu.lapic().read_tsc_deadline_msr(),
        IA32_TSC => context.guest_tsc(),
        IA32_BIOS_SIGN_ID => 0,
        _ => context.read_msr(index).unwrap_or(0),
    }
}

/// Writes an emulated MSR for ioctl or guest instruction handling.
pub(super) fn write_msr(vcpu: &Vcpu, index: u32, value: u64) -> Result<()> {
    match index {
        MSR_KVM_WALL_CLOCK
        | MSR_KVM_SYSTEM_TIME
        | MSR_KVM_WALL_CLOCK_NEW
        | MSR_KVM_SYSTEM_TIME_NEW => {
            vcpu.write_kvmclock_msr(index, value)?;
        }
        IA32_TSC_DEADLINE => {
            vcpu.lapic().write_tsc_deadline_msr(value);
        }
        IA32_TSC => {
            let offset = (value as i64).wrapping_sub(read_tsc() as i64);
            vcpu.guest_context().set_tsc_offset(offset);
            vcpu.update_kvmclock()?;
        }
        IA32_TSC_ADJUST => {
            let tsc_offset_changed = {
                let mut context = vcpu.guest_context();
                let old_value = context.read_msr(IA32_TSC_ADJUST).unwrap_or(0);
                if context.write_msr(IA32_TSC_ADJUST, value) {
                    let delta = (value as i64).wrapping_sub(old_value as i64);
                    context.adjust_tsc_offset(delta);
                    delta != 0
                } else {
                    false
                }
            };
            if tsc_offset_changed {
                vcpu.update_kvmclock()?;
            }
        }
        IA32_EFER => {
            let mut context = vcpu.guest_context();
            let mut sregs = context.sregs();
            sregs.efer = value;
            cr::sync_efer_lma(&mut sregs);
            context.set_sregs(sregs);
        }
        IA32_BIOS_SIGN_ID => {}
        _ => {
            if !vcpu.guest_context().write_msr(index, value) {
                warn!(
                    "hypervisor: ignoring guest write to unknown MSR {:#x}",
                    index
                );
            }
        }
    }

    Ok(())
}
