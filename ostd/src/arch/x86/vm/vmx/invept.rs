// SPDX-License-Identifier: MPL-2.0

use core::sync::atomic::{AtomicUsize, Ordering};

use super::msr::Msr;
use crate::{error, error::Error, prelude::*, sync::SpinLock};

const INVEPT_ALL_CONTEXTS: u64 = 2;
const EPT_VPID_CAP_INVEPT: u64 = 1 << 20;
const EPT_VPID_CAP_INVEPT_ALL_CONTEXTS: u64 = 1 << 26;

pub(crate) fn check_ept_support() -> Result<()> {
    // Check CPUID for VMX support.
    let cpuid_result = core::arch::x86_64::__cpuid(1);
    if (cpuid_result.ecx & (1 << 5)) == 0 {
        error!("VMX not supported by CPU");
        return Err(Error::NotEnoughResources);
    }

    // Check if EPT is supported by reading the VM-execution control VMCS field.
    Ok(())
}

static EPT_FLUSH_LOCK: SpinLock<()> = SpinLock::new(());
static EPT_FLUSH_ACKS: AtomicUsize = AtomicUsize::new(0);
static EPT_FLUSH_ERRORS: AtomicUsize = AtomicUsize::new(0);

#[repr(C, align(16))]
struct InveptDescriptor {
    eptp: u64,
    reserved: u64,
}

/// Invalidates EPT-derived translations on every CPU.
pub(crate) fn flush_ept_all_contexts_sync() -> Result<()> {
    let cap = Msr::IA32_VMX_EPT_VPID_CAP.read();
    if cap & EPT_VPID_CAP_INVEPT == 0 || cap & EPT_VPID_CAP_INVEPT_ALL_CONTEXTS == 0 {
        return Err(Error::NotEnoughResources);
    }

    let _flush_guard = EPT_FLUSH_LOCK.lock();
    EPT_FLUSH_ACKS.store(0, Ordering::Release);
    EPT_FLUSH_ERRORS.store(0, Ordering::Release);

    let targets = crate::cpu::CpuSet::new_full();
    let cpu_count = crate::cpu::num_cpus();
    crate::smp::inter_processor_call(&targets, flush_ept_all_contexts_on_cpu);

    while EPT_FLUSH_ACKS.load(Ordering::Acquire) < cpu_count {
        core::hint::spin_loop();
    }

    if EPT_FLUSH_ERRORS.load(Ordering::Acquire) != 0 {
        return Err(Error::InvalidArgs);
    }
    Ok(())
}

fn flush_ept_all_contexts_on_cpu() {
    if invept_all_contexts().is_err() {
        EPT_FLUSH_ERRORS.fetch_add(1, Ordering::AcqRel);
    }
    EPT_FLUSH_ACKS.fetch_add(1, Ordering::AcqRel);
}

/// Invalidates all EPT contexts on the current CPU.
fn invept_all_contexts() -> Result<()> {
    let descriptor = InveptDescriptor {
        eptp: 0,
        reserved: 0,
    };
    let mut failed: u8;
    let mut invalid: u8;
    unsafe {
        core::arch::asm!(
            "invept {typ}, [{desc}]",
            "setb {failed}",
            "setz {invalid}",
            typ = in(reg) INVEPT_ALL_CONTEXTS,
            desc = in(reg) &descriptor,
            failed = lateout(reg_byte) failed,
            invalid = lateout(reg_byte) invalid,
            options(nostack, readonly)
        );
    }

    if failed != 0 {
        return Err(Error::InvalidArgs);
    }
    if invalid != 0 {
        return Err(Error::InvalidArgs);
    }

    Ok(())
}
