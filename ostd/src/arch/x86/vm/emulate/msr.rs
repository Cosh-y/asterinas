use x86::msr::IA32_TSC_DEADLINE;

use crate::{
    arch::vm::{X86GprIndex, context::GuestContext},
    prelude::*,
};

const MSR_KVM_WALL_CLOCK: u32 = 0x11;
const MSR_KVM_SYSTEM_TIME: u32 = 0x12;
const MSR_KVM_WALL_CLOCK_NEW: u32 = 0x4b56_4d00;
const MSR_KVM_SYSTEM_TIME_NEW: u32 = 0x4b56_4d01;

pub(crate) fn needs_kernel_msr_handler(context: &GuestContext) -> bool {
    needs_kernel_msr_policy(context.arch().gpr(X86GprIndex::Rcx) as u32)
}

pub(crate) fn emulate_msrrw(context: &mut GuestContext, is_write: bool) -> Result<()> {
    let msr_index = context.arch().gpr(X86GprIndex::Rcx) as u32;

    if is_write {
        let msr_value = (context.arch().gpr(X86GprIndex::Rax) as u32 as u64)
            | ((context.arch().gpr(X86GprIndex::Rdx) as u32 as u64) << 32);

        if !context.write_msr(msr_index, msr_value) {
            error!("set_msr: msr {:x} not impl.", msr_index);
        }

        return Ok(());
    }

    // is read
    let msr_value = context.read_msr(msr_index).unwrap_or_else(|| {
        error!("get unknown msr {:x}, return 0.", msr_index);
        0
    });

    context
        .arch_mut()
        .set_gpr(X86GprIndex::Rax, 8, msr_value as u32 as u64);
    context.arch_mut().set_gpr(X86GprIndex::Rdx, 8, msr_value >> 32);
    Ok(())
}

fn is_kvmclock_msr(index: u32) -> bool {
    matches!(
        index,
        MSR_KVM_WALL_CLOCK | MSR_KVM_SYSTEM_TIME | MSR_KVM_WALL_CLOCK_NEW | MSR_KVM_SYSTEM_TIME_NEW
    )
}

fn needs_kernel_msr_policy(index: u32) -> bool {
    is_kvmclock_msr(index) || index == IA32_TSC_DEADLINE
}
