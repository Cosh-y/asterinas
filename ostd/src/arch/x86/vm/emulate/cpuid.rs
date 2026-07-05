use crate::{arch::vm::context::GuestContext, prelude::*};

pub(crate) fn emulate_cpuid(context: &mut GuestContext) -> Result<()> {
    let function = context.arch().gpr(0) as u32;
    let index = context.arch().gpr(2) as u32;
    let entry = context.cpuid_result(function, index);

    context.arch_mut().set_gpr(0, 8, entry.eax as u64);
    context.arch_mut().set_gpr(1, 8, entry.ebx as u64);
    context.arch_mut().set_gpr(2, 8, entry.ecx as u64);
    context.arch_mut().set_gpr(3, 8, entry.edx as u64);

    Ok(())
}
