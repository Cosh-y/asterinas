use crate::{
    arch::vm::{X86GprIndex, context::GuestContext},
    prelude::*,
};

pub(crate) fn emulate_cpuid(context: &mut GuestContext) -> Result<()> {
    let function = context.arch().gpr(X86GprIndex::Rax) as u32;
    let index = context.arch().gpr(X86GprIndex::Rcx) as u32;
    let entry = context.cpuid_result(function, index);

    context
        .arch_mut()
        .set_gpr(X86GprIndex::Rax, 8, entry.eax as u64);
    context
        .arch_mut()
        .set_gpr(X86GprIndex::Rbx, 8, entry.ebx as u64);
    context
        .arch_mut()
        .set_gpr(X86GprIndex::Rcx, 8, entry.ecx as u64);
    context
        .arch_mut()
        .set_gpr(X86GprIndex::Rdx, 8, entry.edx as u64);

    Ok(())
}
