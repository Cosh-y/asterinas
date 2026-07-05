use crate::{
    arch::vm::{X86GprIndex, context::GuestContext, vmx::VmcsReadOnlyNW},
    prelude::*,
};

pub(crate) fn emulate_cr_access(context: &mut GuestContext) -> Result<()> {
    let qualification = VmcsReadOnlyNW::EXIT_QUALIFICATION.read()?;
    let cr_index = (qualification & 0xF) as u8;
    let access = ((qualification >> 4) & 0b11) as u8;
    let gpr_encoding = ((qualification >> 8) & 0xF) as u8;
    let gpr = X86GprIndex::from_x86_reg_encoding(gpr_encoding)?;

    match access {
        // write
        0 => {
            let value = context.arch().gpr(gpr);
            match cr_index {
                // A value different from the shadow was written to the masked bit.
                0 => context.arch_mut().write_cr0(value),
                2 => context.arch_mut().set_cr2(value),
                3 => context.arch_mut().set_cr3(value),
                4 => context.arch_mut().write_cr4(value),
                other => warn!("hypervisor: ignoring guest write to CR{}", other),
            }
        }
        // read
        1 => {
            let value = match cr_index {
                0 => context.arch().cr0(),
                2 => context.arch().cr2(),
                3 => context.arch().cr3(),
                4 => context.arch().cr4(),
                other => {
                    warn!("hypervisor: ignoring guest read from CR{}", other);
                    0
                }
            };
            context.arch_mut().set_gpr(gpr, 8, value);
        }
        other => {
            warn!("hypervisor: unsupported CR access type {}", other);
        }
    }

    Ok(())
}
