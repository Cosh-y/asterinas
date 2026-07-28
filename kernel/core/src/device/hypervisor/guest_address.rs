// SPDX-License-Identifier: MPL-2.0

//! Translates guest virtual addresses to guest physical addresses.

use ostd::{arch::vm::GuestContext, mm::Gpaddr};

use super::vm_memory::VmMemory;
use crate::prelude::*;

pub(super) fn translate_gva_to_gpa(
    context: &GuestContext,
    vm_memory: &VmMemory,
    gva: usize,
) -> Result<Gpaddr> {
    const CR0_PAGING: u64 = 1 << 31;
    const CR4_PAGE_SIZE_EXTENSIONS: u64 = 1 << 4;
    const CR4_PHYSICAL_ADDRESS_EXTENSION: u64 = 1 << 5;
    const CR4_5_LEVEL_PAGING: u64 = 1 << 12;
    const EFER_LONG_MODE_ACTIVE: u64 = 1 << 10;

    let sregs = context.sregs();
    if sregs.cr0 & CR0_PAGING == 0 {
        return Ok(gva);
    }

    if sregs.efer & EFER_LONG_MODE_ACTIVE != 0 {
        let five_level = sregs.cr4 & CR4_5_LEVEL_PAGING != 0;
        return translate_long_mode_gva(vm_memory, gva, sregs.cr3 as Gpaddr, five_level);
    }

    let gva = gva & u32::MAX as usize;
    if sregs.cr4 & CR4_PHYSICAL_ADDRESS_EXTENSION != 0 {
        translate_pae_gva(vm_memory, gva, sregs.cr3 as Gpaddr)
    } else {
        translate_legacy_gva(
            vm_memory,
            gva,
            sregs.cr3 as Gpaddr,
            sregs.cr4 & CR4_PAGE_SIZE_EXTENSIONS != 0,
        )
    }
}

fn translate_long_mode_gva(
    vm_memory: &VmMemory,
    gva: usize,
    cr3: Gpaddr,
    five_level: bool,
) -> Result<Gpaddr> {
    const PTE_PRESENT: u64 = 1 << 0;
    const PTE_HUGE: u64 = 1 << 7;
    const PTE_ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

    let address_width = if five_level { 57 } else { 48 };
    if !is_canonical(gva as u64, address_width) {
        return Err(Error::new(Errno::EFAULT));
    }

    let shifts: &[usize] = if five_level {
        &[48, 39, 30, 21, 12]
    } else {
        &[39, 30, 21, 12]
    };
    let mut table = cr3 & !0xfff;
    for &shift in shifts {
        let entry_gpa = table
            .checked_add(((gva >> shift) & 0x1ff) * size_of::<u64>())
            .ok_or_else(|| Error::new(Errno::EFAULT))?;
        let entry: u64 = vm_memory.read_val(entry_gpa)?;
        if entry & PTE_PRESENT == 0 {
            return Err(Error::new(Errno::EFAULT));
        }

        let huge = entry & PTE_HUGE != 0 && matches!(shift, 21 | 30);
        if shift == 12 || huge {
            let offset_mask = (1_usize << shift) - 1;
            let page_base = (entry & PTE_ADDR_MASK & !(offset_mask as u64)) as Gpaddr;
            return Ok(page_base | (gva & offset_mask));
        }
        table = (entry & PTE_ADDR_MASK) as Gpaddr;
    }
    Err(Error::new(Errno::EFAULT))
}

fn translate_pae_gva(vm_memory: &VmMemory, gva: usize, cr3: Gpaddr) -> Result<Gpaddr> {
    const PTE_PRESENT: u64 = 1 << 0;
    const PTE_HUGE: u64 = 1 << 7;
    const PTE_ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

    let pdpt = cr3 & !0x1f;
    let pdpte: u64 = vm_memory.read_val(pdpt + ((gva >> 30) & 0x3) * size_of::<u64>())?;
    if pdpte & PTE_PRESENT == 0 {
        return Err(Error::new(Errno::EFAULT));
    }

    let pd = (pdpte & PTE_ADDR_MASK) as Gpaddr;
    let pde: u64 = vm_memory.read_val(pd + ((gva >> 21) & 0x1ff) * size_of::<u64>())?;
    if pde & PTE_PRESENT == 0 {
        return Err(Error::new(Errno::EFAULT));
    }
    if pde & PTE_HUGE != 0 {
        const PAGE_2M_MASK: usize = (1 << 21) - 1;
        let page_base = (pde & PTE_ADDR_MASK & !(PAGE_2M_MASK as u64)) as Gpaddr;
        return Ok(page_base | (gva & PAGE_2M_MASK));
    }

    let pt = (pde & PTE_ADDR_MASK) as Gpaddr;
    let pte: u64 = vm_memory.read_val(pt + ((gva >> 12) & 0x1ff) * size_of::<u64>())?;
    if pte & PTE_PRESENT == 0 {
        return Err(Error::new(Errno::EFAULT));
    }
    Ok((pte & PTE_ADDR_MASK) as Gpaddr | (gva & (PAGE_SIZE - 1)))
}

fn translate_legacy_gva(
    vm_memory: &VmMemory,
    gva: usize,
    cr3: Gpaddr,
    page_size_extensions: bool,
) -> Result<Gpaddr> {
    const PTE_PRESENT: u32 = 1 << 0;
    const PTE_HUGE: u32 = 1 << 7;

    let page_directory = cr3 & !0xfff;
    let pde: u32 = vm_memory.read_val(page_directory + ((gva >> 22) & 0x3ff) * size_of::<u32>())?;
    if pde & PTE_PRESENT == 0 {
        return Err(Error::new(Errno::EFAULT));
    }
    if page_size_extensions && pde & PTE_HUGE != 0 {
        const PAGE_4M_MASK: usize = (1 << 22) - 1;
        // Bits 20:13 carry physical bits 39:32 when PSE-36 is in use.
        let page_base =
            usize::try_from(pde & 0xffc0_0000)? | (usize::try_from(pde & 0x001f_e000)? << 19);
        return Ok(page_base | (gva & PAGE_4M_MASK));
    }

    let page_table = usize::try_from(pde & 0xffff_f000)?;
    let pte: u32 = vm_memory.read_val(page_table + ((gva >> 12) & 0x3ff) * size_of::<u32>())?;
    if pte & PTE_PRESENT == 0 {
        return Err(Error::new(Errno::EFAULT));
    }
    Ok(usize::try_from(pte & 0xffff_f000)? | (gva & (PAGE_SIZE - 1)))
}

fn is_canonical(address: u64, width: u32) -> bool {
    let shift = 64 - width;
    ((address << shift) as i64 >> shift) as u64 == address
}

#[cfg(ktest)]
mod tests {
    use ostd::prelude::ktest;

    use super::*;

    #[ktest]
    fn canonical_address_widths_are_validated() {
        assert!(is_canonical(0x0000_7fff_ffff_ffff, 48));
        assert!(is_canonical(0xffff_8000_0000_0000, 48));
        assert!(!is_canonical(0x0000_8000_0000_0000, 48));
        assert!(is_canonical(0x00ff_ffff_ffff_ffff, 57));
        assert!(!is_canonical(0x0100_0000_0000_0000, 57));
    }
}
