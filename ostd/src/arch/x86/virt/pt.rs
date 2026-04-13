//! build a tmp page table to start up the guest VM
#![allow(missing_docs)]
use super::types::*;
use crate::{
    error::Error,
    mm::{FrameAllocOptions, frame::Frame, kspace::paddr_to_vaddr, mem_obj::HasPaddr},
};

#[repr(u64)]
#[derive(Debug, Clone, Copy)]
pub enum PageTableFlags {
    Present = 1 << 0,
    Writable = 1 << 1,
    UserAccessible = 1 << 2,
    WriteThrough = 1 << 3,
    CacheDisable = 1 << 4,
    Accessed = 1 << 5,
    Dirty = 1 << 6,
    HugePage = 1 << 7,
    Global = 1 << 8,
    NoExecute = 1 << 63,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct PageTableEntry(u64);

impl PageTableEntry {
    pub fn new(phys_addr: GuestPhysAddr, flags: u64) -> Self {
        Self((phys_addr & 0x000fffff_fffff000) | (flags & 0xfff0000000000fff))
    }

    pub fn addr(&self) -> GuestPhysAddr {
        self.0 & 0x000fffff_fffff000
    }

    pub fn flags(&self) -> u64 {
        self.0 & 0xfff0000000000fff
    }

    pub fn is_present(&self) -> bool {
        (self.flags() & PageTableFlags::Present as u64) != 0
    }

    pub fn is_huge_page(&self) -> bool {
        (self.flags() & PageTableFlags::HugePage as u64) != 0
    }
}

pub static mut PML4_GPA: GuestPhysAddr = 0;
const PTES_PER_TABLE: usize = 512;

pub struct TmpPageTables {
    pub pml4: Frame<()>,
    // only use one page of pdp
    pub pdpt: Frame<()>,
}

impl TmpPageTables {
    pub fn new() -> Result<Self, Error> {
        let pml4 = FrameAllocOptions::new()
            .zeroed(true)
            .alloc_frame()
            .map_err(|_| Error::NoMemory)?;
        let pdpt = FrameAllocOptions::new()
            .zeroed(true)
            .alloc_frame()
            .map_err(|_| Error::NoMemory)?;

        Ok(Self { pml4, pdpt })
    }

    pub fn pml4_base_addr(&self) -> VirtAddr {
        paddr_to_vaddr(self.pml4.paddr()) as VirtAddr
    }

    pub fn pdpt_base_addr(&self) -> VirtAddr {
        paddr_to_vaddr(self.pdpt.paddr()) as VirtAddr
    }

    /// Map the first 4GB of memory using 1GB pages
    /// @ load_addr: gpa where root page table loads
    /// @ base_addr: gpa(identity to pa) base of 4GB memory which will be mapped
    pub fn setup_identity_map_4GB(
        &mut self,
        load_addr: GuestPhysAddr,
        base_addr: GuestPhysAddr,
    ) -> Result<(), Error> {
        if base_addr % 0x4000_0000 != 0 {
            // base_addr 1GB aligned
            return Err(Error::InvalidArgs);
        }
        if load_addr % 0x1000 != 0 {
            // load_addr 4KB aligned
            return Err(Error::InvalidArgs);
        }
        let pdpt_virt = paddr_to_vaddr(self.pdpt.paddr());
        let pdpt_mut_ref: &mut [PageTableEntry] = unsafe {
            core::slice::from_raw_parts_mut(pdpt_virt as *mut PageTableEntry, PTES_PER_TABLE)
        };
        for i in 0..4 {
            let phys_addr = ((i as GuestPhysAddr) << 30) + base_addr;
            let index = (phys_addr >> 30) as usize; // index represents guest virt addr
            let entry = PageTableEntry::new(
                phys_addr,
                (PageTableFlags::Present as u64)
                    | (PageTableFlags::Writable as u64)
                    | (PageTableFlags::HugePage as u64),
            );
            if index < PTES_PER_TABLE {
                pdpt_mut_ref[index] = entry;
            } else {
                return Err(Error::Overflow);
            }
        }

        let pml4_virt = paddr_to_vaddr(self.pml4.paddr());
        let pml4_mut_ref: &mut [PageTableEntry] = unsafe {
            core::slice::from_raw_parts_mut(pml4_virt as *mut PageTableEntry, PTES_PER_TABLE)
        };
        // Set up PML4 entry to point to PDPT, PDPT will be load to GuestPhysAddr 0x1000
        let pml4_entry = PageTableEntry::new(
            load_addr + 0x1000,
            (PageTableFlags::Present as u64) | (PageTableFlags::Writable as u64),
        );
        pml4_mut_ref[0] = pml4_entry;
        Ok(())
    }
}
