//! EPT (Extended Page Tables) implementation for x86_64

use alloc::vec::Vec;

use super::types::*;
use crate::{
    error::Error,
    mm::{FrameAllocOptions, PAGE_SIZE, frame::Frame, kspace::paddr_to_vaddr, mem_obj::HasPaddr},
};

/// EPT Page Table Entry
#[derive(Debug, Clone, Copy)]
struct EptPte(u64);

impl EptPte {
    const READ: u64 = 1 << 0;
    const WRITE: u64 = 1 << 1;
    const EXECUTE: u64 = 1 << 2;
    const MEMORY_TYPE_WB: u64 = 6 << 3; // Write-back
    const IGNORE_PAT: u64 = 1 << 6;
    const LARGE_PAGE: u64 = 1 << 7;
    const ACCESSED: u64 = 1 << 8;
    const DIRTY: u64 = 1 << 9;
    const PHYS_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

    const fn new() -> Self {
        Self(0)
    }

    const fn is_present(&self) -> bool {
        (self.0 & (Self::READ | Self::WRITE | Self::EXECUTE)) != 0
    }

    const fn addr(&self) -> u64 {
        self.0 & Self::PHYS_ADDR_MASK
    }

    fn set_addr(&mut self, addr: u64) {
        self.0 = (self.0 & !Self::PHYS_ADDR_MASK) | (addr & Self::PHYS_ADDR_MASK);
    }

    fn set_flags(&mut self, flags: u64) {
        self.0 |= flags;
    }
}

/// EPT page table level shifts
const EPT_PML4_SHIFT: usize = 39;
const EPT_PDPT_SHIFT: usize = 30;
const EPT_PD_SHIFT: usize = 21;
const EPT_PT_SHIFT: usize = 12;
const EPT_PTRS_PER_TABLE: usize = 512;

/// EPT Page Table structure
///
/// Manages a 4-level EPT page table for guest physical to host physical address translation.
/// The structure owns all page table frames and automatically cleans them up on drop.
pub struct EptPageTable {
    /// Root page table (PML4) frame
    pml4_frame: Frame<()>,
    /// All allocated page table frames (PDPT, PD, PT levels)
    /// Stored to ensure proper cleanup
    page_table_frames: Vec<Frame<()>>,
}

impl EptPageTable {
    /// Creates a new EPT page table
    pub fn new() -> Result<Self, Error> {
        // Allocate PML4 frame (root level)
        let pml4_frame = FrameAllocOptions::new()
            .zeroed(true)
            .alloc_frame()
            .map_err(|_| Error::NoMemory)?;

        Ok(Self {
            pml4_frame,
            page_table_frames: Vec::new(),
        })
    }

    /// Returns the EPTP value for VMCS
    ///
    /// Format: bits 11:0 = EPT config, bits 51:12 = PML4 physical address
    pub fn eptp(&self) -> u64 {
        let mut eptp = self.pml4_frame.paddr() as u64;

        // EPT Memory Type: Write-back (6)
        eptp |= 6;

        // EPT Page-walk length: 4 levels (encoded as 3)
        eptp |= 3 << 3;

        // Enable accessed and dirty flags
        eptp |= 1 << 6;

        eptp
    }

    /// Maps a guest physical address range to host physical address
    ///
    /// Maps `size` bytes starting from `gpa` to `hpa` using 4KB pages.
    pub fn map_range(
        &mut self,
        gpa: GuestPhysAddr,
        hpa: PhysAddr,
        size: MemSize,
    ) -> Result<(), Error> {
        let mut offset: u64 = 0;

        while offset < size as u64 {
            let gpa_aligned = gpa + offset;
            let hpa_aligned = hpa + offset;

            self.map_4kb_page(gpa_aligned, hpa_aligned)?;
            offset += PAGE_SIZE as u64;
        }

        Ok(())
    }

    /// Maps a single 4KB page
    ///
    /// Performs 4-level page table walk: PML4 -> PDPT -> PD -> PT
    /// EPT page table entries contain host physical addresses, using guest physical address as index
    fn map_4kb_page(&mut self, gpa: u64, hpa: u64) -> Result<(), Error> {
        // Extract page table indices from guest physical address
        let pml4_idx = ((gpa >> EPT_PML4_SHIFT) & 0x1FF) as usize;
        let pdpt_idx = ((gpa >> EPT_PDPT_SHIFT) & 0x1FF) as usize;
        let pd_idx = ((gpa >> EPT_PD_SHIFT) & 0x1FF) as usize;
        let pt_idx = ((gpa >> EPT_PT_SHIFT) & 0x1FF) as usize;

        // Get PML4 table
        let pml4_virt = paddr_to_vaddr(self.pml4_frame.paddr());
        let pml4 = unsafe {
            core::slice::from_raw_parts_mut(pml4_virt as *mut EptPte, EPT_PTRS_PER_TABLE)
        };

        // Get or create PDPT
        if !pml4[pml4_idx].is_present() {
            let pdpt_frame = FrameAllocOptions::new()
                .zeroed(true)
                .alloc_frame()
                .map_err(|_| Error::NoMemory)?;

            let pdpt_phys = pdpt_frame.paddr() as u64;

            // Install PDPT in PML4
            let mut pte = EptPte::new();
            pte.set_addr(pdpt_phys);
            pte.set_flags(EptPte::READ | EptPte::WRITE | EptPte::EXECUTE);
            pml4[pml4_idx] = pte;

            self.page_table_frames.push(pdpt_frame);
        }

        // Get PDPT table
        let pdpt_phys = pml4[pml4_idx].addr();
        let pdpt_virt = paddr_to_vaddr(pdpt_phys as usize);
        let pdpt = unsafe {
            core::slice::from_raw_parts_mut(pdpt_virt as *mut EptPte, EPT_PTRS_PER_TABLE)
        };

        // Get or create PD
        if !pdpt[pdpt_idx].is_present() {
            let pd_frame = FrameAllocOptions::new()
                .zeroed(true)
                .alloc_frame()
                .map_err(|_| Error::NoMemory)?;

            let pd_phys = pd_frame.paddr() as u64;

            // Install PD in PDPT
            let mut pte = EptPte::new();
            pte.set_addr(pd_phys);
            pte.set_flags(EptPte::READ | EptPte::WRITE | EptPte::EXECUTE);
            pdpt[pdpt_idx] = pte;

            self.page_table_frames.push(pd_frame);
        }

        // Get PD table
        let pd_phys = pdpt[pdpt_idx].addr();
        let pd_virt = paddr_to_vaddr(pd_phys as usize);
        let pd =
            unsafe { core::slice::from_raw_parts_mut(pd_virt as *mut EptPte, EPT_PTRS_PER_TABLE) };

        // Get or create PT
        if !pd[pd_idx].is_present() {
            let pt_frame = FrameAllocOptions::new()
                .zeroed(true)
                .alloc_frame()
                .map_err(|_| Error::NoMemory)?;

            let pt_phys = pt_frame.paddr() as u64;

            // Install PT in PD
            let mut pte = EptPte::new();
            pte.set_addr(pt_phys);
            pte.set_flags(EptPte::READ | EptPte::WRITE | EptPte::EXECUTE);
            pd[pd_idx] = pte;

            self.page_table_frames.push(pt_frame);
        }

        // Get PT table
        let pt_phys = pd[pd_idx].addr();
        let pt_virt = paddr_to_vaddr(pt_phys as usize);
        let pt =
            unsafe { core::slice::from_raw_parts_mut(pt_virt as *mut EptPte, EPT_PTRS_PER_TABLE) };

        // Install 4KB page in PT
        let mut pte = EptPte::new();
        pte.set_addr(hpa & !0xFFF); // Align to 4KB
        pte.set_flags(EptPte::READ | EptPte::WRITE | EptPte::EXECUTE);
        pte.set_flags(EptPte::MEMORY_TYPE_WB); // Write-back memory type
        pte.set_flags(EptPte::IGNORE_PAT);
        pt[pt_idx] = pte;

        Ok(())
    }

    /// Translates guest physical address to host physical address
    ///
    /// Returns the host physical address corresponding to the given guest physical address,
    /// or an error if the mapping does not exist.
    pub fn translate(&self, gpa: u64) -> Result<u64, Error> {
        let pml4_idx = ((gpa >> EPT_PML4_SHIFT) & 0x1FF) as usize;
        let pdpt_idx = ((gpa >> EPT_PDPT_SHIFT) & 0x1FF) as usize;
        let pd_idx = ((gpa >> EPT_PD_SHIFT) & 0x1FF) as usize;
        let pt_idx = ((gpa >> EPT_PT_SHIFT) & 0x1FF) as usize;
        let offset = gpa & 0xFFF; // 4KB offset

        // Walk page table
        let pml4_virt = paddr_to_vaddr(self.pml4_frame.paddr());
        let pml4 =
            unsafe { core::slice::from_raw_parts(pml4_virt as *const EptPte, EPT_PTRS_PER_TABLE) };

        if !pml4[pml4_idx].is_present() {
            return Err(Error::InvalidArgs);
        }

        let pdpt_phys = pml4[pml4_idx].addr();
        let pdpt_virt = paddr_to_vaddr(pdpt_phys as usize);
        let pdpt =
            unsafe { core::slice::from_raw_parts(pdpt_virt as *const EptPte, EPT_PTRS_PER_TABLE) };

        if !pdpt[pdpt_idx].is_present() {
            return Err(Error::InvalidArgs);
        }

        let pd_phys = pdpt[pdpt_idx].addr();
        let pd_virt = paddr_to_vaddr(pd_phys as usize);
        let pd =
            unsafe { core::slice::from_raw_parts(pd_virt as *const EptPte, EPT_PTRS_PER_TABLE) };

        if !pd[pd_idx].is_present() {
            return Err(Error::InvalidArgs);
        }

        let pt_phys = pd[pd_idx].addr();
        let pt_virt = paddr_to_vaddr(pt_phys as usize);
        let pt =
            unsafe { core::slice::from_raw_parts(pt_virt as *const EptPte, EPT_PTRS_PER_TABLE) };

        if !pt[pt_idx].is_present() {
            return Err(Error::InvalidArgs);
        }

        let page_phys = pt[pt_idx].addr();
        Ok(page_phys + offset)
    }
}
