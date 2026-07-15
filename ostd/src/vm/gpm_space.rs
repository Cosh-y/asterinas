//! Guest physical memory space.

use core::ops::Range;

use crate::{
    arch::vm::{
        ept::{EptItem, EptPtConfig},
        vmx::flush_ept_all_contexts_sync,
    },
    mm::{
        PageProperty, UFrame,
        page_table::{self, PageTable, PageTableFrag},
    },
    prelude::*,
    task::atomic_mode::AsAtomicModeGuard,
};

/// Manages the guest physical memory space of a VM.
///
/// This type owns the page table that maps guest physical addresses to
/// host physical frames. One `GuestPhysMemSpace` can be reused by multiple
/// vCPUs in the same VM by passing a reference to
/// [`super::GuestMode::execute`].
pub struct GuestPhysMemSpace {
    pt: PageTable<EptPtConfig>,
}

impl GuestPhysMemSpace {
    /// Creates a new guest physical memory space.
    ///
    /// # Errors
    /// Returns an error if the CPU does not support second-stage address
    /// translation.
    pub fn new() -> Result<Self> {
        use crate::arch::vm::vmx::check_ept_support;
        check_ept_support()?;
        Ok(Self {
            pt: PageTable::<EptPtConfig>::empty(),
        })
    }

    /// Gets an immutable cursor over a guest physical address range.
    ///
    /// The cursor behaves like a lock guard, exclusively owning a sub-tree of
    /// the page table, preventing others from creating a cursor in it. So be
    /// sure to drop the cursor as soon as possible.
    ///
    /// The creation of the cursor may block if another cursor having an
    /// overlapping range is alive.
    pub fn cursor<'a, G: AsAtomicModeGuard>(
        &'a self,
        guard: &'a G,
        gpa: &Range<Gpaddr>,
    ) -> Result<Cursor<'a>> {
        Ok(Cursor(self.pt.cursor(guard, gpa)?))
    }

    /// Gets a mutable cursor over a guest physical address range.
    ///
    /// The same as [`Self::cursor`], the cursor behaves like a lock guard,
    /// exclusively owning a sub-tree of the page table, preventing others
    /// from creating a cursor in it. So be sure to drop the cursor as soon as
    /// possible.
    ///
    /// The creation of the cursor may block if another cursor having an
    /// overlapping range is alive. The modification to the mapping by the
    /// cursor may also block or be overridden the mapping of another cursor.
    pub fn cursor_mut<'a, G: AsAtomicModeGuard>(
        &'a self,
        guard: &'a G,
        gpa: &Range<Gpaddr>,
    ) -> Result<CursorMut<'a>> {
        Ok(CursorMut {
            pt_cursor: self.pt.cursor_mut(guard, gpa)?,
        })
    }

    /// Returns the EPT pointer value for this guest memory space.
    ///
    /// The value is used by [`super::GuestMode`] so VM entry can use this EPT
    /// as the guest physical address space.
    pub(super) fn eptp(&self) -> u64 {
        const EPT_MEM_TYPE_WB: u64 = 6;
        const EPT_PAGE_WALK_LENGTH_4_LEVELS: u64 = 3 << 3;

        self.pt.root_paddr() as u64 | EPT_MEM_TYPE_WB | EPT_PAGE_WALK_LENGTH_4_LEVELS
    }
}

// impl Default for GuestPhysMemSpace {
//     fn default() -> Self {
//         Self::new()
//     }
// }

impl Drop for GuestPhysMemSpace {
    fn drop(&mut self) {
        error!("hypervisor: release guest memory space.");
        if let Err(err) = flush_ept_all_contexts_sync() {
            error!(
                "hypervisor: failed to flush EPT translations while dropping guest memory: {:?}",
                err
            );
        }
    }
}

fn flush_and_drop(frags: Vec<PageTableFrag<EptPtConfig>>) {
    if frags.is_empty() {
        return;
    }

    if let Err(err) = flush_ept_all_contexts_sync() {
        // The EPT entries have already been invalidated. Leaking the fragments
        // is safer than freeing frames that may still be cached by hardware.
        core::mem::forget(frags);
        panic!("failed to invalidate EPT translations: {:?}", err);
    }

    drop(frags);
}

/// A queried mapping item.
///
/// The address is the host physical address backing the current guest physical
/// range, together with the page properties used for that mapping.
pub type QueriedItem = (Paddr, PageProperty);

/// The cursor for querying over the guest physical memory space without modifying it.
///
/// It exclusively owns a sub-tree of the page table, preventing others from
/// reading or modifying the same sub-tree. Two read-only cursors can not be
/// created from the same guest physical address range either.
pub struct Cursor<'a>(page_table::Cursor<'a, EptPtConfig>);

impl Cursor<'_> {
    /// Queries the mapping at the current guest physical address.
    ///
    /// If the cursor is pointing to a valid guest physical address that is
    /// locked, it will return the guest physical address range and the mapped
    /// host physical address item.
    pub fn query(&mut self) -> Result<(Range<Gpaddr>, Option<QueriedItem>)> {
        let (range, item) = self.0.query()?;
        Ok((range, item.map(|(frame, prop)| (frame.paddr(), prop))))
    }

    /// Moves the cursor forward to the next mapped guest physical address.
    ///
    /// If there is a mapped guest physical address following the current
    /// address within next `len` bytes, it will return that mapped address. In
    /// this case, the cursor will stop at the mapped address.
    ///
    /// Otherwise, it will return `None`. And the cursor may stop at any
    /// address after `len` bytes.
    ///
    /// # Panics
    ///
    /// Panics if the length is longer than the remaining range of the cursor.
    pub fn find_next(&mut self, len: usize) -> Option<Gpaddr> {
        self.0.find_next(len)
    }

    /// Jumps to the guest physical address.
    pub fn jump(&mut self, gpa: Gpaddr) -> Result<()> {
        self.0.jump(gpa)?;
        Ok(())
    }

    /// Gets the guest physical address of the current slot.
    pub fn guest_physical_addr(&self) -> Gpaddr {
        self.0.virt_addr()
    }
}

/// The cursor for modifying the mappings in guest physical memory space.
///
/// It exclusively owns a sub-tree of the page table, preventing others from
/// reading or modifying the same sub-tree.
pub struct CursorMut<'a> {
    pt_cursor: page_table::CursorMut<'a, EptPtConfig>,
}

impl<'a> CursorMut<'a> {
    /// Queries the mapping at the current guest physical address.
    ///
    /// This is the same as [`Cursor::query`].
    ///
    /// If the cursor is pointing to a valid guest physical address that is
    /// locked, it will return the guest physical address range and the mapped
    /// host physical address item.
    pub fn query(&mut self) -> Result<(Range<Gpaddr>, Option<QueriedItem>)> {
        let (range, item) = self.pt_cursor.query()?;
        Ok((range, item.map(|(frame, prop)| (frame.paddr(), prop))))
    }

    /// Moves the cursor forward to the next mapped guest physical address.
    ///
    /// This is the same as [`Cursor::find_next`].
    pub fn find_next(&mut self, len: usize) -> Option<Gpaddr> {
        self.pt_cursor.find_next(len)
    }

    /// Jumps to the guest physical address.
    ///
    /// This is the same as [`Cursor::jump`].
    pub fn jump(&mut self, gpa: Gpaddr) -> Result<()> {
        self.pt_cursor.jump(gpa)?;
        Ok(())
    }

    /// Gets the guest physical address of the current slot.
    pub fn guest_physical_addr(&self) -> Gpaddr {
        self.pt_cursor.virt_addr()
    }

    /// Maps a frame into the current slot.
    ///
    /// This method will bring the cursor to the next slot after the modification.
    ///
    /// # Panics
    ///
    /// Panics if the current guest physical address is already mapped.
    pub fn map(&mut self, frame: UFrame, prop: PageProperty) {
        let item: EptItem = (frame, prop);

        // SAFETY: It is safe to map untyped memory into guest physical memory.
        unsafe { self.pt_cursor.map(item) };
    }

    /// Unmaps mappings from the current guest physical address.
    ///
    /// The method removes mapped pages or page-table subtrees up to `len`
    /// bytes from the current guest physical address, flushes TLB,
    /// and returns the number of unmapped frames or page-table frames.
    ///
    /// # Panics
    ///
    /// Panics if `len` is longer than the remaining range of the cursor or is
    /// not page-aligned.
    pub fn unmap(&mut self, len: usize) -> usize {
        let end_gpa = self.guest_physical_addr() + len;
        let mut num_unmapped: usize = 0;
        let mut frags = Vec::new();
        loop {
            // SAFETY: It is safe to un-map memory in the guest physical memory space.
            // And the un-mapped items are dropped after TLB flushes.
            let Some(frag) = (unsafe {
                self.pt_cursor
                    .take_next(end_gpa - self.guest_physical_addr())
            }) else {
                break; // No more mappings in the range.
            };

            num_unmapped += match &frag {
                PageTableFrag::Mapped { .. } => 1,
                PageTableFrag::StrayPageTable { num_frames, .. } => *num_frames,
            };

            frags.push(frag);
        }

        flush_and_drop(frags);
        num_unmapped
    }
}
