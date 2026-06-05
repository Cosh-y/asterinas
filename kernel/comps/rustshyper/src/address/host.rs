use alloc::sync::Arc;

use ostd::arch::virt::*;
use ostd::mm::{HasPaddr, PAGE_SIZE, VmSpace};
use ostd::mm::vm_space::VmQueriedItem;

use crate::error::*;

pub(crate) fn query_userspace_page_hpa(vm_space: &Arc<VmSpace>, userspace_addr: VirtAddr) -> Result<PhysAddr> {
    debug_assert!(userspace_addr.is_multiple_of(PAGE_SIZE));

    loop {
        let page_range = userspace_addr..(userspace_addr + PAGE_SIZE);
        let preempt_guard = ostd::task::disable_preempt();
        let mut cursor = vm_space.cursor(&preempt_guard, &page_range).map_err(|_| {
            Error::with_message(
                Errno::Fault,
                "failed to create vm_space cursor for guest memory",
            )
        })?;

        let queried_item = cursor
            .query()
            .map_err(|_| {
                Error::with_message(
                    Errno::Fault,
                    "failed to query vm_space mapping for guest memory",
                )
            })?
            .1;

        match queried_item {
            Some(VmQueriedItem::MappedRam { frame, .. }) => return Ok(frame.paddr() as _),
            Some(VmQueriedItem::MappedIoMem { paddr, .. }) => return Ok(paddr as _),
            None => (),
        }

        drop(cursor);
        drop(preempt_guard);

        touch_userspace_page(vm_space, userspace_addr)?;
    }
}

fn touch_userspace_page(vm_space: &Arc<VmSpace>, userspace_addr: VirtAddr) -> Result<()> {
    let mut reader = vm_space.reader(userspace_addr, 1).map_err(|err| {
        let _ = err;
        Error::with_message(
            Errno::Fault,
            "failed to create userspace reader while faulting in guest memory",
        )
    })?;

    let _: u8 = reader.read_val().map_err(|err| {
        let _ = err;
        Error::with_message(
            Errno::Fault,
            "failed to fault in userspace page for guest memory",
        )
    })?;

    Ok(())
}