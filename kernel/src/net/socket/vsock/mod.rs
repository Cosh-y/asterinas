// SPDX-License-Identifier: MPL-2.0

use alloc::sync::Arc;

use aster_virtio::device::socket::{DEVICE_NAME, get_device, register_recv_callback};
use common::VsockSpace;
use spin::Once;

use crate::prelude::*;

pub mod addr;
pub mod common;
pub mod stream;
pub use addr::VsockSocketAddr;
pub use stream::VsockStreamSocket;

// init static driver
pub static VSOCK_GLOBAL: Once<Arc<VsockSpace>> = Once::new();

pub(in crate::net) fn init() {
    if let Some(driver) = get_device(DEVICE_NAME) {
        VSOCK_GLOBAL.call_once(|| Arc::new(VsockSpace::new(driver)));
        register_recv_callback(DEVICE_NAME, || {
            let vsockspace = VSOCK_GLOBAL.get().unwrap();
            vsockspace.poll().unwrap();
        })
    } else {
        VSOCK_GLOBAL.call_once(|| Arc::new(VsockSpace::new_host_vhost()));
    }
}

pub(crate) fn handle_vhost_event(
    event: aster_virtio::device::socket::connect::VsockEvent,
    body: &[u8],
) -> Result<()> {
    let vsockspace = VSOCK_GLOBAL
        .get()
        .ok_or_else(|| Error::with_message(Errno::ENODEV, "vsock space is not initialized"))?;
    vsockspace.handle_event(event, body)
}
