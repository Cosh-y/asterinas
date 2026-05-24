// SPDX-License-Identifier: MPL-2.0

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use aster_virtio::device::socket::connect::{ConnectionInfo, VsockEvent};

use super::connected::ConnectionID;
use crate::{
    events::IoEvents,
    net::socket::vsock::{VSOCK_GLOBAL, addr::VsockSocketAddr},
    prelude::*,
    process::signal::{PollHandle, Pollee},
};

pub struct Connecting {
    id: ConnectionID,
    token: u64,
    info: SpinLock<ConnectionInfo>,
    state: SpinLock<ConnectState>,
    port_moved_to_connected: AtomicBool,
    pollee: Pollee,
}

static NEXT_CONNECTING_TOKEN: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConnectState {
    Connecting,
    Connected,
    Failed(Errno),
}

impl Connecting {
    pub fn new(peer_addr: VsockSocketAddr, local_addr: VsockSocketAddr) -> Self {
        Self {
            info: SpinLock::new(ConnectionInfo::new(peer_addr.into(), local_addr.port)),
            id: ConnectionID::new(local_addr, peer_addr),
            token: NEXT_CONNECTING_TOKEN.fetch_add(1, Ordering::Relaxed),
            state: SpinLock::new(ConnectState::Connecting),
            port_moved_to_connected: AtomicBool::new(false),
            pollee: Pollee::new(),
        }
    }

    pub fn peer_addr(&self) -> VsockSocketAddr {
        self.id.peer_addr
    }

    pub fn local_addr(&self) -> VsockSocketAddr {
        self.id.local_addr
    }

    pub fn id(&self) -> ConnectionID {
        self.id
    }

    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn info(&self) -> ConnectionInfo {
        self.info.disable_irq().lock().clone()
    }

    pub fn update_info(&self, event: &VsockEvent) {
        self.info.disable_irq().lock().update_for_event(event)
    }

    pub fn poll(&self, mask: IoEvents, poller: Option<&mut PollHandle>) -> IoEvents {
        self.pollee
            .poll_with(mask, poller, || self.check_io_events())
    }

    fn check_io_events(&self) -> IoEvents {
        match *self.state.disable_irq().lock() {
            ConnectState::Connected | ConnectState::Failed(_) => IoEvents::IN,
            ConnectState::Connecting => IoEvents::empty(),
        }
    }

    pub fn set_connected(&self) {
        *self.state.disable_irq().lock() = ConnectState::Connected;
        error!(
            "vhost-vsock connecting socket marked connected local={:?} peer={:?}",
            self.local_addr(),
            self.peer_addr()
        );
        self.pollee.notify(IoEvents::IN);
    }

    pub fn set_failed(&self, errno: Errno) {
        *self.state.disable_irq().lock() = ConnectState::Failed(errno);
        error!(
            "vhost-vsock connecting socket failed local={:?} peer={:?} errno={:?}",
            self.local_addr(),
            self.peer_addr(),
            errno
        );
        self.pollee.notify(IoEvents::IN);
    }

    pub fn result(&self) -> Result<()> {
        match *self.state.disable_irq().lock() {
            ConnectState::Connected => Ok(()),
            ConnectState::Failed(errno) => Err(Error::new(errno)),
            ConnectState::Connecting => {
                return_errno_with_message!(Errno::EINPROGRESS, "vsock is still connecting")
            }
        }
    }

    pub fn move_port_to_connected(&self) {
        self.port_moved_to_connected.store(true, Ordering::Relaxed);
    }
}

impl Drop for Connecting {
    fn drop(&mut self) {
        if self.port_moved_to_connected.load(Ordering::Relaxed) {
            return;
        }
        let vsockspace = VSOCK_GLOBAL.get().unwrap();
        if vsockspace.remove_connecting_socket_for_drop(&self.local_addr(), self.token) {
            vsockspace.recycle_port(&self.local_addr().port);
        }
    }
}
