// SPDX-License-Identifier: MPL-2.0

use alloc::collections::BTreeSet;

use aster_virtio::device::socket::{
    connect::{ConnectionInfo, VsockEvent, VsockEventType},
    device::SocketDevice,
    error::SocketError,
};
use ostd::sync::LocalIrqDisabled;

use super::{
    addr::VsockSocketAddr,
    stream::{
        connected::{Connected, ConnectionID},
        connecting::Connecting,
        listen::Listen,
    },
};
use crate::{prelude::*, return_errno_with_message, util::MultiRead};

/// Manage all active sockets
pub struct VsockSpace {
    driver: Option<Arc<SpinLock<SocketDevice>>>,
    // (key, value) = (local_addr, connecting)
    connecting_sockets: SpinLock<BTreeMap<VsockSocketAddr, Arc<Connecting>>>,
    // (key, value) = (local_addr, listen)
    listen_sockets: SpinLock<BTreeMap<VsockSocketAddr, Arc<Listen>>>,
    // (key, value) = (id(local_addr,peer_addr), connected)
    connected_sockets: RwLock<BTreeMap<ConnectionID, Arc<Connected>>, LocalIrqDisabled>,
    // Used ports
    used_ports: SpinLock<BTreeSet<u32>>,
}

impl VsockSpace {
    /// Create a new global VsockSpace
    pub fn new(driver: Arc<SpinLock<SocketDevice>>) -> Self {
        Self {
            driver: Some(driver),
            connecting_sockets: SpinLock::new(BTreeMap::new()),
            listen_sockets: SpinLock::new(BTreeMap::new()),
            connected_sockets: RwLock::new(BTreeMap::new()),
            used_ports: SpinLock::new(BTreeSet::new()),
        }
    }

    /// Creates a host-side vsock space backed by `/dev/vhost-vsock`.
    pub fn new_host_vhost() -> Self {
        Self {
            driver: None,
            connecting_sockets: SpinLock::new(BTreeMap::new()),
            listen_sockets: SpinLock::new(BTreeMap::new()),
            connected_sockets: RwLock::new(BTreeMap::new()),
            used_ports: SpinLock::new(BTreeSet::new()),
        }
    }

    /// Check whether the event is for this socket space
    fn is_event_for_socket(&self, event: &VsockEvent) -> bool {
        self.find_connecting_socket(event).is_some()
            || self.find_listen_socket(event).is_some()
            || self.find_connected_socket(event).is_some()
    }

    fn find_connecting_socket(&self, event: &VsockEvent) -> Option<Arc<Connecting>> {
        let destination: VsockSocketAddr = event.destination.into();
        let peer: VsockSocketAddr = event.source.into();
        let connecting_sockets = self.connecting_sockets.disable_irq().lock();
        connecting_sockets.get(&destination).cloned().or_else(|| {
            connecting_sockets
                .values()
                .find(|socket| {
                    socket.local_addr().port == destination.port && socket.peer_addr() == peer
                })
                .cloned()
        })
    }

    fn find_listen_socket(&self, event: &VsockEvent) -> Option<Arc<Listen>> {
        let destination: VsockSocketAddr = event.destination.into();
        let listen_sockets = self.listen_sockets.disable_irq().lock();
        listen_sockets.get(&destination).cloned().or_else(|| {
            listen_sockets
                .values()
                .find(|socket| socket.addr().port == destination.port)
                .cloned()
        })
    }

    fn find_connected_socket(&self, event: &VsockEvent) -> Option<Arc<Connected>> {
        let id: ConnectionID = (*event).into();
        let destination: VsockSocketAddr = event.destination.into();
        let peer: VsockSocketAddr = event.source.into();
        let connected_sockets = self.connected_sockets.read();
        connected_sockets.get(&id).cloned().or_else(|| {
            connected_sockets
                .iter()
                .find(|(id, _)| id.local_addr.port == destination.port && id.peer_addr == peer)
                .map(|(_, socket)| socket.clone())
        })
    }

    /// Alloc an unused port range
    pub fn alloc_ephemeral_port(&self) -> Result<u32> {
        let mut used_ports = self.used_ports.disable_irq().lock();
        // FIXME: the maximal port number is not defined by spec
        for port in 1024..u32::MAX {
            if !used_ports.contains(&port) {
                used_ports.insert(port);
                return Ok(port);
            }
        }
        return_errno_with_message!(Errno::EAGAIN, "cannot find unused high port");
    }

    /// Bind a port
    pub fn bind_port(&self, port: u32) -> bool {
        let mut used_ports = self.used_ports.disable_irq().lock();
        used_ports.insert(port)
    }

    /// Recycle a port
    pub fn recycle_port(&self, port: &u32) -> bool {
        let mut used_ports = self.used_ports.disable_irq().lock();
        used_ports.remove(port)
    }

    /// Insert a connected socket
    pub fn insert_connected_socket(
        &self,
        id: ConnectionID,
        connected: Arc<Connected>,
    ) -> Option<Arc<Connected>> {
        let mut connected_sockets = self.connected_sockets.write();
        connected_sockets.insert(id, connected)
    }

    /// Remove a connected socket
    pub fn remove_connected_socket(&self, id: &ConnectionID) -> Option<Arc<Connected>> {
        let mut connected_sockets = self.connected_sockets.write();
        connected_sockets.remove(id)
    }

    /// Insert a connecting socket
    pub fn insert_connecting_socket(
        &self,
        addr: VsockSocketAddr,
        connecting: Arc<Connecting>,
    ) -> Option<Arc<Connecting>> {
        error!(
            "vhost-vsock insert connecting key={:?} local={:?} peer={:?}",
            addr,
            connecting.local_addr(),
            connecting.peer_addr()
        );
        let mut connecting_sockets = self.connecting_sockets.disable_irq().lock();
        connecting_sockets.insert(addr, connecting)
    }

    /// Remove a connecting socket
    pub fn remove_connecting_socket(&self, addr: &VsockSocketAddr) -> Option<Arc<Connecting>> {
        error!("vhost-vsock remove connecting key={:?}", addr);
        let mut connecting_sockets = self.connecting_sockets.disable_irq().lock();
        connecting_sockets.remove(addr)
    }

    /// Removes this socket's map entry when it is dropped.
    ///
    /// A stale `Connecting` may be dropped after a newer attempt reused the
    /// same local address. In that case, keep the newer map entry and do not
    /// recycle its port.
    pub fn remove_connecting_socket_for_drop(&self, addr: &VsockSocketAddr, token: u64) -> bool {
        let mut connecting_sockets = self.connecting_sockets.disable_irq().lock();
        let Some(socket) = connecting_sockets.get(addr) else {
            return true;
        };
        if socket.token() != token {
            error!(
                "vhost-vsock keep newer connecting socket key={:?} drop_token={} live_token={}",
                addr,
                token,
                socket.token()
            );
            return false;
        }

        connecting_sockets.remove(addr);
        true
    }

    /// Insert a listening socket
    pub fn insert_listen_socket(
        &self,
        addr: VsockSocketAddr,
        listen: Arc<Listen>,
    ) -> Option<Arc<Listen>> {
        let mut listen_sockets = self.listen_sockets.disable_irq().lock();
        listen_sockets.insert(addr, listen)
    }

    /// Remove a listening socket
    pub fn remove_listen_socket(&self, addr: &VsockSocketAddr) -> Option<Arc<Listen>> {
        let mut listen_sockets = self.listen_sockets.disable_irq().lock();
        listen_sockets.remove(addr)
    }
}

impl VsockSpace {
    /// Get the CID of the guest
    pub fn guest_cid(&self) -> u32 {
        const VMADDR_CID_HOST: u32 = 2;
        let Some(driver) = self.driver.as_ref() else {
            return VMADDR_CID_HOST;
        };
        let driver = driver.disable_irq().lock();
        driver.guest_cid() as u32
    }

    /// Send a request packet for initializing a new connection.
    pub fn request(&self, info: &ConnectionInfo) -> Result<()> {
        if crate::device::misc::vhost_vsock::request(info)? {
            return Ok(());
        }
        let Some(driver) = self.driver.as_ref() else {
            return_errno_with_message!(Errno::ENODEV, "no vsock transport is available");
        };
        let mut driver = driver.disable_irq().lock();
        driver
            .request(info)
            .map_err(|_| Error::with_message(Errno::EIO, "cannot send connect packet"))
    }

    /// Send a response packet for accepting a new connection.
    pub fn response(&self, info: &ConnectionInfo) -> Result<()> {
        let Some(driver) = self.driver.as_ref() else {
            return_errno_with_message!(Errno::EOPNOTSUPP, "vhost-vsock listen is not implemented");
        };
        let mut driver = driver.disable_irq().lock();
        driver
            .response(info)
            .map_err(|_| Error::with_message(Errno::EIO, "cannot send response packet"))
    }

    /// Send a shutdown packet to close a connection
    #[expect(dead_code)]
    pub fn shutdown(&self, info: &ConnectionInfo) -> Result<()> {
        let Some(driver) = self.driver.as_ref() else {
            return crate::device::misc::vhost_vsock::reset(info).map(|_| ());
        };
        let mut driver = driver.disable_irq().lock();
        driver
            .shutdown(info)
            .map_err(|_| Error::with_message(Errno::EIO, "cannot send shutdown packet"))
    }

    /// Send a reset packet to reset a connection
    pub fn reset(&self, info: &ConnectionInfo) -> Result<()> {
        if crate::device::misc::vhost_vsock::reset(info)? {
            return Ok(());
        }
        let Some(driver) = self.driver.as_ref() else {
            return Ok(());
        };
        let mut driver = driver.disable_irq().lock();
        driver
            .reset(info)
            .map_err(|_| Error::with_message(Errno::EIO, "cannot send reset packet"))
    }

    /// Send a credit request packet
    #[expect(dead_code)]
    pub fn request_credit(&self, info: &ConnectionInfo) -> Result<()> {
        let Some(driver) = self.driver.as_ref() else {
            return_errno_with_message!(
                Errno::EOPNOTSUPP,
                "vhost-vsock credit request is not implemented"
            );
        };
        let mut driver = driver.disable_irq().lock();
        driver
            .credit_request(info)
            .map_err(|_| Error::with_message(Errno::EIO, "cannot send credit request packet"))
    }

    /// Send a credit update packet
    #[expect(dead_code)]
    pub fn update_credit(&self, info: &ConnectionInfo) -> Result<()> {
        let Some(driver) = self.driver.as_ref() else {
            return crate::device::misc::vhost_vsock::credit_update(info).map(|_| ());
        };
        let mut driver = driver.disable_irq().lock();
        driver
            .credit_update(info)
            .map_err(|_| Error::with_message(Errno::EIO, "cannot send credit update packet"))
    }

    /// Send a data packet
    pub fn send(&self, reader: &mut dyn MultiRead, info: &mut ConnectionInfo) -> Result<()> {
        // FIXME: Creating this buffer should be avoided
        // if the underlying driver can accept reader.
        let mut buffer = vec![0u8; reader.sum_lens()];
        reader.read(&mut VmWriter::from(buffer.as_mut_slice()))?;

        if crate::device::misc::vhost_vsock::send(info, &buffer)? {
            return Ok(());
        }
        let Some(driver) = self.driver.as_ref() else {
            return_errno_with_message!(Errno::ENODEV, "no vsock transport is available");
        };
        let mut driver = driver.disable_irq().lock();
        driver
            .send(&buffer, info)
            .map_err(|_| Error::with_message(Errno::EIO, "cannot send data packet"))
    }

    fn dump_connecting_sockets(&self) {
        let connecting_sockets = self.connecting_sockets.disable_irq().lock();
        for (addr, socket) in connecting_sockets.iter() {
            error!(
                "vhost-vsock pending connecting key={:?} local={:?} peer={:?}",
                addr,
                socket.local_addr(),
                socket.peer_addr()
            );
        }
    }

    pub fn handle_event(&self, event: VsockEvent, body: &[u8]) -> Result<()> {
        error!(
            "vhost-vsock handle event {:?} body_len={}",
            event,
            body.len()
        );
        if let Some(connected) = self.find_connected_socket(&event) {
            connected.update_info(&event);
        }

        match event.event_type {
            VsockEventType::ConnectionRequest => {
                let Some(listen) = self.find_listen_socket(&event) else {
                    error!(
                        "vhost-vsock request has no listening socket destination={:?}",
                        event.destination
                    );
                    return Ok(());
                };
                let peer = event.source;
                let connected = Arc::new(Connected::new(peer.into(), listen.addr()));
                connected.update_info(&event);
                listen.push_incoming(connected).unwrap();
            }
            VsockEventType::ConnectionResponse => {
                let Some(connecting) = self.find_connecting_socket(&event) else {
                    error!(
                        "vhost-vsock response has no connecting socket destination={:?}",
                        event.destination
                    );
                    self.dump_connecting_sockets();
                    return Ok(());
                };
                error!(
                    "match a connecting socket. Peer{:?}; local{:?}",
                    connecting.peer_addr(),
                    connecting.local_addr()
                );
                connecting.update_info(&event);
                connecting.set_connected();
                crate::device::misc::vhost_vsock::complete_request(&event);
            }
            VsockEventType::Disconnected { .. } => {
                if let Some(connecting) = self.find_connecting_socket(&event) {
                    connecting.update_info(&event);
                    connecting.set_failed(Errno::ECONNRESET);
                    return Ok(());
                }
                let Some(connected) = self.find_connected_socket(&event) else {
                    error!("vhost-vsock disconnected event has no socket {:?}", event);
                    return Ok(());
                };
                connected.set_peer_requested_shutdown();
            }
            VsockEventType::Received { .. } => {
                let Some(connected) = self.find_connected_socket(&event) else {
                    error!("vhost-vsock data event has no connected socket {:?}", event);
                    return Ok(());
                };
                if !connected.add_connection_buffer(body) {
                    return_errno_with_message!(Errno::ENOBUFS, "vsock receive buffer is full");
                }
            }
            VsockEventType::CreditRequest => {
                let Some(connected) = self.find_connected_socket(&event) else {
                    error!(
                        "vhost-vsock credit request has no connected socket {:?}",
                        event
                    );
                    return Ok(());
                };
                self.update_credit(&connected.get_info())?;
            }
            VsockEventType::CreditUpdate => {
                let Some(connected) = self.find_connected_socket(&event) else {
                    error!(
                        "vhost-vsock credit update has no connected socket {:?}",
                        event
                    );
                    return Ok(());
                };
                connected.update_info(&event);
            }
        }
        Ok(())
    }

    /// Poll for each event from the driver
    pub fn poll(&self) -> Result<()> {
        let Some(driver) = self.driver.as_ref() else {
            return Ok(());
        };
        let mut driver = driver.disable_irq().lock();

        while let Some(event) = self.poll_single(&mut driver)? {
            self.handle_event(event, &[])?;
        }
        Ok(())
    }

    fn poll_single(&self, driver: &mut SocketDevice) -> Result<Option<VsockEvent>> {
        driver
            .poll(|event, body| {
                // Deal with Received before the buffer are recycled.
                if let VsockEventType::Received { .. } = event.event_type {
                    // Only consider the connected socket and copy body to buffer
                    let connected_sockets = self.connected_sockets.read();
                    let connected = connected_sockets.get(&event.into()).unwrap();
                    debug!("Rw matches a connection with id {:?}", connected.id());
                    if !connected.add_connection_buffer(body) {
                        return Err(SocketError::BufferTooShort);
                    }
                }
                Ok(Some(event))
            })
            .map_err(|_| Error::with_message(Errno::EIO, "driver poll failed"))
    }
}
