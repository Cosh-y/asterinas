// SPDX-License-Identifier: MPL-2.0

use super::ioctl::{
    IoEventFdConfig, KVM_IOEVENTFD_FLAG_DATAMATCH, KVM_IOEVENTFD_FLAG_DEASSIGN,
    KVM_IOEVENTFD_FLAG_PIO,
};
use crate::{events::KernelEventFile, prelude::*};

const KVM_IOEVENTFD_VALID_FLAGS: u32 =
    KVM_IOEVENTFD_FLAG_DATAMATCH | KVM_IOEVENTFD_FLAG_PIO | KVM_IOEVENTFD_FLAG_DEASSIGN;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum IoEventAddressSpace {
    Mmio,
    Pio,
}

pub(super) struct IoEventFdBinding {
    eventfd: Arc<KernelEventFile>,
    address_space: IoEventAddressSpace,
    addr: u64,
    len: u32,
    datamatch: Option<u64>,
    active: Mutex<bool>,
}

impl IoEventFdBinding {
    pub(super) fn new(config: IoEventFdConfig, eventfd: Arc<KernelEventFile>) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            eventfd,
            address_space: address_space(&config),
            addr: config.addr,
            len: config.len,
            datamatch: (config.flags & KVM_IOEVENTFD_FLAG_DATAMATCH != 0)
                .then_some(config.datamatch),
            active: Mutex::new(true),
        })
    }

    pub(super) fn conflicts_with(&self, other: &Self) -> bool {
        if self.address_space != other.address_space || self.addr != other.addr {
            return false;
        }

        self.len == 0
            || other.len == 0
            || (self.len == other.len
                && (self.datamatch.is_none()
                    || other.datamatch.is_none()
                    || self.datamatch == other.datamatch))
    }

    pub(super) fn matches_config(
        &self,
        config: &IoEventFdConfig,
        eventfd: &Arc<KernelEventFile>,
    ) -> bool {
        Arc::ptr_eq(&self.eventfd, eventfd)
            && self.address_space == address_space(config)
            && self.addr == config.addr
            && self.len == config.len
            && self.datamatch
                == (config.flags & KVM_IOEVENTFD_FLAG_DATAMATCH != 0).then_some(config.datamatch)
    }

    pub(super) fn matches_io(
        &self,
        address_space: IoEventAddressSpace,
        addr: u64,
        len: u32,
        value: u64,
    ) -> bool {
        self.matches_address(address_space, addr, len)
            && self.datamatch.is_none_or(|expected| expected == value)
    }

    pub(super) fn matches_address(
        &self,
        address_space: IoEventAddressSpace,
        addr: u64,
        len: u32,
    ) -> bool {
        self.address_space == address_space
            && self.addr == addr
            && (self.len == 0 || self.len == len)
    }

    pub(super) fn signal(&self) -> bool {
        let active = self.active.lock();
        if !*active {
            return false;
        }
        self.eventfd.signal();
        true
    }

    pub(super) fn deactivate(&self) {
        *self.active.lock() = false;
    }
}

pub(super) fn validate_config(config: &IoEventFdConfig) -> Result<()> {
    if config.flags & !KVM_IOEVENTFD_VALID_FLAGS != 0 {
        return_errno_with_message!(Errno::EINVAL, "unsupported ioeventfd flags");
    }
    if !matches!(config.len, 0 | 1 | 2 | 4 | 8) {
        return_errno_with_message!(Errno::EINVAL, "invalid ioeventfd length");
    }
    if config.len == 0 && config.flags & KVM_IOEVENTFD_FLAG_DATAMATCH != 0 {
        return_errno_with_message!(Errno::EINVAL, "zero-length ioeventfd cannot use datamatch");
    }
    config
        .addr
        .checked_add(u64::from(config.len))
        .ok_or_else(|| Error::new(Errno::EINVAL))?;
    if address_space(config) == IoEventAddressSpace::Pio
        && (config.addr >= 1 << 16 || config.addr + u64::from(config.len) > 1 << 16)
    {
        return_errno_with_message!(Errno::EINVAL, "ioeventfd PIO range is invalid");
    }
    Ok(())
}

fn address_space(config: &IoEventFdConfig) -> IoEventAddressSpace {
    if config.flags & KVM_IOEVENTFD_FLAG_PIO != 0 {
        IoEventAddressSpace::Pio
    } else {
        IoEventAddressSpace::Mmio
    }
}
