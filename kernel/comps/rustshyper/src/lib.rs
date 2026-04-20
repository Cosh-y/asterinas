//! RustShyper - Asterinas version of KVM
//!
//! This module provides hypervisor functionality similar to KVM,
//! allowing userspace processes to create and manage virtual machines.

#![no_std]
#![deny(unsafe_code)]
#![feature(trait_upcasting)]

extern crate alloc;

mod emulate;
mod error;
mod handler;
pub mod interrupt;
pub mod vm;

pub use error::{Errno, Error};
pub use vm::{Vcpu, Vm};

use crate::error::*;

/// Initializes the RustShyper subsystem
pub fn init() -> Result<()> {
    log::info!("Initializing RustShyper hypervisor");
    ostd::arch::virt::init_vmx().map_err(Error::from)?;
    log::info!("RustShyper VMX backend initialized");
    Ok(())
}
