// SPDX-License-Identifier: MPL-2.0

//! Guest virtualization support.

mod gpm_space;
mod interrupt;
mod timer;

pub use self::{
    gpm_space::GuestPhysMemSpace, interrupt::GuestInterruptPort, timer::GuestTimerPort,
};
pub use crate::arch::vm::guest_mode::{GuestMode, GuestRunResult};
