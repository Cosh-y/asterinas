mod apic;
mod cpuid;
mod cr;
mod device;
mod guest_address;
mod ioctl;
mod ioeventfd;
mod irqfd;
mod kvmclock;
mod mmio;
mod msr;
mod pio;
mod vcpu;
mod vcpu_file;
mod vm;
mod vm_file;
mod vm_memory;

pub use device::HypervisorDevice;

use crate::{device::registry::char, prelude::*};

const KVM_MAJOR: u16 = 10;
const KVM_MINOR: u16 = 232;

pub(super) fn init_in_first_process() -> Result<()> {
    char::register(Arc::new(HypervisorDevice::new()))?;
    Ok(())
}
