// SPDX-License-Identifier: MPL-2.0

use super::{
    ioctl::{
        SetTssAddr, SetGsiRouting, IrqLine,
        KVM_IRQ_ROUTING_IRQCHIP, KvmGsiRoutingEntry, read_gsi_routing_entries
    },
    apic::Ioapic,
};
use crate::{
    dispatch_ioctl,
    kvm::vm::Vm,
    prelude::*,
    util::ioctl::RawIoctl
};

pub(in crate::kvm) struct VmArch {
    ioapic: Ioapic,
    gsi_routing: Mutex<BTreeMap<u32, GsiRoute>>,
}

impl VmArch {
    pub(in crate::kvm) fn new() -> Self {
        Self {

        }
    }
}

impl Vm {
    pub(crate) fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        dispatch_ioctl!(match raw_ioctl {
            SetTssAddr => {
                Ok(0)
            }
            cmd @ SetGsiRouting => {
                let entries = read_gsi_routing_entries(&cmd, raw_ioctl)?;
                self.set_gsi_routing(&entries)?;
                Ok(0)
            }
            cmd @ IrqLine => {
                // 查 gsi_routing
                let gsi = cmd.read()?;
                let routings = self.arch().gsi_routing.lock();
                let routing = routings
                    .get(&gsi.gsi)
                    .copied()
                    .ok_or(Error::with_message(Errno::EINVAL, "unknown GSI"))?;
                drop(routings);

                match routing {
                    GsiRoute::Ioapic { pin } => {
                        self.arch().ioapic.set_irq_line(pin, gsi.level == 1)?;
                    }
                }
                Ok(0)
            }
            _ => {
                return_errno_with_message!(Errno::ENOTTY, "unsupported VM ioctl")
            }
        })
    }

    fn set_gsi_routing(&self, entries: &[KvmGsiRoutingEntry]) -> Result<()> {
        let gsi_routing = &mut self.arch().gsi_routing.lock();

        for entry in entries {
            match entry.type_ {
                KVM_IRQ_ROUTING_IRQCHIP => {
                    let pin = entry.data[1] as usize;
                    gsi_routing.insert(entry.gsi, GsiRoute::Ioapic { pin });
                }
                _ => {
                    return_errno_with_message!(Errno::EINVAL, "unsupported gsi routing entry type");
                }
            }
        }

        Ok(())
    }
}

enum GsiRoute {
    Ioapic { pin: usize },
}
