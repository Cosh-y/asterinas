// SPDX-License-Identifier: MPL-2.0

use core::fmt::Result;

use aster_input::event_type_codes::KeyCode::G;
use miniz_oxide::MZStatus::Ok;

use super::{
    ioctl::{
        SetTssAddr, SetGsiRouting, IrqLine, CreatePit2, GetPit2, SetPit2, CreateIrqchip, GetIrqchip, SetIrqchip,
        KVM_IRQ_ROUTING_IRQCHIP, KVM_IRQCHIP_IOAPIC, KVM_IRQCHIP_PIC_MASTER, KVM_IRQCHIP_PIC_SLAVE, 
        KvmGsiRouting, KvmGsiRoutingEntry, KvmPitConfig, read_trailing_array,
    },
    apic::{Ioapic, Pic8259, Pit},
};
use crate::{
    dispatch_ioctl,
    kvm::vm::Vm,
    prelude::*,
    util::ioctl::RawIoctl
};

pub(super) enum IrqchipMode {
    None,
    Split,
    InKernel,
}

pub(in crate::kvm) struct VmArch {
    vm: Weak<Vm>,
    irqchip_mode: SpinLock<IrqchipMode>,
    ioapic: SpinLock<Option<Ioapic>>,
    pic: SpinLock<Option<Pic8259>>,
    pit: SpinLock<Option<Pit>>,
    gsi_routing: SpinLock<BTreeMap<u32, Vec<GsiRoute>>>,
}

impl VmArch {
    pub(in crate::kvm) fn new(vm: Weak<Vm>) -> Self {
        Self {
            vm,
            irqchip_mode: SpinLock::new(IrqchipMode::None),
            ioapic: SpinLock::new(None),
            pic: SpinLock::new(None),
            pit: SpinLock::new(None),
            gsi_routing: SpinLock::new(BTreeMap::new()),
        }
    }

    pub(super) fn irqchip_mode(&self) -> SpinLockGuard<'_, IrqchipMode> {
        self.irqchip_mode.lock()
    }

    pub(super) fn ioapic(&self) -> SpinLockGuard<'_, Ioapic> {
        self.ioapic.lock()
    }

    pub(super) fn pic(&self) -> SpinLockGuard<'_, Pic8259> {
        self.pic.lock()
    }

    pub(super) fn pit(&self) -> SpinLockGuard<'_, Pit> {
        self.pit.lock()
    }
}

impl Vm {
    pub(crate) fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        dispatch_ioctl!(match raw_ioctl {
            SetTssAddr => {
                Ok(0)
            }
            cmd @ CreateIrqchip => {
                *self.arch().irqchip_mode.lock() = IrqchipMode::InKernel;
                *self.arch().ioapic.lock() = Some(Ioapic::new(self.arch().vm.clone()));
                *self.arch().pic.lock() = Some(Pic8259::new());
                Ok(0)
            }
            cmd @ GetIrqchip => {
                Ok(0)
            }
            cmd @ SetIrqchip => {
                Ok(0)
            }
            cmd @ SetGsiRouting => {
                let entries = read_trailing_array(raw_ioctl, size_of::<KvmGsiRouting>(), cmd.read()?.nr)?;
                self.set_gsi_routing(&entries)?;
                Ok(0)
            }
            cmd @ CreatePit2 => {
                if self.arch().pit.lock().is_some() {
                    return_errno_with_message!(Errno::EEXIST, "PIT already exists");
                }

                if self.arch().pic.lock().is_none() {
                    return_errno_with_message!(Errno::ENOENT, "PIC does not exist");
                }

                let config = cmd.read()?;
                *self.arch().pit.lock() = Some(Pit::new(self.arch().vm.clone(), config.flags & 1 != 0));
                Ok(0)
            }
            cmd @ GetPit2 => {
                Ok(0)
            }
            cmd @ SetPit2 => {
                Ok(0)
            }
            cmd @ IrqLine => {
                // 查 gsi_routing
                let gsi = cmd.read()?;
                self.set_gsi_level(gsi.gsi, gsi.level)
            }
            _ => {
                return_errno_with_message!(Errno::ENOTTY, "unsupported VM ioctl")
            }
        })
    }

    fn set_gsi_routing(&self, entries: &[KvmGsiRoutingEntry]) -> Result<()> {
        let routings = &mut self.arch().gsi_routing.lock();

        for entry in entries {
            debug_assert!(entry.type_ == KVM_IRQ_ROUTING_IRQCHIP && entry.flags == 0);
            
            let routings_gsi = routings.entry(entry.gsi).or_default();

            match entry.data[0] {
                KVM_IRQCHIP_IOAPIC => {
                    let pin = entry.data[1] as usize;
                    routings_gsi.push(GsiRoute::Ioapic { pin });
                }
                KVM_IRQCHIP_PIC_MASTER => {
                    let pin = entry.data[1] as usize;
                    routings_gsi.push(GsiRoute::PicMaster { pin });
                }
                KVM_IRQCHIP_PIC_SLAVE => {
                    let pin = entry.data[1] as usize;
                    routings_gsi.push(GsiRoute::PicSlave { pin });
                }
                _ => {
                    return_errno_with_message!(Errno::EINVAL, "unsupported gsi routing entry type");
                }
            }
        }

        Ok(())
    }

    pub(super) fn set_gsi_level(&self, gsi: u32, level: u32) -> Result<()> {
        let routings = self.arch().gsi_routing.lock();
        let routings_gsi = routings
            .get(&gsi)
            .copied()
            .ok_or(Error::with_message(Errno::EINVAL, "unknown GSI"))?;
        drop(routings);

        for routing in routings_gsi {
            match routing {
                GsiRoute::PicMaster { pin } => {
                    self.arch().pic().set_ir(pin, level == 1);
                    
                    if self.arch().pic().quiry_irq().is_some() {
                        let vcpu = self
                            .vcpus()
                            .values()
                            .filter_map(Weak::upgrade)
                            .find(|vcpu| vcpu.arch().lapic().can_inject_external_interrupt());

                        if let Some(vcpu) = vcpu {
                            vcpu.notify();
                        }
                    }
                }
                GsiRoute::PicSlave { pin } => {
                    self.arch().pic().set_ir(pin + 8, level == 1);

                    if self.arch().pic().quiry_irq().is_some() {
                        let vcpu = self
                            .vcpus()
                            .values()
                            .filter_map(Weak::upgrade)
                            .find(|vcpu| vcpu.arch().lapic().can_inject_external_interrupt());

                        if let Some(vcpu) = vcpu {
                            vcpu.notify();
                        }
                    }
                }
                GsiRoute::Ioapic { pin } => {
                    self.arch().ioapic().set_irq_line(pin, level == 1)?;
                }
            }
        }

        Ok(())
    }
}

enum GsiRoute {
    PicMaster { pin: usize }, // Pic master's IR0 - IR7
    PicSlave { pin: usize }, // Pic slave's IR0 - IR7
    Ioapic { pin: usize },
}
