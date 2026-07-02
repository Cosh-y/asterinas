use ostd::vm::GuestPhysMemSpace;

use super::{
    apic::{
        IOAPIC_NUM_PINS, Icr, Ioapic, Lapic, default_lapic_ldr, default_lapic_lvt_lint0,
        icr_matches_destination,
    },
    ioctl::{
        ClockData, EnableCapData, IrqLevel, IrqRoutingEntry, KVM_CAP_MAX_VCPU_ID,
        KVM_CAP_SPLIT_IRQCHIP, KVM_IRQ_ROUTING_IRQCHIP, KVM_IRQ_ROUTING_MSI, KVM_IRQCHIP_IOAPIC,
    },
    vcpu::Vcpu,
};
use crate::prelude::*;

const KVM_CLOCK_REALTIME: u32 = 1 << 2;
const KVM_CLOCK_HOST_TSC: u32 = 1 << 3;

#[derive(Clone, Copy, Debug)]
enum IrqRoute {
    Ioapic { pin: usize },
}

pub(super) struct Vm {
    pub(super) id: u32,
    guest_mem: GuestPhysMemSpace,
    vcpus: Mutex<BTreeMap<u32, Arc<Vcpu>>>,
    ioapic: Mutex<Ioapic>,
    irqchip_created: Mutex<bool>,
    irq_routes: Mutex<BTreeMap<u32, Vec<IrqRoute>>>,
    clock: Mutex<ClockData>,
}

impl Vm {
    pub fn new(id: u32) -> Arc<Self> {
        Arc::new(Self {
            id,
            guest_mem: GuestPhysMemSpace::new(),
            vcpus: Mutex::new(BTreeMap::new()),
            ioapic: Mutex::new(Ioapic::default()),
            irqchip_created: Mutex::new(false),
            irq_routes: Mutex::new(BTreeMap::new()),
            clock: Mutex::new(ClockData::default()),
        })
    }

    pub fn ioapic(&self) -> MutexGuard<'_, Ioapic> {
        self.ioapic.lock()
    }

    pub fn guest_mem(&self) -> &GuestPhysMemSpace {
        &self.guest_mem
    }

    pub(super) fn create_vcpu(self: &Arc<Self>, vcpu_id: u32) -> Result<Arc<Vcpu>> {
        let mut vcpus = self.vcpus.lock();
        if vcpus.contains_key(&vcpu_id) {
            return_errno_with_message!(Errno::EEXIST, "vCPU already exists");
        }

        let mut lapic = Lapic::default();
        lapic.id = vcpu_id;
        lapic.ldr = default_lapic_ldr(vcpu_id);
        lapic.lvt_lint0 = default_lapic_lvt_lint0(vcpu_id);

        let vcpu = Vcpu::new(vcpu_id, self, lapic)?;
        vcpus.insert(vcpu_id, vcpu.clone());
        drop(vcpus);
        Ok(vcpu)
    }

    pub(super) fn create_irqchip(&self) -> Result<()> {
        // TODO: Add PIC state and stricter KVM irqchip lifecycle checks.
        *self.ioapic.lock() = Ioapic::default();
        self.irq_routes.lock().clear();
        *self.irqchip_created.lock() = true;
        Ok(())
    }

    pub(super) fn set_clock(&self, clock: ClockData) {
        *self.clock.lock() = clock;
    }

    pub(super) fn get_clock(&self) -> ClockData {
        let mut clock = *self.clock.lock();
        clock.clock = monotonic_nanos();
        clock.host_tsc = ostd::arch::read_tsc();
        clock.flags |= KVM_CLOCK_HOST_TSC;
        if let Ok(realtime) = realtime_nanos() {
            clock.realtime = realtime;
            clock.flags |= KVM_CLOCK_REALTIME;
        }
        clock
    }

    pub(super) fn enable_cap(&self, cap: EnableCapData) -> Result<()> {
        match usize::try_from(cap.cap)? {
            KVM_CAP_SPLIT_IRQCHIP => {
                return_errno_with_message!(Errno::EINVAL, "split irqchip is not supported");
            }
            KVM_CAP_MAX_VCPU_ID => Ok(()),
            _ => {
                return_errno_with_message!(Errno::EINVAL, "unsupported VM capability");
            }
        }
    }

    pub(super) fn set_gsi_routing(&self, entries: &[IrqRoutingEntry]) -> Result<()> {
        self.ensure_irqchip_created()?;

        let mut irq_routes = BTreeMap::new();
        for entry in entries {
            match entry.type_ {
                KVM_IRQ_ROUTING_IRQCHIP => {
                    let irqchip = entry.data[0];
                    let pin = usize::try_from(entry.data[1])?;
                    if irqchip != KVM_IRQCHIP_IOAPIC {
                        continue;
                    }
                    if pin >= IOAPIC_NUM_PINS {
                        return_errno_with_message!(
                            Errno::EINVAL,
                            "GSI route references an out-of-range IOAPIC pin"
                        );
                    }

                    irq_routes
                        .entry(entry.gsi)
                        .or_insert_with(Vec::new)
                        .push(IrqRoute::Ioapic { pin });
                }
                KVM_IRQ_ROUTING_MSI => {
                    debug!("hypervisor: ignoring MSI GSI route {}", entry.gsi);
                }
                route_type => {
                    debug!(
                        "hypervisor: ignoring unsupported GSI route type {} for GSI {}",
                        route_type, entry.gsi
                    );
                }
            }
        }

        *self.irq_routes.lock() = irq_routes;
        Ok(())
    }

    pub(super) fn set_irq_line(&self, irq_level: IrqLevel) -> Result<bool> {
        self.ensure_irqchip_created()?;

        if irq_level.level == 0 {
            return Ok(false);
        }

        let (routes, has_routing_table) = {
            let irq_routes = self.irq_routes.lock();
            (
                irq_routes.get(&irq_level.irq).cloned(),
                !irq_routes.is_empty(),
            )
        };
        let routes = match routes {
            Some(routes) => routes,
            None if !has_routing_table => {
                let pin = usize::try_from(irq_level.irq)?;
                if pin >= IOAPIC_NUM_PINS {
                    return_errno_with_message!(
                        Errno::EINVAL,
                        "IRQ line is out of range for the emulated I/O APIC"
                    );
                }
                vec![IrqRoute::Ioapic { pin }]
            }
            None => return Ok(false),
        };

        let mut delivered = false;
        for route in routes {
            match route {
                IrqRoute::Ioapic { pin } => {
                    self.inject_ioapic_pin(pin)?;
                    delivered = true;
                }
            }
        }
        Ok(delivered)
    }

    fn ensure_irqchip_created(&self) -> Result<()> {
        if *self.irqchip_created.lock() {
            return Ok(());
        }

        return_errno_with_message!(Errno::EINVAL, "in-kernel irqchip has not been created");
    }

    fn inject_ioapic_pin(&self, pin: usize) -> Result<()> {
        if pin >= IOAPIC_NUM_PINS {
            return_errno_with_message!(
                Errno::EINVAL,
                "IOAPIC pin is out of range for the emulated I/O APIC"
            );
        }

        let vcpus = self.vcpus.lock().values().cloned().collect::<Vec<_>>();
        if vcpus.is_empty() {
            return_errno_with_message!(Errno::EINVAL, "cannot inject IRQ without any vCPU");
        }

        let mut lapics = vcpus.iter().map(|vcpu| vcpu.lapic()).collect::<Vec<_>>();
        let mut ioapic = self.ioapic.lock();
        ioapic.inject_irq_line(lapics.iter_mut().map(|lapic| &mut **lapic), pin);
        Ok(())
    }

    pub fn inject_ipi(&self, icr: Icr) -> Result<()> {
        let vcpus: Vec<_> = self
            .vcpus
            .lock()
            .iter()
            .map(|(&vcpu_id, vcpu)| (vcpu_id, vcpu.clone()))
            .collect();

        for (vcpu_id, vcpu) in vcpus {
            if !icr_matches_destination(&vcpu.lapic(), &icr) {
                continue;
            }

            const APIC_ICR_DELIVERY_MODE_FIXED: u8 = 0;
            const APIC_ICR_DELIVERY_MODE_INIT: u8 = 5;
            const APIC_ICR_DELIVERY_MODE_STARTUP: u8 = 6;

            match icr.delivery_mode {
                APIC_ICR_DELIVERY_MODE_FIXED => {
                    if icr.vector >= 16 {
                        vcpu.lapic().add_pending_interrupt(icr.vector);
                    }
                }
                APIC_ICR_DELIVERY_MODE_INIT => {
                    // Vcpu recieves INIT IPI, do nothing.
                    warn!(
                        "rustshyper: INIT IPI for vcpu {} is not fully migrated yet",
                        vcpu_id
                    );
                }
                APIC_ICR_DELIVERY_MODE_STARTUP => vcpu.receive_sipi(icr.vector),
                _ => {
                    error!(
                        "hypervisor: unsupported LAPIC ICR delivery mode {}",
                        icr.delivery_mode,
                    );
                }
            }
        }

        Ok(())
    }
}

pub(super) fn monotonic_nanos() -> u64 {
    let nanos = aster_time::read_monotonic_time().as_nanos();
    saturating_u128_to_u64(nanos)
}

pub(super) fn realtime_nanos() -> Result<u64> {
    let duration =
        crate::time::SystemTime::now().duration_since(&crate::time::SystemTime::UNIX_EPOCH)?;
    Ok(saturating_u128_to_u64(duration.as_nanos()))
}

fn saturating_u128_to_u64(nanos: u128) -> u64 {
    if nanos > u128::from(u64::MAX) {
        u64::MAX
    } else {
        nanos as u64
    }
}
