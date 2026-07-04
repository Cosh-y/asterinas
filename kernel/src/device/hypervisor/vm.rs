use core::cmp;

use ostd::{
    mm::{Gpaddr, PageProperty, UFrame, io::util::HasVmReaderWriter},
    task::disable_preempt,
    vm::GuestPhysMemSpace,
};

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

struct MemorySlot {
    guest_start: Gpaddr,
    guest_end: Gpaddr,
    frames: Vec<UFrame>,
}

impl MemorySlot {
    fn new(guest_start: Gpaddr, memory_size: usize, frames: Vec<UFrame>) -> Result<Self> {
        let guest_end = guest_start
            .checked_add(memory_size)
            .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
        if frames.len().checked_mul(PAGE_SIZE) != Some(memory_size) {
            return_errno_with_message!(Errno::EINVAL, "guest memory frame count is invalid");
        }

        Ok(Self {
            guest_start,
            guest_end,
            frames,
        })
    }

    fn guest_range(&self) -> core::ops::Range<Gpaddr> {
        self.guest_start..self.guest_end
    }

    fn overlaps_guest_range(&self, guest_range: &core::ops::Range<Gpaddr>) -> bool {
        self.guest_start < guest_range.end && guest_range.start < self.guest_end
    }

    fn backing_frame(&self, gpa: Gpaddr) -> Option<(UFrame, usize, usize)> {
        if !(self.guest_start..self.guest_end).contains(&gpa) {
            return None;
        }

        let offset = gpa - self.guest_start;
        let frame_index = offset / PAGE_SIZE;
        let frame_offset = offset % PAGE_SIZE;
        let len = cmp::min(PAGE_SIZE - frame_offset, self.guest_end - gpa);
        Some((self.frames[frame_index].clone(), frame_offset, len))
    }
}

pub(super) struct Vm {
    pub(super) id: u32,
    guest_mem: GuestPhysMemSpace,
    memory_slots: Mutex<BTreeMap<u32, MemorySlot>>,
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
            memory_slots: Mutex::new(BTreeMap::new()),
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

    pub(super) fn set_user_memory_region(
        &self,
        slot: u32,
        guest_start: Gpaddr,
        memory_size: usize,
        frames: Vec<UFrame>,
        prop: PageProperty,
    ) -> Result<()> {
        if memory_size == 0 {
            let old_slot = self.memory_slots.lock().remove(&slot);
            if let Some(old_slot) = old_slot {
                self.unmap_guest_memory(old_slot.guest_range())?;
            }
            return Ok(());
        }

        let new_slot = MemorySlot::new(guest_start, memory_size, frames)?;
        let new_guest_range = new_slot.guest_range();

        let mut memory_slots = self.memory_slots.lock();
        for (&existing_slot_id, existing_slot) in memory_slots.iter() {
            if existing_slot_id != slot && existing_slot.overlaps_guest_range(&new_guest_range) {
                return_errno_with_message!(Errno::EINVAL, "guest memory slots overlap");
            }
        }

        let old_slot = memory_slots.remove(&slot);
        if let Some(old_slot) = old_slot {
            self.unmap_guest_memory(old_slot.guest_range())?;
        }
        self.map_guest_memory(new_guest_range, &new_slot.frames, prop)?;
        memory_slots.insert(slot, new_slot);
        Ok(())
    }

    pub(super) fn read_guest_bytes(&self, gpa: Gpaddr, bytes: &mut [u8]) -> Result<()> {
        let mut writer = VmWriter::from(&mut *bytes).to_fallible();
        let read_len = self.read_guest(gpa, &mut writer)?;
        if read_len != bytes.len() {
            return_errno_with_message!(Errno::EFAULT, "guest memory read was incomplete");
        }
        Ok(())
    }

    pub(super) fn read_guest_val<T: Pod>(&self, gpa: Gpaddr) -> Result<T> {
        let mut value = T::new_zeroed();
        let mut writer = VmWriter::from(value.as_mut_bytes()).to_fallible();
        let read_len = self.read_guest(gpa, &mut writer)?;
        if read_len != size_of::<T>() {
            return_errno_with_message!(Errno::EFAULT, "guest memory read was incomplete");
        }
        Ok(value)
    }

    pub(super) fn write_guest_val<T: Pod>(&self, gpa: Gpaddr, value: &T) -> Result<()> {
        let mut reader = VmReader::from(value.as_bytes()).to_fallible();
        let written_len = self.write_guest(gpa, &mut reader)?;
        if written_len != size_of::<T>() {
            return_errno_with_message!(Errno::EFAULT, "guest memory write was incomplete");
        }
        Ok(())
    }

    fn map_guest_memory(
        &self,
        guest_range: core::ops::Range<Gpaddr>,
        frames: &[UFrame],
        prop: PageProperty,
    ) -> Result<()> {
        let preempt_guard = disable_preempt();
        let mut cursor = self.guest_mem.cursor_mut(&preempt_guard, &guest_range)?;
        for frame in frames {
            cursor.map(frame.clone(), prop);
        }
        Ok(())
    }

    fn unmap_guest_memory(&self, guest_range: core::ops::Range<Gpaddr>) -> Result<usize> {
        let preempt_guard = disable_preempt();
        let mut cursor = self.guest_mem.cursor_mut(&preempt_guard, &guest_range)?;
        Ok(cursor.unmap(guest_range.end - guest_range.start))
    }

    fn read_guest(&self, gpa: Gpaddr, writer: &mut VmWriter) -> Result<usize> {
        let mut current_gpa = gpa;
        let mut total_read = 0;

        while writer.avail() > 0 {
            let (frame, frame_offset, max_len) = self.guest_backing_frame(current_gpa)?;
            let read_len = cmp::min(writer.avail(), max_len);
            let mut frame_reader = frame.reader();
            frame_reader.skip(frame_offset).limit(read_len);

            let copied = frame_reader.read_fallible(writer).map_err(Error::from)?;
            total_read += copied;
            current_gpa = current_gpa
                .checked_add(copied)
                .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
        }

        Ok(total_read)
    }

    fn write_guest(&self, gpa: Gpaddr, reader: &mut VmReader) -> Result<usize> {
        let mut current_gpa = gpa;
        let mut total_written = 0;

        while reader.remain() > 0 {
            let (frame, frame_offset, max_len) = self.guest_backing_frame(current_gpa)?;
            let write_len = cmp::min(reader.remain(), max_len);
            let mut frame_writer = frame.writer();
            frame_writer.skip(frame_offset).limit(write_len);

            let copied = frame_writer.write_fallible(reader).map_err(Error::from)?;
            total_written += copied;
            current_gpa = current_gpa
                .checked_add(copied)
                .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
        }

        Ok(total_written)
    }

    fn guest_backing_frame(&self, gpa: Gpaddr) -> Result<(UFrame, usize, usize)> {
        let memory_slots = self.memory_slots.lock();
        for memory_slot in memory_slots.values() {
            if let Some(backing_frame) = memory_slot.backing_frame(gpa) {
                return Ok(backing_frame);
            }
        }

        return_errno_with_message!(Errno::EFAULT, "guest physical address is not backed");
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
