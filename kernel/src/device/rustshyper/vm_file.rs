// SPDX-License-Identifier: MPL-2.0

//! VM file descriptor implementation

use ostd::{
    arch::vm::{GuestContext, GuestCpuConfig},
    mm::{PAGE_SIZE, PageFlags, PageProperty, vm_space::VmQueriedItem},
    task::Task,
    vm::GuestPhysMemSpace,
};

use super::{
    apic::{IOAPIC_NUM_PINS, Icr, Ioapic, Lapic, default_lapic_ldr, icr_matches_destination},
    ioctl_defs,
    vcpu_file::{Vcpu, VcpuFile},
};
use crate::{
    fs::{
        file::{FileLike, file_table::FdFlags},
        pseudofs::AnonInodeFs,
        vfs::path::Path,
    },
    prelude::*,
    process::posix_thread::AsThreadLocal,
    util::ioctl::{RawIoctl, dispatch_ioctl},
    vm::vmar::{PageFaultInfo, Vmar},
};

/// VM file descriptor
pub struct VmFile {
    /// VmFile owns the Vm instance, but why 'Arc'?
    /// VcpuFiles need to reference the Vm, but can't act like
    /// struct VcpuFile<'a> { vm: &'a Vm, ... } because the
    /// VcpuFile needs to be 'static to be stored in the file table.
    vm: Arc<Vm>,
    pseudo_path: Path,
}

pub(super) struct Vm {
    id: u32,
    guest_mem: GuestPhysMemSpace,
    vcpus: Mutex<BTreeMap<u32, Arc<Vcpu>>>,
    ioapic: Mutex<Ioapic>,
}

impl Vm {
    pub fn new(id: u32) -> Arc<Self> {
        Arc::new(Self {
            id,
            guest_mem: GuestPhysMemSpace::new(),
            vcpus: Mutex::new(BTreeMap::new()),
            ioapic: Mutex::new(Ioapic::default()),
        })
    }

    pub fn ioapic(&self) -> MutexGuard<'_, Ioapic> {
        self.ioapic.lock()
    }

    pub fn guest_mem(&self) -> &GuestPhysMemSpace {
        &self.guest_mem
    }

    fn create_vcpu(self: &Arc<Self>, vcpu_id: u32) -> Result<Arc<Vcpu>> {
        let mut lapic = Lapic::default();
        lapic.id = vcpu_id;
        lapic.ldr = default_lapic_ldr(vcpu_id);

        let vcpu = Arc::new(Vcpu {
            id: vcpu_id,
            vm: Arc::downgrade(self),
            guest_context: Mutex::new(GuestContext::new(vcpu_id)?),
            lapic: SpinLock::new(lapic),
        });
        self.vcpus.lock().insert(vcpu_id, vcpu.clone());
        self.refresh_guest_cpu_config();
        Ok(vcpu)
    }

    fn refresh_guest_cpu_config(&self) {
        let vcpus = self.vcpus.lock();
        let vcpu_count = vcpus.len() as u32;
        for (&vcpu_id, vcpu) in vcpus.iter() {
            vcpu.guest_context().set_guest_cpu_config(GuestCpuConfig {
                vcpu_id,
                lapic_id: vcpu_id,
                vcpu_count,
            });
        }
    }

    fn inject_irq_line(&self, irq: usize) -> Result<()> {
        if irq >= IOAPIC_NUM_PINS {
            return_errno_with_message!(
                Errno::EINVAL,
                "IRQ line is out of range for the emulated I/O APIC"
            );
        }

        let vcpus = self.vcpus.lock().values().cloned().collect::<Vec<_>>();
        if vcpus.is_empty() {
            return_errno_with_message!(Errno::EINVAL, "cannot inject IRQ without any vCPU");
        }

        let mut lapics = vcpus.iter().map(|vcpu| vcpu.lapic()).collect::<Vec<_>>();
        let mut ioapic = self.ioapic.lock();
        ioapic.inject_irq_line(lapics.iter_mut().map(|lapic| &mut **lapic), irq);
        Ok(())
    }

    pub fn inject_ipi(&self, icr: Icr) -> Result<()> {
        let vcpus: alloc::vec::Vec<_> = self
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
                        "rustshyper: unsupported LAPIC ICR delivery mode {}",
                        icr.delivery_mode,
                    );
                }
            }
        }

        Ok(())
    }
}

fn query_user_ram_frame(vmar: &Vmar, userspace_addr: Vaddr) -> Result<ostd::mm::UFrame> {
    loop {
        let preempt_guard = ostd::task::disable_preempt();
        let vm_space = vmar.vm_space();
        let mut host_cursor = vm_space.cursor(
            &preempt_guard,
            &(userspace_addr..(userspace_addr + PAGE_SIZE)),
        )?;

        match host_cursor.query()?.1 {
            Some(VmQueriedItem::MappedRam { frame, .. }) => return Ok(frame.clone()),
            Some(VmQueriedItem::MappedIoMem { .. }) => {
                return_errno_with_message!(
                    Errno::EOPNOTSUPP,
                    "guest memory cannot be backed by userspace MMIO"
                );
            }
            None => {}
        }

        drop(host_cursor);
        drop(preempt_guard);

        vmar.handle_page_fault(&PageFaultInfo::new(userspace_addr, PageFlags::W.into()))?;
    }
}

impl VmFile {
    /// Creates a new VM file
    pub fn new(vm: Arc<Vm>) -> Self {
        let pseudo_path = AnonInodeFs::new_path(|_| "anon_inode:[rustshyper-vm]".to_string());
        Self { vm, pseudo_path }
    }
}

fn log_vm_error(context: &str, err: &Error) {
    match err.message() {
        Some(msg) => {
            error!(
                "rustshyper: {} failed: errno={:?}, msg={}",
                context,
                err.error(),
                msg
            );
        }
        None => {
            error!("rustshyper: {} failed: errno={:?}", context, err.error());
        }
    }
}

impl FileLike for VmFile {
    fn read(&self, _writer: &mut VmWriter) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot read from VM file");
    }

    fn write(&self, _reader: &mut VmReader) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot write to VM file");
    }

    fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        use ioctl_defs::*;

        dispatch_ioctl!(match raw_ioctl {
            cmd @ CreateVcpu => {
                let vcpu_id = cmd.read()?;

                // Create the VCPU
                let vcpu = self.vm.create_vcpu(vcpu_id)?;

                // Create a file descriptor for the VCPU
                let vcpu_file = Arc::new(VcpuFile::new(vcpu));

                // Insert into the current process's file table
                let current = Task::current().unwrap();
                let mut file_table = current.as_thread_local().unwrap().borrow_file_table_mut();
                let mut file_table_locked = file_table.unwrap().write();
                let vcpu_fd = file_table_locked.insert(vcpu_file, FdFlags::empty());

                Ok(vcpu_fd.into())
            }
            cmd @ SetUserMemoryRegion => {
                let region: UserMemoryRegion = cmd.read()?;

                let current = match Task::current() {
                    Some(current) => current,
                    None => {
                        error!("rustshyper: no current task found for rustshyper ioctl");
                        return Err(Error::new(Errno::ESRCH));
                    }
                };
                let thread_local = match current.as_thread_local() {
                    Some(thread_local) => thread_local,
                    None => {
                        error!("rustshyper: current task has no ThreadLocal for rustshyper ioctl");
                        return Err(Error::new(Errno::EFAULT));
                    }
                };
                let vmar = {
                    let vmar = thread_local.vmar().borrow();
                    match vmar.as_ref() {
                        Some(vmar) => vmar.clone(),
                        None => {
                            error!(
                                "rustshyper: current thread has no active VMAR for rustshyper ioctl"
                            );
                            return Err(Error::new(Errno::EFAULT));
                        }
                    }
                };

                let memory_size = usize::try_from(region.memory_size)?;
                let userspace_start = usize::try_from(region.userspace_addr)?;
                let guest_start = usize::try_from(region.guest_phys_addr)?;
                let userspace_end = userspace_start
                    .checked_add(memory_size)
                    .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
                let guest_end = guest_start
                    .checked_add(memory_size)
                    .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;

                let guest_space = self.vm.guest_mem();
                let mut userspace_addr = userspace_start;
                let mut guest_phys_addr = guest_start;
                guest_space.record_map(userspace_addr, guest_phys_addr);
                let prop = PageProperty::new_user(PageFlags::RWX, ostd::mm::CachePolicy::Writeback);
                while userspace_addr < userspace_end && guest_phys_addr < guest_end {
                    let frame = query_user_ram_frame(&vmar, userspace_addr)?;
                    let preempt_guard = ostd::task::disable_preempt();
                    let mut guest_cursor_mut = guest_space.cursor_mut(
                        &preempt_guard,
                        &(guest_phys_addr..guest_phys_addr + PAGE_SIZE),
                    )?;
                    guest_cursor_mut.map(frame, prop);
                    userspace_addr += PAGE_SIZE;
                    guest_phys_addr += PAGE_SIZE;
                }

                Ok(0)
            }
            cmd @ InjectIrq => {
                let irq = cmd.read()?;
                self.vm.inject_irq_line(irq as usize)?;
                Ok(0)
            }
            GetDirtyLog => {
                // TODO: Implement dirty log tracking
                return_errno_with_message!(Errno::ENOSYS, "GetDirtyLog not yet implemented");
            }
            _ => {
                return_errno_with_message!(Errno::ENOTTY, "unknown VM ioctl command");
            }
        })
    }

    fn path(&self) -> &Path {
        &self.pseudo_path
    }

    fn dump_proc_fdinfo(self: Arc<Self>, _fd_flags: FdFlags) -> Box<dyn core::fmt::Display> {
        Box::new(alloc::format!("vm_id: {}\n", self.vm.id))
    }
}

impl crate::process::signal::Pollable for VmFile {
    fn poll(
        &self,
        _mask: crate::events::IoEvents,
        _poller: Option<&mut crate::process::signal::PollHandle>,
    ) -> crate::events::IoEvents {
        // VMs don't support polling
        crate::events::IoEvents::empty()
    }
}
