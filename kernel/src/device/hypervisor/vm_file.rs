// SPDX-License-Identifier: MPL-2.0

//! VM file descriptor implementation

use ostd::{
    mm::{CachePolicy, Gpaddr, PageFlags, PageProperty, VmIo, vm_space::VmQueriedItem},
    task::Task,
};

use super::{ioctl::*, vcpu_file::VcpuFile, vm::Vm};
use crate::{
    fs::{
        file::{AccessMode, FileLike, file_table::FdFlags},
        pseudofs::AnonInodeFs,
        vfs::path::Path,
    },
    prelude::*,
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

impl VmFile {
    /// Creates a new VM file
    pub fn new(vm: Arc<Vm>) -> Self {
        let pseudo_path = AnonInodeFs::new_path(|_| "anon_inode:[hypervisor-vm]".to_string());
        Self { vm, pseudo_path }
    }

    fn set_user_memory_region(&self, region: UserMemoryRegion) -> Result<()> {
        let memory_size = usize::try_from(region.memory_size)?;
        if region.flags & !KVM_MEM_READONLY != 0 {
            return_errno_with_message!(Errno::EINVAL, "unsupported guest memory flags");
        }
        if memory_size == 0 {
            self.vm.set_user_memory_region(
                region.slot,
                0,
                0,
                Vec::new(),
                default_guest_mem_prop(false),
            )?;
            return Ok(());
        }

        let vmar = current_vmar()?;
        let userspace_start = usize::try_from(region.userspace_addr)?;
        let guest_start = usize::try_from(region.guest_phys_addr)?;
        let userspace_end = userspace_start
            .checked_add(memory_size)
            .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
        validate_user_memory_region(userspace_start, guest_start, memory_size)?;

        let mut frames = Vec::new();
        let mut userspace_addr = userspace_start;
        while userspace_addr < userspace_end {
            frames.push(query_user_ram_frame(&vmar, userspace_addr)?);
            userspace_addr += PAGE_SIZE;
        }

        let prop = default_guest_mem_prop(region.flags & KVM_MEM_READONLY != 0);
        self.vm
            .set_user_memory_region(region.slot, guest_start, memory_size, frames, prop)?;

        Ok(())
    }
}

fn default_guest_mem_prop(is_readonly: bool) -> PageProperty {
    let guest_page_flags = if is_readonly {
        PageFlags::RX
    } else {
        PageFlags::RWX
    };
    PageProperty::new_user(guest_page_flags, CachePolicy::Writeback)
}

fn validate_user_memory_region(
    userspace_start: Vaddr,
    guest_start: Gpaddr,
    memory_size: usize,
) -> Result<()> {
    if !userspace_start.is_multiple_of(PAGE_SIZE)
        || !guest_start.is_multiple_of(PAGE_SIZE)
        || !memory_size.is_multiple_of(PAGE_SIZE)
    {
        return_errno_with_message!(Errno::EINVAL, "guest memory region must be page-aligned");
    }
    Ok(())
}

impl FileLike for VmFile {
    fn access_mode(&self) -> AccessMode {
        AccessMode::O_RDWR
    }

    fn read(&self, _writer: &mut VmWriter) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot read from VM file");
    }

    fn write(&self, _reader: &mut VmReader) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot write to VM file");
    }

    fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        dispatch_ioctl!(match raw_ioctl {
            CheckExtension => {
                Ok(check_extension(raw_ioctl.arg()))
            }
            CreateVcpu => {
                let vcpu_id = u32::try_from(raw_ioctl.arg())?;

                // Create the VCPU
                let vcpu = self.vm.create_vcpu(vcpu_id)?;

                // Create a file descriptor for the VCPU
                let vcpu_file = Arc::new(VcpuFile::new(self.vm.clone(), vcpu));

                // Insert into the current process's file table
                let current = Task::current().unwrap();
                let mut file_table = current.as_thread_local().unwrap().borrow_file_table_mut();
                let mut file_table_locked = file_table.unwrap().write();
                let vcpu_fd = file_table_locked.insert(vcpu_file, FdFlags::empty());

                Ok(vcpu_fd.into())
            }
            SetNrMmuPages => {
                Ok(0)
            }
            cmd @ SetUserMemoryRegion => {
                let region: UserMemoryRegion = cmd.read()?;
                self.set_user_memory_region(region)?;
                Ok(0)
            }
            SetTssAddr => {
                // TODO:
                Ok(0)
            }
            cmd @ SetIdentityMapAddr => {
                // KVM quirk api.
                Ok(0)
            }
            CreateIrqchip => {
                self.vm.create_irqchip()?;
                Ok(0)
            }
            cmd @ IrqLine => {
                let irq_level = cmd.read()?;
                self.vm.set_irq_line(irq_level)?;
                Ok(0)
            }
            cmd @ GetIrqchip => {
                let irqchip = cmd.read()?;
                match irqchip.chip_id {
                    KVM_IRQCHIP_PIC_MASTER | KVM_IRQCHIP_PIC_SLAVE | KVM_IRQCHIP_IOAPIC => {
                        let irqchip = IrqChip {
                            chip_id: irqchip.chip_id,
                            ..IrqChip::default()
                        };
                        cmd.write(&irqchip)?;
                        Ok(0)
                    }
                    _ => {
                        return_errno_with_message!(Errno::EINVAL, "unknown IRQ chip id");
                    }
                }
            }
            SetIrqchip => {
                let irqchip = read_irqchip(raw_ioctl.arg())?;
                match irqchip.chip_id {
                    KVM_IRQCHIP_PIC_MASTER | KVM_IRQCHIP_PIC_SLAVE | KVM_IRQCHIP_IOAPIC => Ok(0),
                    _ => {
                        return_errno_with_message!(Errno::EINVAL, "unknown IRQ chip id");
                    }
                }
            }
            cmd @ IrqLineStatus => {
                let mut irq_level = cmd.read()?;
                let delivered = self.vm.set_irq_line(irq_level)?;
                irq_level.irq = if delivered { 1 } else { 0 };
                cmd.write(&irq_level)?;
                Ok(0)
            }
            cmd @ RegisterCoalescedMmio => {
                let _zone = cmd.read()?;
                // TODO: Implement coalesced MMIO registration
                Ok(0)
            }
            cmd @ UnregisterCoalescedMmio => {
                let _zone = cmd.read()?;
                // TODO: Implement coalesced MMIO unregistration
                Ok(0)
            }
            cmd @ SetGsiRouting => {
                let routing = cmd.read()?;
                let entries = read_irq_routing_entries(routing, raw_ioctl.arg())?;
                self.vm.set_gsi_routing(&entries)?;
                Ok(0)
            }
            cmd @ IrqFd => {
                let _irqfd = cmd.read()?;
                // TODO: Implement IRQFD handling
                Ok(0)
            }
            cmd @ CreatePit2 => {
                let _pit_config = cmd.read()?;
                Ok(0)
            }
            cmd @ IoEventFd => {
                let _ioeventfd = cmd.read()?;
                // TODO: Implement IOEventFD handling
                Ok(0)
            }
            cmd @ SetClock => {
                let clock = cmd.read()?;
                self.vm.set_clock(clock);
                // TODO: Make it work with kvmclock.rs.
                Ok(0)
            }
            cmd @ GetClock => {
                let clock = self.vm.get_clock();
                cmd.write(&clock)?;
                Ok(0)
            }
            cmd @ SignalMsi => {
                let _msi = cmd.read()?;
                // TODO: Implement MSI signaling
                Ok(1)
            }
            cmd @ EnableCap => {
                let cap = cmd.read()?;
                self.vm.enable_cap(cap)?;
                Ok(0)
            }
            GetStatsFd => {
                return_errno_with_message!(Errno::ENOTTY, "KVM stats fd is not supported");
            }
            _ => {
                let ioctl_nr = raw_ioctl.cmd() & 0xff;
                error!(
                    "hypervisor: unimplemented VM ioctl command: cmd={:#x}, nr={:#x}",
                    raw_ioctl.cmd(),
                    ioctl_nr
                );
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

fn read_irq_routing_entries(routing: IrqRouting, arg: usize) -> Result<Vec<IrqRoutingEntry>> {
    let nr = usize::try_from(routing.nr)?;
    if nr > KVM_MAX_IRQ_ROUTES {
        return_errno_with_message!(Errno::E2BIG, "too many GSI routing entries");
    }

    let entries_addr = arg
        .checked_add(size_of::<IrqRouting>())
        .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
    let entries_len = nr
        .checked_mul(size_of::<IrqRoutingEntry>())
        .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
    let current = Task::current().unwrap();
    let thread_local = current.as_thread_local().unwrap();
    let user_space = CurrentUserSpace::new(thread_local);
    let mut reader = user_space.reader(entries_addr, entries_len)?;
    let mut entries = Vec::new();
    for _ in 0..nr {
        entries.push(reader.read_val()?);
    }

    Ok(entries)
}

fn read_irqchip(arg: usize) -> Result<IrqChip> {
    let current = Task::current().unwrap();
    let thread_local = current.as_thread_local().unwrap();
    let user_space = CurrentUserSpace::new(thread_local);
    Ok(user_space.read_val(arg)?)
}

fn current_vmar() -> Result<Arc<Vmar>> {
    let current = match Task::current() {
        Some(current) => current,
        None => {
            error!("hypervisor: no current task found for hypervisor ioctl");
            return Err(Error::new(Errno::ESRCH));
        }
    };
    let thread_local = match current.as_thread_local() {
        Some(thread_local) => thread_local,
        None => {
            error!("hypervisor: current task has no ThreadLocal for hypervisor ioctl");
            return Err(Error::new(Errno::EFAULT));
        }
    };
    let vmar = thread_local.vmar().borrow();
    match vmar.as_ref() {
        Some(vmar) => Ok(vmar.clone_arc()),
        None => {
            error!("hypervisor: current thread has no active VMAR for hypervisor ioctl");
            Err(Error::new(Errno::EFAULT))
        }
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

        vmar.handle_page_fault(&PageFaultInfo::new(userspace_addr, PageFlags::R.into()))?;
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
