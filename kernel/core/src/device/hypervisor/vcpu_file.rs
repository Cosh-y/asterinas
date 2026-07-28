// SPDX-License-Identifier: MPL-2.0

//! VCPU file descriptor implementation

use ostd::{
    arch::vm::{GuestExitInfo, VmxExitReason},
    task::Task,
    vm::GuestRunResult,
};

pub(super) use super::vcpu::Vcpu;
use super::{
    apic::emulate_apic_mmio,
    cpuid, cr,
    ioctl::*,
    ioeventfd::IoEventAddressSpace,
    mmio::{MmioDirection, decode_current_mmio_instruction},
    msr::{self, MsrAccess},
    pio::{PioDirection, PioOperation},
    vcpu::{PendingMmioOperation, PendingOperation, PendingPioOperation},
    vm::Vm,
};
use crate::{
    fs::{
        file::{AccessMode, FileCommon, FileLike, Mappable, StatusFlags, file_table::FdFlags},
        pseudofs::AnonInodeFs,
    },
    prelude::*,
    util::ioctl::{RawIoctl, dispatch_ioctl},
};

/// VCPU file descriptor
pub struct VcpuFile {
    vm: Arc<Vm>,
    vcpu: Arc<Vcpu>,
    common: FileCommon,
    compat_state: Mutex<VcpuCompatState>,
}

// Compatibility state for KVM ioctls that are accepted but not wired into
// guest execution yet. QEMU copies these GET results back into CPUX86State,
// so keep the last SET value instead of returning a fresh default state.
struct VcpuCompatState {
    debug_regs: DebugRegs,
    vcpu_events: VcpuEvents,
    xsave: XsaveState,
}

impl Default for VcpuCompatState {
    fn default() -> Self {
        Self {
            debug_regs: default_debug_regs(),
            vcpu_events: VcpuEvents::default(),
            xsave: XsaveState::default(),
        }
    }
}

impl VcpuFile {
    /// Creates a new VCPU file
    pub fn new(vm: Arc<Vm>, vcpu: Arc<Vcpu>) -> Self {
        let pseudo_path = AnonInodeFs::new_path(|_| "anon_inode:[hypervisor-vcpu]".to_string());
        Self {
            vm,
            vcpu,
            common: FileCommon::new(pseudo_path, StatusFlags::empty()),
            compat_state: Mutex::new(VcpuCompatState::default()),
        }
    }
}

impl FileLike for VcpuFile {
    fn access_mode(&self) -> AccessMode {
        AccessMode::O_RDWR
    }

    fn read(&self, _writer: &mut VmWriter) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot read from VCPU file");
    }

    fn write(&self, _reader: &mut VmReader) -> Result<usize> {
        return_errno_with_message!(Errno::EINVAL, "cannot write to VCPU file");
    }

    fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        dispatch_ioctl!(match raw_ioctl {
            Run => {
                self.ioctl_run()
            }
            cmd @ GetRegs => {
                let regs = self.vcpu.get_regs()?;
                cmd.write(&regs)?;
                Ok(0)
            }
            cmd @ SetRegs => {
                let regs = cmd.read()?;
                self.vcpu.set_regs(regs)?;
                Ok(0)
            }
            cmd @ GetSregs => {
                let sregs = self.vcpu.get_sregs()?;
                cmd.write(&sregs)?;
                Ok(0)
            }
            cmd @ SetSregs => {
                let sregs = cmd.read()?;
                self.vcpu.set_sregs(sregs)?;
                Ok(0)
            }
            cmd @ GetMsrs => {
                let (msrs, mut entries) = read_get_msr_entries(&cmd, raw_ioctl)?;
                let handled_count = self.vcpu.get_msrs(&mut entries)?;
                write_get_msr_entries(&cmd, raw_ioctl, msrs, &entries)?;
                Ok(handled_count)
            }
            cmd @ SetMsrs => {
                let entries = read_set_msr_entries(&cmd, raw_ioctl)?;
                self.vcpu.set_msrs(&entries)
            }
            cmd @ SetFpu => {
                let _fpu = cmd.read()?;
                // No-op compatibility API; FPU/XMM state is not installed yet.
                Ok(0)
            }
            cmd @ GetLapic => {
                let lapic = self.vcpu.get_lapic()?;
                cmd.write(&lapic)?;
                Ok(0)
            }
            cmd @ SetLapic => {
                let lapic = cmd.read()?;
                self.vcpu.set_lapic(&lapic)?;
                Ok(0)
            }
            cmd @ SetCpuid2 => {
                let entries = read_cpuid_entries(&cmd, raw_ioctl)?;
                self.vcpu.set_cpuid_entries(entries)?;
                Ok(0)
            }
            cmd @ TprAccessReporting => {
                let ctl = cmd.read()?;
                // No-op compatibility API; report the accepted control back.
                cmd.write(&ctl)?;
                Ok(0)
            }
            cmd @ SetVapicAddr => {
                let _addr = cmd.read()?;
                // No-op compatibility API; VAPIC access-page acceleration is not modeled yet.
                Ok(0)
            }
            cmd @ GetMpState => {
                let state = self.vcpu.get_mp_state()?;
                cmd.write(&state)?;
                Ok(0)
            }
            cmd @ SetMpState => {
                let state = cmd.read()?;
                self.vcpu.set_mp_state(state)?;
                Ok(0)
            }
            cmd @ X86SetupMce => {
                let _mcg_cap = cmd.read()?;
                // No-op compatibility API; machine-check state is not modeled yet.
                Ok(0)
            }
            cmd @ GetVcpuEvents => {
                // No-op compatibility API; return the last accepted value.
                let events = self.compat_state.lock().vcpu_events;
                cmd.write(&events)?;
                Ok(0)
            }
            cmd @ SetVcpuEvents => {
                let events = cmd.read()?;
                self.compat_state.lock().vcpu_events = events;
                Ok(0)
            }
            cmd @ GetDebugRegs => {
                // No-op compatibility API; return the last accepted value.
                let debug_regs = self.compat_state.lock().debug_regs;
                cmd.write(&debug_regs)?;
                Ok(0)
            }
            cmd @ SetDebugRegs => {
                let debug_regs = cmd.read()?;
                self.compat_state.lock().debug_regs = debug_regs;
                Ok(0)
            }
            SetTscKhz => {
                self.vcpu.set_tsc_khz(read_tsc_khz(raw_ioctl)?)?;
                Ok(0)
            }
            GetTscKhz => {
                self.vcpu.get_tsc_khz()
            }
            cmd @ GetXsave => {
                // No-op compatibility API; return the last accepted value.
                let xsave = self.compat_state.lock().xsave;
                cmd.write(&xsave)?;
                Ok(0)
            }
            cmd @ SetXsave => {
                let xsave = cmd.read()?;
                self.compat_state.lock().xsave = xsave;
                Ok(0)
            }
            GetStatsFd => {
                return_errno_with_message!(Errno::ENOTTY, "KVM stats fd is not supported");
            }
            _ => {
                let ioctl_nr = raw_ioctl.cmd() & 0xff;
                error!(
                    "hypervisor: unimplemented VCPU ioctl command: cmd={:#x}, nr={:#x}",
                    raw_ioctl.cmd(),
                    ioctl_nr
                );
                return_errno_with_message!(Errno::ENOTTY, "unknown VCPU ioctl command");
            }
        })
    }

    fn common(&self) -> &FileCommon {
        &self.common
    }

    fn mappable(&self) -> Result<Mappable> {
        Ok(Mappable::Vmo(self.vcpu.run_page()))
    }

    fn dump_proc_fdinfo(self: Arc<Self>, _fd_flags: FdFlags) -> Box<dyn core::fmt::Display> {
        Box::new("hypervisor_vcpu\n")
    }
}

fn default_debug_regs() -> DebugRegs {
    DebugRegs {
        dr6: 0xffff0ff0,
        dr7: 0x400,
        ..DebugRegs::default()
    }
}

impl VcpuFile {
    fn ioctl_run(&self) -> Result<i32> {
        self.complete_pending_operation()?;
        if self.immediate_exit()? {
            return_errno_with_message!(Errno::EINTR, "KVM_RUN interrupted by immediate_exit");
        }
        self.vcpu.clear_run_output()?;
        let mut guest_mode = self.vcpu.guest_mode();

        loop {
            let run_result = {
                let mut context = self.vcpu.guest_context();
                match guest_mode.execute(
                    &mut context,
                    self.vm.memory().guest_mem(),
                    &self.vcpu.lapic,
                    &self.vcpu.lapic,
                ) {
                    Ok(exit_info) => exit_info,
                    Err(err) => {
                        error!("hypervisor: GuestMode::execute failed: {:?}", err);
                        return Err(err.into());
                    }
                }
            };
            let GuestRunResult::VmExit(exit_info) = run_result else {
                loop {
                    if self.vcpu.wait_for_sipi_wakeup() {
                        break;
                    }

                    // TODO: Use a more efficient wait mechanism instead of busy-waiting.
                    Task::yield_now();
                }
                continue;
            };
            let exit_info = match VmxExitReason::try_from(exit_info.exit_reason) {
                Ok(VmxExitReason::IO_INSTRUCTION) => {
                    if self.handle_pio_ioeventfd(&exit_info)? {
                        None
                    } else {
                        Some(exit_info)
                    }
                }
                Ok(VmxExitReason::EPT_VIOLATION) => {
                    let handled =
                        emulate_apic_mmio(self.vcpu.clone(), exit_info.guest_phys_addr as u64)
                            .inspect_err(|err| {
                                error!(
                                    "hypervisor: APIC MMIO handling failed: reason={:#x}, len={}, \
                             rip={:#x}, gpa={:#x}, qualification={:#x}, err={:?}",
                                    exit_info.exit_reason,
                                    exit_info.instruction_len,
                                    exit_info.guest_rip,
                                    exit_info.guest_phys_addr,
                                    exit_info.exit_qualification,
                                    err
                                );
                            })?;
                    if handled || self.handle_mmio_ioeventfd(&exit_info)? {
                        None
                    } else {
                        Some(exit_info)
                    }
                }
                Ok(VmxExitReason::PREEMPTION_TIMER) => None,
                Ok(VmxExitReason::CPUID) => {
                    cpuid::emulate_cpuid(&self.vcpu, &exit_info)?;
                    None
                }
                Ok(VmxExitReason::CR_ACCESS) => {
                    cr::emulate_cr_access(&self.vcpu, &exit_info)?;
                    None
                }
                Ok(VmxExitReason::HLT) => {
                    self.vcpu
                        .guest_context()
                        .advance_rip(exit_info.instruction_len as _);
                    loop {
                        if self.vcpu.wait_for_hlt_wakeup() {
                            break None;
                        }

                        // TODO: Use a more efficient wait mechanism instead of busy-waiting.
                        Task::yield_now();
                    }
                }
                Ok(VmxExitReason::PAUSE_INSTRUCTION) => None,
                Ok(VmxExitReason::MSR_READ) => {
                    msr::emulate_msr(&self.vcpu, &exit_info, MsrAccess::Read)?;
                    None
                }
                Ok(VmxExitReason::MSR_WRITE) => {
                    msr::emulate_msr(&self.vcpu, &exit_info, MsrAccess::Write)?;
                    None
                }
                Ok(_) => Some(exit_info),
                Err(_) => Some(exit_info),
            };

            if let Some(exit_info) = exit_info {
                self.write_exit_to_run_page(exit_info)?;
                return Ok(0);
            }
        }
    }

    fn complete_pending_operation(&self) -> Result<()> {
        let Some(operation) = self.vcpu.take_pending_operation() else {
            return Ok(());
        };

        if let Err(err) = self.complete_operation(operation) {
            self.vcpu.set_pending_operation(operation);
            return Err(err);
        }
        Ok(())
    }

    fn complete_operation(&self, operation: PendingOperation) -> Result<()> {
        match operation {
            PendingOperation::Pio(pio) => self.complete_pio_operation(pio),
            PendingOperation::Mmio(mmio) => self.complete_mmio_operation(mmio),
        }
    }

    fn complete_pio_operation(&self, operation: PendingPioOperation) -> Result<()> {
        let input_data = if operation.operation.direction() == PioDirection::In {
            let data_len = usize::try_from(operation.count)?
                .checked_mul(usize::from(operation.operation.size()))
                .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
            let mut bytes = vec![0_u8; data_len];
            self.vcpu
                .read_run_bytes(KVM_RUN_IO_DATA_OFFSET, &mut bytes)?;
            Some(bytes)
        } else {
            None
        };

        operation.operation.complete(
            &mut self.vcpu.guest_context(),
            self.vm.memory(),
            operation.count,
            input_data.as_deref(),
        )
    }

    fn complete_mmio_operation(&self, operation: PendingMmioOperation) -> Result<()> {
        let instruction = operation.instruction;
        let read_value = if instruction.direction() == MmioDirection::Read {
            let size = instruction.size() as usize;
            let mut bytes = [0_u8; size_of::<u64>()];
            self.vcpu
                .read_run_bytes(KVM_RUN_MMIO_DATA_OFFSET, &mut bytes[..size])?;
            Some(u64::from_le_bytes(bytes))
        } else {
            None
        };

        let mut context = self.vcpu.guest_context();
        if let Some(value) = read_value {
            instruction.complete_read(&mut context, value)?;
        }

        context.advance_rip(u64::from(instruction.len()));
        Ok(())
    }

    fn immediate_exit(&self) -> Result<bool> {
        let immediate_exit = self
            .vcpu
            .read_run_val::<u8>(KVM_RUN_IMMEDIATE_EXIT_OFFSET)?;
        Ok(immediate_exit != 0)
    }

    /// Handles a PIO exit by checking for an ioeventfd and signaling it if present.
    ///
    /// Returns `Ok(true)` if the PIO exit was handled by signaling an ioeventfd,
    ///         `Ok(false)` otherwise.
    fn handle_pio_ioeventfd(&self, exit_info: &GuestExitInfo) -> Result<bool> {
        let context = self.vcpu.guest_context();
        let Some(operation) = PioOperation::decode(&context, self.vm.memory(), exit_info)?
        else {
            return Ok(false);
        };
        let count = operation.batch_count(&context, KVM_RUN_IO_DATA_CAPACITY);
        if count == 0 {
            drop(context);
            let input_data = (operation.direction() == PioDirection::In).then_some(&[] as &[u8]);
            operation.complete(
                &mut self.vcpu.guest_context(),
                self.vm.memory(),
                0,
                input_data,
            )?;
            return Ok(true);
        }

        // Pio ioeventfd is only supported for output operations.
        if operation.direction() != PioDirection::Out {
            return Ok(false);
        }
        if !self.vm.has_ioeventfd(
            IoEventAddressSpace::Pio,
            u64::from(operation.port()),
            u32::from(operation.size()),
        ) {
            return Ok(false);
        }
        let data = operation.output_data(&context, self.vm.memory(), count)?;
        let values = operation.output_values(&data)?;
        drop(context);

        if !self.vm.signal_ioeventfd_batch(
            IoEventAddressSpace::Pio,
            u64::from(operation.port()),
            u32::from(operation.size()),
            &values,
        ) {
            return Ok(false);
        }

        operation.complete(
            &mut self.vcpu.guest_context(),
            self.vm.memory(),
            count,
            None,
        )?;
        Ok(true)
    }

    fn handle_mmio_ioeventfd(&self, exit_info: &GuestExitInfo) -> Result<bool> {
        if exit_info.exit_qualification & 0b111 != 0b010 {
            return Ok(false);
        }
        let context = self.vcpu.guest_context();
        let instruction = decode_current_mmio_instruction(&context, self.vm.memory())?;
        let Some(instruction) = instruction else {
            return Ok(false);
        };
        if instruction.direction() != MmioDirection::Write {
            return Ok(false);
        }
        let Some(value) = instruction.write_value(&context) else {
            return Ok(false);
        };
        drop(context);
        if !self.vm.signal_ioeventfd(
            IoEventAddressSpace::Mmio,
            exit_info.guest_phys_addr as u64,
            u32::from(instruction.size()),
            value,
        ) {
            return Ok(false);
        }

        self.vcpu
            .guest_context()
            .advance_rip(u64::from(instruction.len()));
        Ok(true)
    }

    fn write_exit_to_run_page(&self, exit_info: GuestExitInfo) -> Result<()> {
        self.vcpu.clear_run_output()?;
        self.write_common_run_state()?;

        match VmxExitReason::try_from(exit_info.exit_reason) {
            Ok(VmxExitReason::IO_INSTRUCTION) => self.write_io_exit(exit_info),
            Ok(VmxExitReason::EPT_VIOLATION) => self.write_mmio_exit(exit_info),
            Ok(VmxExitReason::HLT) => self.write_simple_exit(KVM_EXIT_HLT),
            Ok(VmxExitReason::TRIPLE_FAULT) => self.write_simple_exit(KVM_EXIT_SHUTDOWN),
            _ => self.write_internal_error_exit(exit_info),
        }
    }

    fn write_common_run_state(&self) -> Result<()> {
        let sregs = self.vcpu.get_sregs()?;
        self.vcpu
            .write_run_val(KVM_RUN_READY_FOR_INTERRUPT_INJECTION_OFFSET, &0_u8)?;
        self.vcpu.write_run_val(KVM_RUN_IF_FLAG_OFFSET, &0_u8)?;
        self.vcpu.write_run_val(KVM_RUN_FLAGS_OFFSET, &0_u16)?;
        self.vcpu.write_run_val(KVM_RUN_CR8_OFFSET, &sregs.cr8)?;
        self.vcpu
            .write_run_val(KVM_RUN_APIC_BASE_OFFSET, &sregs.apic_base)?;
        Ok(())
    }

    fn write_simple_exit(&self, exit_reason: u32) -> Result<()> {
        self.vcpu
            .write_run_val(KVM_RUN_EXIT_REASON_OFFSET, &exit_reason)
    }

    fn write_io_exit(&self, exit_info: GuestExitInfo) -> Result<()> {
        let (operation, count, data) = {
            let context = self.vcpu.guest_context();
            let Some(operation) =
                PioOperation::decode(&context, self.vm.memory(), &exit_info)?
            else {
                return self.write_internal_error_exit(exit_info);
            };
            let count = operation.batch_count(&context, KVM_RUN_IO_DATA_CAPACITY);
            if count == 0 {
                return self.write_internal_error_exit(exit_info);
            }
            let data = if operation.direction() == PioDirection::Out {
                operation.output_data(&context, self.vm.memory(), count)?
            } else {
                let data_len = usize::try_from(count)?
                    .checked_mul(usize::from(operation.size()))
                    .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
                vec![0_u8; data_len]
            };
            (operation, count, data)
        };
        let kvm_direction = match operation.direction() {
            PioDirection::In => KVM_EXIT_IO_IN,
            PioDirection::Out => KVM_EXIT_IO_OUT,
        };
        let size = operation.size();
        let port = operation.port();
        let data_offset = KVM_RUN_IO_DATA_OFFSET as u64;

        self.write_simple_exit(KVM_EXIT_IO)?;
        self.vcpu
            .write_run_val(KVM_RUN_IO_DIRECTION_OFFSET, &kvm_direction)?;
        self.vcpu.write_run_val(KVM_RUN_IO_SIZE_OFFSET, &size)?;
        self.vcpu.write_run_val(KVM_RUN_IO_PORT_OFFSET, &port)?;
        self.vcpu.write_run_val(KVM_RUN_IO_COUNT_OFFSET, &count)?;
        self.vcpu
            .write_run_val(KVM_RUN_IO_DATA_OFFSET_OFFSET, &data_offset)?;

        self.vcpu.write_run_bytes(KVM_RUN_IO_DATA_OFFSET, &data)?;

        self.vcpu
            .set_pending_operation(PendingOperation::Pio(PendingPioOperation {
                operation,
                count,
            }));
        Ok(())
    }

    fn write_mmio_exit(&self, exit_info: GuestExitInfo) -> Result<()> {
        let direction = match exit_info.exit_qualification & 0b111 {
            0b001 => MmioDirection::Read,
            0b010 => MmioDirection::Write,
            _ => return self.write_internal_error_exit(exit_info),
        };
        let context = self.vcpu.guest_context();
        let instruction = decode_current_mmio_instruction(&context, self.vm.memory())?;
        let Some(instruction) = instruction else {
            return self.write_internal_error_exit(exit_info);
        };
        if instruction.direction() != direction {
            return self.write_internal_error_exit(exit_info);
        }

        let size = instruction.size();
        let len = u32::from(size);
        let is_write = u8::from(direction == MmioDirection::Write);
        let mut data = [0_u8; 8];
        if direction == MmioDirection::Write {
            let Some(value) = instruction.write_value(&context) else {
                return self.write_internal_error_exit(exit_info);
            };
            data[..size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
        }
        drop(context);

        self.write_simple_exit(KVM_EXIT_MMIO)?;
        self.vcpu.write_run_val(
            KVM_RUN_MMIO_PHYS_ADDR_OFFSET,
            &(exit_info.guest_phys_addr as u64),
        )?;
        self.vcpu.write_run_bytes(KVM_RUN_MMIO_DATA_OFFSET, &data)?;
        self.vcpu.write_run_val(KVM_RUN_MMIO_LEN_OFFSET, &len)?;
        self.vcpu
            .write_run_val(KVM_RUN_MMIO_IS_WRITE_OFFSET, &is_write)?;

        self.vcpu
            .set_pending_operation(PendingOperation::Mmio(PendingMmioOperation { instruction }));
        Ok(())
    }

    fn write_internal_error_exit(&self, exit_info: GuestExitInfo) -> Result<()> {
        warn!(
            "hypervisor: unsupported VM exit for KVM_RUN: reason={:#x}, len={}, rip={:#x}, \
             gpa={:#x}, qualification={:#x}",
            exit_info.exit_reason,
            exit_info.instruction_len,
            exit_info.guest_rip,
            exit_info.guest_phys_addr,
            exit_info.exit_qualification,
        );
        self.write_simple_exit(KVM_EXIT_INTERNAL_ERROR)
    }
}

impl crate::process::signal::Pollable for VcpuFile {
    fn poll(
        &self,
        _mask: crate::events::IoEvents,
        _poller: Option<&mut crate::process::signal::PollHandle>,
    ) -> crate::events::IoEvents {
        // VCPUs don't support polling
        crate::events::IoEvents::empty()
    }
}
