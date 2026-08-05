// SPDX-License-Identifier: MPL-2.0

use spin::Once;
use x86_64::registers::control::Cr2;

use super::{vmx::Msr, x86::write_cr2_raw};
use crate::{
    arch::cpu::context::FpuContext,
    cpu::{PinCurrentCpu, all_cpus},
    cpu_local,
    sync::SpinLock,
};

cpu_local! {
    static HOST_CONTEXT: Once<SpinLock<HostContext>> = Once::new();
}

pub(super) fn init_host_contexts() {
    for cpu in all_cpus() {
        HOST_CONTEXT
            .get_on_cpu(cpu)
            .call_once(|| SpinLock::new(HostContext::new()));
    }
}

pub(super) fn with_host_context<R>(
    pin_guard: &dyn PinCurrentCpu,
    f: impl FnOnce(&mut HostContext) -> R,
) -> R {
    let host_context = HOST_CONTEXT
        .get_on_cpu(pin_guard.current_cpu())
        .get()
        .expect("host contexts must be initialized before guest execution");
    f(&mut host_context.lock())
}

#[derive(Debug)]
/// Host contexts that cannot be automatically loaded or saved by VMCS.
pub(super) struct HostContext {
    msrs: HostRunMsrs,
    fpu: FpuContext,
    cr2: u64,
}

impl HostContext {
    fn new() -> Self {
        Self {
            msrs: HostRunMsrs::default(),
            fpu: FpuContext::new(),
            cr2: 0,
        }
    }

    pub(super) fn save(&mut self) {
        self.fpu.save();
        self.msrs = HostRunMsrs::read_current();
        self.cr2 = Cr2::read_raw();
    }

    pub(super) fn load(&self) {
        write_cr2_raw(self.cr2);
        self.fpu.load();
        self.msrs.restore();
    }

    pub(super) fn load_after_vmexit(&self) {
        // VM-exit has already restored the five syscall-related host MSRs
        // from the VM-exit MSR-load area. CR2 and FPU state are not managed by
        // VMCS and remain OSTD's responsibility.
        write_cr2_raw(self.cr2);
        self.fpu.load();
    }

    pub(super) fn run_msr_values(&self) -> [u64; 5] {
        self.msrs.values()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct HostRunMsrs {
    star: u64,
    lstar: u64,
    cstar: u64,
    syscall_mask: u64,
    kernel_gs_base: u64,
}

impl HostRunMsrs {
    fn read_current() -> Self {
        Self {
            star: Msr::IA32_STAR.read(),
            lstar: Msr::IA32_LSTAR.read(),
            cstar: Msr::IA32_CSTAR.read(),
            syscall_mask: Msr::IA32_FMASK.read(),
            kernel_gs_base: Msr::IA32_KERNEL_GSBASE.read(),
        }
    }

    fn restore(self) {
        Msr::IA32_STAR.write(self.star);
        Msr::IA32_LSTAR.write(self.lstar);
        Msr::IA32_CSTAR.write(self.cstar);
        Msr::IA32_FMASK.write(self.syscall_mask);
        Msr::IA32_KERNEL_GSBASE.write(self.kernel_gs_base);
    }

    fn values(self) -> [u64; 5] {
        [
            self.star,
            self.lstar,
            self.cstar,
            self.syscall_mask,
            self.kernel_gs_base,
        ]
    }
}
