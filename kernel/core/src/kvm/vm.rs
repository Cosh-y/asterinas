// SPDX-License-Identifier: MPL-2.0

use super::{
    arch::vm::VmArch,
    vcpu::Vcpu,
    vm_memory::VmMemory
};
use crate::prelude::*;

pub(crate) struct Vm {
    _id: u32,
    memory: VmMemory,
    vcpus: Mutex<BTreeMap<u32, Weak<Vcpu>>>,
    arch: VmArch,
}

impl Vm {
    pub(crate) fn new(id: u32) -> Result<Arc<Self>> {
        let memory = VmMemory::new()?;

        Ok(Arc::new_cyclic(move |vm| Self {
            _id: id,
            memory,
            vcpus: Mutex::new(BTreeMap::new()),
            arch: VmArch::new(vm.clone()),
        }))
    }

    pub(super) fn arch(&self) -> &VmArch {
        &self.arch
    }

    pub(crate) fn create_vcpu(self: &Arc<Self>, vcpu_id: u32) -> Result<Arc<Vcpu>> {
        let mut vcpus = self.vcpus.lock();

        if vcpus.get(&vcpu_id).and_then(Weak::upgrade).is_some() {
            return_errno_with_message!(Errno::EEXIST, "vCPU already exists");
        }

        let vcpu = Vcpu::new(vcpu_id, self)?;
        vcpus.insert(vcpu_id, Arc::downgrade(&vcpu));
        Ok(vcpu)
    }

    pub(crate) fn memory(&self) -> &VmMemory {
        &self.memory
    }

    pub(super) fn vcpus(&self) -> MutexGuard<'_, BTreeMap<u32, Weak<Vcpu>>> {
        self.vcpus.lock()
    }
}
