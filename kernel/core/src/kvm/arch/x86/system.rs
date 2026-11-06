use ostd::arch::cpu::cpuid;

use super::{
    cpuid::supported_cpuid,
    ioctl::{
        GetSupportedCpuid, KVM_CPUID_FLAG_SIGNIFCANT_INDEX, KvmCpuid2, KvmCpuidEntry2,
        write_trailing_array,
    }
};
use crate::{dispatch_ioctl, prelude::*, util::ioctl::RawIoctl};

pub(crate) fn ioctl(raw_ioctl: RawIoctl) -> Result<i32> {
    dispatch_ioctl!(match raw_ioctl {
        cmd @ GetSupportedCpuid => {
            let mut header = cmd.read()?;
            let entries = supported_cpuid();

            if (header.nent as usize) < entries.len() {
                return_errno!(Errno::E2BIG);
            }

            write_trailing_array(raw_ioctl, size_of::<KvmCpuid2>(), &entries)?;
            header.nent = entries.len() as u32;
            cmd.write(&header)?;
            Ok(0)
        }
        _ => return_errno_with_message!(Errno::ENOTTY, "unsupported KVM ioctl"),
    })
}
