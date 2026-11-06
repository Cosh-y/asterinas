// SPDX-License-Identifier: MPL-2.0

use ostd::{
    arch::vm::VcpuRegs,
    vm::Gpaddr,
};

use super::vcpu::VcpuState;
use crate::{kvm::vcpu::Vcpu, prelude::*};

pub(super) struct MmioOp {
    pub len: usize,
    pub access: MmioAccess,
    pub next_rip: usize,
}

pub(super) enum MmioAccess {
    Read {
        reg: Gpr,
    },
    Write {
        /// Only the first `len` bytes are transferred.
        data: [u8; 8],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Gpr {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl Gpr {
    pub(super) fn write(self, regs: &mut VcpuRegs, value: usize, len: usize) {
        let target = match self {
            Self::Rax => &mut regs.rax,
            Self::Rcx => &mut regs.rcx,
            Self::Rdx => &mut regs.rdx,
            Self::Rbx => &mut regs.rbx,
            Self::Rsp => &mut regs.rsp,
            Self::Rbp => &mut regs.rbp,
            Self::Rsi => &mut regs.rsi,
            Self::Rdi => &mut regs.rdi,
            Self::R8 => &mut regs.r8,
            Self::R9 => &mut regs.r9,
            Self::R10 => &mut regs.r10,
            Self::R11 => &mut regs.r11,
            Self::R12 => &mut regs.r12,
            Self::R13 => &mut regs.r13,
            Self::R14 => &mut regs.r14,
            Self::R15 => &mut regs.r15,
        };

        *target = match len {
            1 => (*target & !0xff) | (value & 0xff),
            2 => (*target & !0xffff) | (value & 0xffff),
            4 => value as u32 as usize,
            8 => value,
            _ => unreachable!(),
        };
    }
}

impl Vcpu {
    pub(super) fn decode_mmio_op(&self, state: &VcpuState) -> Result<MmioOp> {
        // TODO: 这个文件中的实现需要重新整理
        // 当前支持的指令
        // [REX] 89 /r [SIB] [disp]
        // [REX] 8b /r [SIB] [disp]
        // [REX] c7 /0 id [SIB] [disp]
        // REX 前缀，W 位表示操作数大小，R 位扩展 ModRM.reg, ...
        // REX 后紧邻的是 opcode
        // /r 和 /0 表示 ModRM，ModRM 描述两个操作数，一个字节，包含 mod、reg 和 rm 三个字段。
        // SIB 用于计算内存地址偏移，包含基地址寄存器、索引寄存器和比例因子。
        //     存在条件：mod != 0b11 && rm_field == 0b100
        // disp 同样用于内存地址偏移，是一个有符号的长度。
        //     存在条件： mod   disp
        //              00    32-bit
        //              01    8-bit
        //              10    32-bit

        const REX_W: u8 = 1 << 3;
        const REX_R: u8 = 1 << 2;

        const MOV_STORE: u8 = 0x89;
        const MOV_LOAD: u8 = 0x8b;
        const MOV_IMMEDIATE: u8 = 0xc7;

        let rip_gpa = self.translate_gva2gpa(state, state.guest_context.regs().rip)?;

        let regs = state.guest_context.regs();
        let mut offset = 0;

        let mut opcode = self.read_mmio_instruction_byte(state, rip_gpa, &mut offset)?;
        let rex = if (0x40..=0x4f).contains(&opcode) {
            let prefix = opcode;
            opcode = self.read_mmio_instruction_byte(state, rip_gpa, &mut offset)?;
            prefix
        } else {
            0
        };
        // 66、67、LOCK 等前缀也会在这里作为不支持的编码返回。
        // 66、67 是 REX 之前的前缀，REX 是紧邻操作码的前缀。
        if rex & REX_W != 0 || !matches!(opcode, MOV_STORE | MOV_LOAD | MOV_IMMEDIATE) {
            return_errno_with_message!(Errno::EOPNOTSUPP, "only 32-bit MMIO MOV is supported");
        }

        let modrm = self.read_mmio_instruction_byte(state, rip_gpa, &mut offset)?;
        let mode = modrm >> 6;
        let reg_field = (modrm >> 3) & 7;
        let rm_field = modrm & 7;
        if mode == 3 || (opcode == MOV_IMMEDIATE && reg_field != 0) {
            return_errno_with_message!(Errno::EOPNOTSUPP, "unsupported MMIO MOV operands");
        }

        // 不必重新计算设备地址，但要读过 SIB 和位移，才能定位立即数和下一条指令。
        // REX.B/X 扩展寻址寄存器，不影响这里的 SIB、位移长度判断。
        let has_disp32 = if rm_field == 4 {
            let sib = self.read_mmio_instruction_byte(state, rip_gpa, &mut offset)?;
            mode == 0 && sib & 7 == 5
        } else {
            mode == 0 && rm_field == 5
        };
        let displacement_len = match mode {
            0 if has_disp32 => 4,
            1 => 1,
            2 => 4,
            _ => 0,
        };
        for _ in 0..displacement_len {
            self.read_mmio_instruction_byte(state, rip_gpa, &mut offset)?;
        }

        // x86 的寄存器编码顺序与 VcpuRegs 的字段顺序不同，显式列出对应关系。
        let gprs = [
            (Gpr::Rax, regs.rax),
            (Gpr::Rcx, regs.rcx),
            (Gpr::Rdx, regs.rdx),
            (Gpr::Rbx, regs.rbx),
            (Gpr::Rsp, regs.rsp),
            (Gpr::Rbp, regs.rbp),
            (Gpr::Rsi, regs.rsi),
            (Gpr::Rdi, regs.rdi),
            (Gpr::R8, regs.r8),
            (Gpr::R9, regs.r9),
            (Gpr::R10, regs.r10),
            (Gpr::R11, regs.r11),
            (Gpr::R12, regs.r12),
            (Gpr::R13, regs.r13),
            (Gpr::R14, regs.r14),
            (Gpr::R15, regs.r15),
        ];
        let reg_index = usize::from(reg_field | ((rex & REX_R) << 1));
        let access = if opcode == MOV_LOAD {
            // 完成读操作时，写 r32 要将对应 64 位寄存器的高 32 位清零。
            MmioAccess::Read {
                reg: gprs[reg_index].0,
            }
        } else {
            let mut data = [0; 8];
            if opcode == MOV_IMMEDIATE {
                // C7 /0 的 reg 字段是操作码扩展，REX.R 在这里不表示源寄存器。
                for byte in &mut data[..4] {
                    *byte = self.read_mmio_instruction_byte(state, rip_gpa, &mut offset)?;
                }
            } else {
                data[..4].copy_from_slice(&(gprs[reg_index].1 as u32).to_le_bytes());
            }
            MmioAccess::Write { data }
        };

        // 普通 EPT violation 的 instruction_len 不保证有效，使用实际解码长度。
        // 调用方完成设备访问后再写回 RIP，解码期间不修改 guest 状态。
        Ok(MmioOp {
            len: 4,
            access,
            next_rip: regs
                .rip
                .checked_add(offset)
                .ok_or_else(|| Error::new(Errno::EOVERFLOW))?,
        })
    }

    fn read_mmio_instruction_byte(
        &self,
        state: &VcpuState,
        rip_gpa: Gpaddr,
        offset: &mut usize,
    ) -> Result<u8> {
        // 只读取解码需要的字节，避免一次预读 15 字节碰到未映射的下一页。
        // 跨 guest 虚拟页时重新翻译，不能假定下一页的 GPA 也连续。
        let first_page_remaining = PAGE_SIZE - (rip_gpa & (PAGE_SIZE - 1));
        let gpa = if *offset < first_page_remaining {
            rip_gpa + *offset
        } else {
            let gva = state
                .guest_context
                .regs()
                .rip
                .checked_add(*offset)
                .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
            self.translate_gva2gpa(state, gva)?
        };
        // TODO: VmMemory::read_val 找不到 slot 时仍会 unwrap，后续应返回访存错误。
        let byte = self.vm().memory().read_val::<u8>(gpa)?;
        *offset += 1;
        Ok(byte)
    }

    fn translate_gva2gpa(&self, state: &VcpuState, gva: usize) -> Result<Gpaddr> {
        const LARGE_PAGE: u64 = 1 << 7;
        const ENTRY_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;

        // TODO: 这里只支持四级分页，后续还需检查 Present、保留位及访问权限。
        let address_mask = ENTRY_ADDRESS_MASK;
        let mut table = state.guest_context.sregs().cr3 & address_mask;
        for shift in [39u32, 30, 21, 12] {
            let index = (gva >> shift) & 0x1ff;
            // TODO: 此处 table as _ 无法推断目标类型，接入时应明确写为 Gpaddr。
            let entry = self.vm().memory().read_val::<u64>(table as _ + index * 8)?;
            let large = entry & LARGE_PAGE != 0;
            if shift == 12 || large {
                let mask = (1 << shift) - 1;
                // TODO: 大页物理基址还应清除 mask 内的位，包括 bit 12 的 PAT。
                return Ok((entry & address_mask) as usize | (gva & mask));
            }
            table = entry & address_mask;
        }

        return_errno!(Errno::EINVAL);
    }
}
