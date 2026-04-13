# RustShyper - Asterinas Hypervisor Interface

RustShyper is the Asterinas version of KVM, providing hypervisor functionality that allows userspace processes to create and manage virtual machines.

## Architecture

The RustShyper interface consists of three main file descriptor types:

1. **Device FD** (`/dev/rustshyper`) - Main hypervisor device
2. **VM FD** - Virtual machine instance
3. **VCPU FD** - Virtual CPU instance

## Usage Flow

### 1. Open the RustShyper Device

```c
int dev_fd = open("/dev/rustshyper", O_RDWR);
```

### 2. Check API Version

```c
int version = ioctl(dev_fd, RSH_GET_API_VERSION);
// Returns RUSTSHYPER_API_VERSION (currently 1)
```

### 3. Create a VM

```c
int vm_fd = ioctl(dev_fd, RSH_CREATE_VM);
// Returns a file descriptor for the new VM
```

### 4. Set VM Memory

```c
struct UserMemoryRegion region = {
    .slot = 0,
    .flags = 0,
    .guest_phys_addr = 0x0,
    .memory_size = 0x100000,  // 1MB
    .userspace_addr = (uint64_t)host_memory,
};
ioctl(vm_fd, RSH_SET_USER_MEMORY_REGION, &region);
```

### 5. Create VCPUs

```c
uint32_t vcpu_id = 0;
int vcpu_fd = ioctl(vm_fd, RSH_CREATE_VCPU, &vcpu_id);
// Returns a file descriptor for the new VCPU
```

### 6. Configure VCPU Registers

```c
struct VcpuRegs regs = {
    .rip = 0x1000,
    .rsp = 0x8000,
    // ... other registers
};
ioctl(vcpu_fd, RSH_SET_REGS, &regs);

struct VcpuSregs sregs = {
    .cr0 = 0x60000010,
    .cr3 = 0x0,
    .cr4 = 0x0,
    // ... other special registers
};
ioctl(vcpu_fd, RSH_SET_SREGS, &sregs);
```

### 7. Run the VCPU

```c
ioctl(vcpu_fd, RSH_RUN);
```

### 8. Read VCPU State

```c
struct VcpuRegs regs;
ioctl(vcpu_fd, RSH_GET_REGS, &regs);

struct VcpuSregs sregs;
ioctl(vcpu_fd, RSH_GET_SREGS, &sregs);
```

## IOCTL Commands

### Device IOCTLs

- `RSH_GET_API_VERSION` (0x00) - Get API version
- `RSH_CREATE_VM` (0x01) - Create a new VM, returns VM FD
- `RSH_CHECK_EXTENSION` (0x03) - Check if an extension is supported

### VM IOCTLs

- `RSH_CREATE_VCPU` (0x41) - Create a VCPU, returns VCPU FD
- `RSH_SET_USER_MEMORY_REGION` (0x46) - Map memory to the VM
- `RSH_GET_DIRTY_LOG` (0x42) - Get dirty page bitmap

### VCPU IOCTLs

- `RSH_RUN` (0x80) - Run the VCPU
- `RSH_GET_REGS` (0x81) - Get general purpose registers
- `RSH_SET_REGS` (0x82) - Set general purpose registers
- `RSH_GET_SREGS` (0x83) - Get special registers
- `RSH_SET_SREGS` (0x84) - Set special registers

## Data Structures

### UserMemoryRegion

```c
struct UserMemoryRegion {
    uint32_t slot;              // Memory slot number
    uint32_t flags;             // Flags (e.g., read-only)
    uint64_t guest_phys_addr;   // Guest physical address
    uint64_t memory_size;       // Size in bytes
    uint64_t userspace_addr;    // Host virtual address
};
```

### VcpuRegs

```c
struct VcpuRegs {
    uint64_t rax, rbx, rcx, rdx;
    uint64_t rsi, rdi, rsp, rbp;
    uint64_t r8, r9, r10, r11;
    uint64_t r12, r13, r14, r15;
    uint64_t rip, rflags;
};
```

### VcpuSregs

```c
struct VcpuSregs {
    uint64_t cr0, cr2, cr3, cr4, cr8;
    uint64_t efer;
};
```

## Implementation Status

Currently implemented:
- ✅ Device file descriptor with ioctl interface
- ✅ VM creation and file descriptor management
- ✅ VCPU creation and file descriptor management
- ✅ Memory region configuration
- ✅ Register get/set operations

To be implemented:
- ⏳ Actual VM execution (hardware virtualization)
- ⏳ VM exit handling
- ⏳ Interrupt injection
- ⏳ Dirty page tracking
- ⏳ Device emulation

## Notes

- All file descriptors (VM FD and VCPU FD) are automatically managed by the kernel
- When a VM FD is closed, all associated VCPUs are automatically cleaned up
- The interface is designed to be compatible with KVM-style userspace VMMs

