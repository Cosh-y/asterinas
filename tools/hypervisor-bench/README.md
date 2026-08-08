# Asterinas Hypervisor 部署与社区测试指南

本文面向希望体验、验证或复现 Asterinas Hypervisor 的社区用户。最快的验证方式是：
启动带 Hypervisor 的 Asterinas NixOS，在其中通过 `/dev/kvm` 运行
`kvm-hello-world` 和一个使用 `kvmtool` 启动的 Linux guest。

> 当前实现仍处于开发阶段，只应用于测试和研究，不要用于生产环境或运行不可信
> guest。

## 当前支持范围

| 项目 | 当前状态 |
| --- | --- |
| Host/guest 架构 | x86-64 |
| 硬件虚拟化 | Intel VMX（VT-x） |
| 用户态接口 | KVM-compatible `/dev/kvm` |
| 已验证 VMM | kvmtool、QEMU 9.1.50 |
| 已验证 guest | Linux 6.18.5 `bzImage` + BusyBox initramfs |
| vCPU | 单 vCPU、多 vCPU；实测到 16 vCPU |
| 并发 VM | 已验证两个单 vCPU VM 并发运行 |
| 已验证 I/O | 串口、定时器、virtio-blk 基础路径 |

AMD-V、非 x86-64 架构以及生产级隔离目前不在这份测试指南的支持范围内。

## 下载

### 社区测试包

- 文件名：`hypervisor-community-bundle-20260809-hypervisor-dev.tar.gz`
- 下载地址：https://pan.baidu.com/s/1xrUi0a1wQrkMnjArypk2ug?pwd=wvr7
- 提取码：wvr7
- 文件 SHA256：hypervisor-community-bundle-20260809-hypervisor-dev.tar.gz.sha256

测试包的目录结构如下：

```text
hypervisor-community-bundle-<版本>/
├── README.md
├── SHA256SUMS
├── SOURCE-REVISIONS
├── artifacts/
│   ├── bzImage
│   └── initramfs.cpio.gz
├── bin/
│   ├── kvm-hello-world
│   └── lkvm-static
├── docs/
├── licenses/
└── scripts/
```

其中 `initramfs.cpio.gz` 已包含交互式 `/init`、功能测试 init、`clockcheck` 和
`vmbench`。普通启动和功能测试只需要这个基础测试包。

### 可选的大文件

以下文件建议在网盘中单独提供，不要放进基础测试包：

| 文件 | 用途 | 是否必需 |
| --- | --- | --- |
| `asterinas.img.zst` | 免编译启动 Asterinas NixOS host | 可选 |
| `qemu-9.1.50-asterinas-bundle.tar.zst` | 运行完整 QEMU 功能/性能测试 | 可选 |
| `lmbench-initramfs.cpio.gz` | 运行 lmbench | 可选 |
| `l2-virtio.raw.zst` | 文件后端 virtio-blk 测试盘 | 可选 |

如果发布预构建的 `asterinas.img`，请同时写明它对应的 Asterinas commit，并在每次
替换镜像时更新 SHA256。不要直接上传未压缩的 8 GiB raw image。

## 1. 准备宿主机

推荐使用带 Intel VT-x 的 x86-64 Linux。需要安装：

- Git、Docker、QEMU；
- `tar`、`gzip`、`sha256sum`；
- `losetup`、`mount` 和 `sudo`，用于向 Asterinas raw image 注入测试程序。

先检查 KVM 和 nested VMX：

```bash
test -c /dev/kvm && echo "/dev/kvm: OK"
grep -m1 -o vmx /proc/cpuinfo
cat /sys/module/kvm_intel/parameters/nested
```

最后一条应输出 `Y` 或 `1`。如果没有启用，可先执行：

```bash
sudo modprobe -r kvm_intel
sudo modprobe kvm_intel nested=1
```

这会暂时卸载并重新加载 `kvm_intel`，请先关闭正在使用 KVM 的虚拟机。若宿主本身
也是虚拟机，还必须由更外层 hypervisor 向它暴露 VMX。

## 2. 获取源代码和测试包

当前社区测试基线使用 `hypervisor-dev` 分支：

```bash
git clone --branch hypervisor-dev https://github.com/Cosh-y/asterinas.git
cd asterinas
git checkout 864324f75e333559f432e125a3913b81ae3d02e0
cd ..
```

下载测试包后解压并校验。下文用环境变量保存解压后的绝对路径：

```bash
tar -xzf hypervisor-community-bundle-<版本>.tar.gz
export HYPERVISOR_BUNDLE="$PWD/hypervisor-community-bundle-<版本>"
cd "$HYPERVISOR_BUNDLE"
sha256sum -c SHA256SUMS
cd -
```

所有文件都应显示 `OK`。`SOURCE-REVISIONS` 记录了打包时的源码 revision 和 guest
镜像信息。

## 3. 构建或放置 Asterinas NixOS image

### 方式 A：使用网盘中的预构建 image

```bash
mkdir -p asterinas/target/nixos
zstd -d asterinas.img.zst -o asterinas/target/nixos/asterinas.img
```

### 方式 B：从源码构建

在 `asterinas/` 的父目录运行：

```bash
docker run --name asterinas-dev -it --privileged --network=host \
  -v /dev:/dev \
  -v "$PWD/asterinas:/root/asterinas" \
  asterinas/asterinas:0.18.0-20260702
```

进入容器后：

```bash
cd /root/asterinas
make nixos
exit
```

产物应为 `asterinas/target/nixos/asterinas.img`。后续再次进入同一个开发容器可用
`docker start -ai asterinas-dev`。

## 4. 注入测试程序和 guest 镜像

执行注入前必须关闭 Asterinas VM。以下命令会使用 loop device 挂载 image 的第二个
分区，因此需要 `sudo`：

```bash
export ASTERINAS_IMG="$PWD/asterinas/target/nixos/asterinas.img"

ASTERINAS_IMG="$ASTERINAS_IMG" \
KVMTOOL_SKIP_BUILD=1 \
KVMTOOL_STATIC=1 \
KVMTOOL_BIN="$HYPERVISOR_BUNDLE/bin/lkvm-static" \
KVMTOOL_KERNEL="$HYPERVISOR_BUNDLE/artifacts/bzImage" \
KVMTOOL_INITRD="$HYPERVISOR_BUNDLE/artifacts/initramfs.cpio.gz" \
"$HYPERVISOR_BUNDLE/scripts/nixos/sync_kvmtool.sh"

ASTERINAS_IMG="$ASTERINAS_IMG" \
KVM_HELLO_WORLD_SKIP_BUILD=1 \
KVM_HELLO_WORLD_BIN="$HYPERVISOR_BUNDLE/bin/kvm-hello-world" \
"$HYPERVISOR_BUNDLE/scripts/nixos/sync_kvm_hello_world.sh"
```

脚本结束时应列出 image 内的下列文件：

```text
/usr/local/bin/lkvm
/usr/local/bin/kvm-hello-world
/root/bzImage
/root/initramfs.cpio.gz
```

## 5. 启动 Asterinas host

```bash
cd asterinas
ENABLE_KVM=1 SMP=4 MEM=8G make run_nixos
```

`ENABLE_KVM=1` 同时启用外层 KVM 加速并向 Asterinas 暴露 VMX。看到
`root@asterinas` 提示符后，先检查：

```sh
uname -a
ls -l /dev/kvm
```

`/dev/kvm` 必须存在。本文后续命令都在 Asterinas shell 中执行。

## 6. 运行测试

### 测试 1：KVM API smoke test

依次检查 real mode、protected mode、32-bit paging 和 64-bit mode：

```sh
/usr/local/bin/kvm-hello-world -r
/usr/local/bin/kvm-hello-world -s
/usr/local/bin/kvm-hello-world -p
/usr/local/bin/kvm-hello-world -l
echo $?
```

四条命令都应正常返回，最后的退出码应为 `0`，并分别打印 `Testing ...`。这组测试
覆盖 `/dev/kvm` 打开、VM/VCPU 创建、guest memory、寄存器设置、`KVM_RUN` 和
`KVM_EXIT_HLT`。

### 测试 2：启动 Linux guest

```sh
/usr/local/bin/lkvm run \
  --kernel /root/bzImage \
  --initrd /root/initramfs.cpio.gz \
  --console serial \
  --network mode=none \
  --mem 128 \
  --cpus 1 \
  --params "root=/dev/ram0 console=ttyS0 earlyprintk=serial earlycon=uart8250,io,0x3f8 rdinit=/init init=/init nokaslr"
```

成功时可看到：

```text
init: start
guest linux is up
```

随后会进入 BusyBox shell。可以执行 `uname -a`、`cat /proc/cpuinfo` 等命令；输入
`exit` 后 guest 会重启或关机并返回 Asterinas shell。

### 测试 3：多 vCPU Linux guest

将上一条命令中的 `--cpus 1` 改为 `--cpus 4`。进入 guest 后检查：

```sh
cat /sys/devices/system/cpu/online
grep -c '^processor' /proc/cpuinfo
```

4 vCPU 时预期分别输出 `0-3` 和 `4`。建议从 2 或 4 vCPU 开始，不要直接把代码中
的 64-vCPU 上限当作已验证上限；当前实际测试覆盖到 16 vCPU。

### 测试 4：QEMU 自动化功能测试

QEMU 路径用于验证单 vCPU、多 vCPU、CPU affinity 和两个 VM 并发运行。它要求
QEMU 9.1.50 及其运行库已经被同步到 Asterinas image。完整的构建、同步和基准对照
流程见源码仓库中的 `asterinas/tools/hypervisor-bench/README.md`；测试包内的副本为
`docs/BENCHMARKS.md`。同步完成后，在 Asterinas 中运行：

```sh
/root/qemu-test/run-functional.sh --vcpus 4 all
```

完整成功输出以这一行结束：

```text
HYPERTEST_SUITE_DONE version=1 cases=3 status=PASS
```

失败日志保存在 Asterinas 的 `/tmp/hypervisor-functional/*.log`。
