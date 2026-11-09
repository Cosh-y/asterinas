{ config, lib, pkgs, ... }:

let
  busyboxRoot = pkgs.runCommand "kata-busybox-root" { } ''
    mkdir -p "$out/bin"
    cp ${pkgs.pkgsStatic.busybox}/bin/busybox "$out/bin/busybox"
    chmod 0755 "$out/bin/busybox"
    ln -s busybox "$out/bin/sh"
  '';

  busyboxImage = pkgs.dockerTools.buildImage {
    name = "busybox";
    tag = "latest";
    copyToRoot = busyboxRoot;
    config = {
      Cmd = [ "/bin/sh" ];
    };
  };

  registryProxy = "http://10.0.2.2:18089";
  registryNoProxy = "localhost,127.0.0.1,::1,10.0.2.2";
  registryPullImages = "docker.io/library/busybox:latest docker.m.daocloud.io/library/busybox:latest";
  kataQemuAccel = "kvm";
  kataQemuBootDiag = "1";

  qemuWrapper = pkgs.writeShellScriptBin "kata-qemu-wrapper" ''
    out_dir="''${KATA_DEBUG_OUT:-/tmp/kata-debug-out}"
    mkdir -p "$out_dir" 2>/dev/null || out_dir=/tmp
    stamp=$(${pkgs.coreutils}/bin/date +%s 2>/dev/null || echo now)
    base="$out_dir/qemu-wrapper-$stamp-$$"
    accel="''${KATA_QEMU_ACCEL:-kvm}"
    case "$accel" in
      kvm|tcg) ;;
      *)
        echo "unsupported KATA_QEMU_ACCEL=$accel" > "$base.stderr"
        exit 2
        ;;
    esac
    printf 'KATA_QEMU_ACCEL=%s\n' "$accel" > "$base.mode" 2>/dev/null || true

    args=()
    append_arg=0
    q35_machine=0
    for arg in "$@"; do
      if [ "$append_arg" = 1 ]; then
        if [ "''${KATA_QEMU_BOOT_DIAG:-0}" = 1 ]; then
          args+=("$arg console=ttyS0,115200 earlycon=uart8250,io,0x3f8,115200n8 ignore_loglevel")
        else
          args+=("$arg")
        fi
        append_arg=0
        continue
      fi

      case "$arg" in
        -append)
          args+=("$arg")
          append_arg=1
          ;;
        q35|q35,*)
          q35_machine=1
          if [ "$accel" = tcg ]; then
            args+=("''${arg/accel=kvm/accel=tcg}")
          else
            args+=("$arg")
          fi
          ;;
        *accel=kvm*)
          if [ "$accel" = tcg ]; then
            args+=("''${arg/accel=kvm/accel=tcg}")
          else
            args+=("$arg")
          fi
          ;;
        host|host,*)
          if [ "$accel" = tcg ]; then
            args+=("max")
          else
            args+=("$arg")
          fi
          ;;
        *)
          args+=("$arg")
          ;;
      esac
    done
    if [ "''${KATA_QEMU_BOOT_DIAG:-0}" = 1 ] && [ "$q35_machine" = 1 ]; then
      args+=(
        -serial "file:$base.boot-serial.log"
        -d guest_errors,cpu_reset
        -D "$base.qemu-debug.log"
      )
    fi

    {
      printf '%s\n' "${pkgs.qemu_test}/bin/qemu-system-x86_64"
      for arg in "$@"; do
        printf '%s\n' "$arg"
      done
    } > "$base.argv" 2>/dev/null || true
    {
      printf '%s\n' "${pkgs.qemu_test}/bin/qemu-system-x86_64"
      for arg in "''${args[@]}"; do
        printf '%s\n' "$arg"
      done
    } > "$base.effective-argv" 2>/dev/null || true
    exec ${pkgs.qemu_test}/bin/qemu-system-x86_64 "''${args[@]}" 2>"$base.stderr"
  '';

  kataRuntimeRs = pkgs.rustPlatform.buildRustPackage rec {
    pname = "kata-runtime-rs";
    version = pkgs.kata-runtime.version;

    src = pkgs.kata-runtime.src;
    sourceRoot = "${src.name}/src/runtime-rs";

    cargoLock = {
      lockFile = "${src}/src/runtime-rs/Cargo.lock";
      outputHashes = {
        "api_client-0.1.0" = "sha256-aWtVgYlcbssL7lQfMFGJah8DrJN0s/w1ZFncCPHT1aE=";
      };
    };

    nativeBuildInputs = with pkgs; [
      pkg-config
      protobuf
    ];

    buildInputs = with pkgs; [
      openssl
      systemd
      zlib
    ];

    postPatch = ''
      chmod -R u+w ..
      patch -p1 < ${extraFileDir + "/runtime-rs-qemu-pci-coldplug.patch"}
      patch -p1 < ${extraFileDir + "/runtime-rs-qemu-qmp-timeout.patch"}
      substituteInPlace crates/runtimes/virt_container/Cargo.toml \
        --replace 'default = ["cloud-hypervisor"]' \
                  'default = []'
    '';

    preBuild = ''
      make static-checks-build \
        HYPERVISOR=qemu \
        USE_BUILDIN_DB=false \
        DBCMD= \
        CLHCMD= \
        FCCMD= \
        REMOTE= \
        PREFIX=/run/current-system/sw \
        QEMUBINDIR=/run/current-system/sw/bin \
        LIBEXECDIR=/run/current-system/sw/bin
    '';

    cargoBuildFlags = [
      "-p"
      "shim"
      "--bin"
      "containerd-shim-kata-v2"
    ];

    doCheck = false;

    installPhase = ''
      runHook preInstall

      shim_bin=$(find target -type f -path '*/release/containerd-shim-kata-v2' | head -n1)
      install -Dm755 "$shim_bin" "$out/bin/containerd-shim-kata-v2"

      install -Dm644 \
        config/configuration-qemu-runtime-rs.toml \
        "$out/share/defaults/kata-containers/runtime-rs/configuration-qemu-runtime-rs.toml"
      cfg="$out/share/defaults/kata-containers/runtime-rs/configuration-qemu-runtime-rs.toml"
      awk '
        /^\[agent\.kata\]/ { section = "agent" }
        /^\[hypervisor\.qemu\]/ { section = "qemu" }
        /^\[runtime\]/ { section = "runtime" }
        /^\[/ && !/^\[agent\.kata\]/ && !/^\[hypervisor\.qemu\]/ && !/^\[runtime\]/ { section = "" }
        section == "qemu" && /^path = / {
          print "path = \"${qemuWrapper}/bin/kata-qemu-wrapper\""
          next
        }
        section == "qemu" && /^valid_hypervisor_paths = / {
          print "valid_hypervisor_paths = [\"${qemuWrapper}/bin/kata-qemu-wrapper\"]"
          next
        }
        section == "qemu" && /^vm_rootfs_driver = / {
          print "vm_rootfs_driver = \"virtio-blk-pci\""
          next
        }
        section == "qemu" && /^machine_accelerators=/ {
          print "machine_accelerators=\"kernel-irqchip=on,pit=off,pic=off,acpi=off\""
          next
        }
        section == "qemu" && /^block_device_driver = / {
          print "block_device_driver = \"virtio-blk-pci\""
          next
        }
        section == "qemu" && /^#?disable_image_nvdimm = / {
          print "disable_image_nvdimm = true"
          next
        }
        section == "qemu" && /^#?default_vcpus = / {
          print "default_vcpus = 1"
          next
        }
        section == "qemu" && /^default_memory = / {
          print "default_memory = 128"
          next
        }
        section == "qemu" && /^default_maxmemory = / {
          print "default_maxmemory = 256"
          next
        }
        section == "qemu" && /^#?default_maxvcpus = / {
          print "default_maxvcpus = 1"
          next
        }
        { print }
      ' "$cfg" > "$cfg.tmp"
      mv "$cfg.tmp" "$cfg"

      ln -s configuration-qemu-runtime-rs.toml \
        "$out/share/defaults/kata-containers/runtime-rs/configuration.toml"

      runHook postInstall
    '';
  };
in

{
  hardware.enableRedistributableFirmware = lib.mkForce false;

  virtualisation.containerd = {
    enable = true;
    settings = {
      version = 2;
      plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata = {
        runtime_type = "io.containerd.kata.v2";
      };
    };
  };

  environment.systemPackages = with pkgs; [
    curl
    fio
    iproute2
    kataRuntimeRs
    kata-runtime.passthru.kata-images
    qemuWrapper
    qemu_test
    runc
    socat
    virtiofsd
    (writeShellScriptBin "kata-debug-check" ''
      exec sh /etc/kata-debug/guest/00-check-kata-env.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-start-containerd" ''
      exec sh /etc/kata-debug/guest/01-start-containerd-debug.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-run-rootfs" ''
      exec sh /etc/kata-debug/guest/02-run-rootfs-probe.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-watch-serial" ''
      exec sh /etc/kata-debug/guest/03-watch-qemu-serial.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-collect" ''
      exec sh /etc/kata-debug/guest/04-collect-logs.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-run-all" ''
      exec sh /etc/kata-debug/guest/10-run-all.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-run-image-probe" ''
      exec sh /etc/kata-debug/guest/20-run-image-probe.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-run-image" ''
      exec sh /etc/kata-debug/guest/11-run-image-all.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-pull-image" ''
      exec sh /etc/kata-debug/guest/21-pull-image.sh "$@"
    '')
    (writeShellScriptBin "kata-debug-stop" ''
      exec sh /etc/kata-debug/guest/99-stop-containerd-debug.sh "$@"
    '')
  ];

  environment.variables = {
    HTTP_PROXY = registryProxy;
    HTTPS_PROXY = registryProxy;
    ALL_PROXY = registryProxy;
    KATA_PULL_IMAGES = registryPullImages;
    KATA_QEMU_ACCEL = kataQemuAccel;
    KATA_QEMU_BOOT_DIAG = kataQemuBootDiag;
    NO_PROXY = registryNoProxy;
    http_proxy = registryProxy;
    https_proxy = registryProxy;
    all_proxy = registryProxy;
    no_proxy = registryNoProxy;
  };

  environment.pathsToLink = [
    "/share/kata-containers"
  ];

  environment.etc."kata-containers/configuration.toml".source =
    "${kataRuntimeRs}/share/defaults/kata-containers/runtime-rs/configuration.toml";

  environment.etc."kata-debug/guest".source = extraFileDir + "/guest-scripts";
  environment.etc."kata-debug/busybox.tar".source = busyboxImage;

  systemd.services.containerd = {
    environment = {
      KATA_CONF_FILE = "/etc/kata-containers/configuration.toml";
      HTTP_PROXY = registryProxy;
      HTTPS_PROXY = registryProxy;
      ALL_PROXY = registryProxy;
      KATA_PULL_IMAGES = registryPullImages;
      KATA_QEMU_ACCEL = kataQemuAccel;
      KATA_QEMU_BOOT_DIAG = kataQemuBootDiag;
      NO_PROXY = registryNoProxy;
      http_proxy = registryProxy;
      https_proxy = registryProxy;
      all_proxy = registryProxy;
      no_proxy = registryNoProxy;
    };
    path = [
      kataRuntimeRs
      pkgs.qemu_test
      pkgs.virtiofsd
    ];
  };
}
