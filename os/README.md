# Stompers OS

OS images for Open Stompers robots.

| Platform      | `PLATFORM`       | BSP                                              |
| ------------- | ---------------- | ------------------------------------------------ |
| Nvidia Jetson | `jetson-orin-nx` | meta-tegra, JetPack 7.2.1 (Jetson Linux R39.2.1) |
| Radxa ZERO 3W | `radxa-zero-3w`  | meta-rockchip, Linux 6.18 and U-Boot             |

## Prerequisites

- For building: Docker or Podman
- For flashing Jetson: dtc, cpp, sgdisk, udisksctl, bmaptool
- For flashing Radxa: bmaptool

## Building

```bash
./build.sh PLATFORM image   # Flashable image and .swu update bundle
./build.sh PLATFORM sdk     # Cross-compilation SDK installer
./build.sh PLATFORM shell   # Bitbake shell inside the build container
./build.sh PLATFORM lock    # Re-pin layer revisions in kas/PLATFORM.lock.yml
```

Everything goes to `build/`.
Images are in `build/PLATFORM/tmp/deploy/images/MACHINE/`, where `MACHINE` is `p3768-0000-p3767-0000` for Jetson and `radxa-zero-3w` for Radxa.

## Nvidia Jetson

| File                                                             | Purpose           |
| ---------------------------------------------------------------- | ----------------- |
| `stompers-image-p3768-0000-p3767-0000.rootfs.tegraflash-tar.zst` | Initial flashing  |
| `stompers-image-swupdate-p3768-0000-p3767-0000.rootfs.swu`       | OTA update bundle |

### Flashing

1. **Put the device into recovery mode** - Any of the following will work:
   a. `sudo reboot --force forced-recovery`
   b. Physically short the `FC REC` and `GND` pins on the button header.
   c. In the BIOS `NVIDIA Configuration > Boot Configuration > Boot Into Recovery`
2. **Unpack and flash**:

```bash
 mkdir flash && tar -xf build/jetson-orin-nx/tmp/deploy/images/p3768-0000-p3767-0000/stompers-image-p3768-0000-p3767-0000.rootfs.tegraflash-tar.zst -C flash
```

```bash
cd flash && ./initrd-flash
```

The first flash also writes the QSPI boot firmware, which is required when coming from an older JetPack. Later flashes can skip it with `./initrd-flash --external-only`.

### Over-the-Air Updating

The bundle writes the inactive rootfs slot and stages a UEFI capsule. On reboot UEFI applies the capsule to the
inactive bootloader slot and reboots again into the new bootloader and rootfs. Check slots with
`sudo nvbootctrl dump-slots-info`.

### NVMe layout

| Partitions              | Size        | Notes                                    |
| ----------------------- | ----------- | ---------------------------------------- |
| `A_kernel` … `reserved` | ~1.5 GB     | NVIDIA boot, recovery, ESP and UDA       |
| `APP`, `APP_b`          | 16 GiB each | Rootfs slots A and B                     |
| `data`                  | Rest        | Formatted as ext4 on first boot, `/data` |

### Hardware notes

- NVIDIA's `tegra_se` Security Engine crypto driver is disabled (`/etc/modprobe.d/tegra-se-disable.conf`). It oopses in
  `crypto_dequeue_request()` when a Wi-Fi connection sets up its keys, panicking the kernel; without it the kernel uses
  the ARMv8 crypto extensions.
- The kernel panics on soft and hard lockups and keeps the panic log in ramoops. After the reboot, systemd-pstore moves
  it into the journal: `journalctl --all SYSLOG_IDENTIFIER=systemd-pstore`.
- The `fusb301` Type-C controller fails to probe, so `stompers-usb-device-role.service` puts the USB-C port in device
  mode.

## Radxa ZERO 3W

| File                                               | Purpose           |
| -------------------------------------------------- | ----------------- |
| `stompers-image-radxa-zero-3w.rootfs.wic`          | Initial flashing  |
| `stompers-image-swupdate-radxa-zero-3w.rootfs.swu` | OTA update bundle |

### Flashing

Write the image to a microSD card (at least 16 GB) and boot from it:

```bash
bmaptool copy build/radxa-zero-3w/tmp/deploy/images/radxa-zero-3w/stompers-image-radxa-zero-3w.rootfs.wic /dev/sdX
```

To install to eMMC, boot from the card and write the same image to `/dev/mmcblk0`.

### Over-the-Air Updating

The bundle writes the inactive rootfs slot and points `BOOT_ORDER` in the U-Boot environment at it. U-Boot gives a
slot three boot attempts; `stompers-boot-success.service` resets the counter once the system is up, otherwise U-Boot
falls back to the previous slot. Check the slots with `fw_printenv BOOT_ORDER BOOT_A_LEFT BOOT_B_LEFT`.

### SD card and eMMC layout

| Partitions           | Size       | Notes                                            |
| -------------------- | ---------- | ------------------------------------------------ |
| `loader1` … `atf`    | 16 MiB     | Rockchip boot loader and U-Boot environment      |
| `rootfsA`, `rootfsB` | 4 GiB each | Rootfs slots A and B                             |
| `data`               | Rest       | Created by systemd-repart on first boot, `/data` |

The slot size is `STOMPERS_ROOTFS_SLOT_SIZE` in `layers/meta-stompers/conf/distro/stompers.conf`.

### Hardware notes

- USB device mode uses the USB 2.0 Type-C port, which the image switches to peripheral mode. The USB 3.0 port stays
  a host port.

## Over-the-Air Updating

Install a `.swu` bundle with the web UI at `http://stomper.local:8080`, or from a shell:

```bash
scp stompers-image-swupdate-MACHINE.rootfs.swu stomper@stomper.local:/tmp/
ssh stomper@stomper.local sudo swupdate-client /tmp/stompers-image-swupdate-MACHINE.rootfs.swu
ssh stomper@stomper.local sudo reboot
```

Bundles are not signed yet, so anyone with access to the swupdate web UI or socket can install one.

## Connecting

```bash
ssh stomper@192.168.55.1    # Over the USB-C cable
ssh stomper@stomper.local   # Over Ethernet or Wi-Fi
```

Join a Wi-Fi network with `iwctl station wlan0 connect <SSID>`.

## Cross-compiling

```bash
./build.sh PLATFORM sdk
build/PLATFORM/tmp/deploy/sdk/stompers-glibc-x86_64-stompers-image-*-toolchain-*.sh
source /opt/stompers/1.0.0/MACHINE/environment-setup-*-stompers-linux
```

After sourcing, `$CC`, `$CXX`, `CMAKE_TOOLCHAIN_FILE` (via `OECORE_CMAKE_TOOLCHAIN` helpers) and `pkg-config`
point at the target sysroot.

## Layout

- `kas/common.yml` – layers and settings shared by all platforms; `kas/PLATFORM.yml` adds the BSP.
- `layers/meta-stompers/` – distro, images and shared recipes.
- `layers/meta-stompers/dynamic-layers/{tegra,rockchip}/` – platform recipes, enabled when the BSP layer is present.
