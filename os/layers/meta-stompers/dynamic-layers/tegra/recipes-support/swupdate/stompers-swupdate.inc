inherit l4t_version

# Version recorded for the bootloader capsule in sw-versions.
STOMPERS_SWUPDATE_BOOTLOADER_VERSION ?= "${L4T_VERSION}"

# UEFI picks up capsules from \EFI\UpdateCapsule on the ESP, mounted at /boot/efi.
STOMPERS_SWUPDATE_CAPSULE_INSTALL_PATH ?= "/boot/efi/EFI/UpdateCapsule/TEGRA_BL.Cap"
