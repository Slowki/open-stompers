SUMMARY = "swupdate bundle for Stompers OS"
DESCRIPTION = "Writes stompers-image to the inactive rootfs slot and stages the \
bootloader capsule, which switches both slots on the next reboot."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COREBASE}/meta/COPYING.MIT;md5=3da9cfbcb788c80a0384361b4de20420"

SRC_URI = "file://sw-description"

inherit swupdate image_types_tegra

require ../../recipes-support/swupdate/stompers-swupdate.inc

SWUPDATE_CORE_IMAGE_NAME = "stompers-image"
ROOTFS_DEVICE_PATH = "/dev/disk/by-partlabel"
ROOTFS_FILENAME = "${SWUPDATE_CORE_IMAGE_NAME}-${MACHINE}.rootfs.tar.gz"

# L4TLauncher lives on the ESP unless it is built into a minimal UEFI image.
ESP_ARCHIVE = "${TEGRA_ESP_IMAGE}-${MACHINE}.tar.gz"
USES_ESP = "${@'0' if bb.utils.to_boolean(d.getVar('TEGRA_MINIMAL_BOOT')) else '1'}"

def stompers_esp_update(d):
    if d.getVar('USES_ESP') != '1':
        return ""
    return f""",
                    {{
                        filename = "{d.getVar('ESP_ARCHIVE')}";
                        type = "archive";
                        installed-directly = true;
                        path = "/boot/efi";
                        name = "tegra-bootloader-capsule";
                        version = "{d.getVar('STOMPERS_SWUPDATE_BOOTLOADER_VERSION')}";
                    }}"""

OPTIONAL_ESP_UPDATE = "${@stompers_esp_update(d)}"

IMAGE_DEPENDS = "${SWUPDATE_CORE_IMAGE_NAME}${@' tegra-espimage' if d.getVar('USES_ESP') == '1' else ''}"

SWUPDATE_IMAGES = "${ROOTFS_FILENAME} tegra-bl.cap stompers-swupdate-script.lua${@' ' + d.getVar('ESP_ARCHIVE') if d.getVar('USES_ESP') == '1' else ''}"

do_swuimage[depends] += " \
    tegra-uefi-capsules:do_deploy \
    stompers-swupdate-script:do_deploy \
    ${@'tegra-espimage:do_image_complete' if d.getVar('USES_ESP') == '1' else ''} \
"
