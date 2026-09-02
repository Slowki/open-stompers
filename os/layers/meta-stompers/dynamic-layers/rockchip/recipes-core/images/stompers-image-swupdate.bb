SUMMARY = "swupdate bundle for Stompers OS"
DESCRIPTION = "Writes stompers-image to the inactive rootfs slot and makes U-Boot \
boot it next. U-Boot falls back to the old slot after three failed boots."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COREBASE}/meta/COPYING.MIT;md5=3da9cfbcb788c80a0384361b4de20420"

SRC_URI = "file://sw-description"

inherit swupdate

SWUPDATE_CORE_IMAGE_NAME = "stompers-image"
ROOTFS_DEVICE_PATH = "/dev/disk/by-partlabel"
ROOTFS_FILENAME = "${SWUPDATE_CORE_IMAGE_NAME}-${MACHINE}.rootfs.tar.gz"

IMAGE_DEPENDS = "${SWUPDATE_CORE_IMAGE_NAME}"
SWUPDATE_IMAGES = "${ROOTFS_FILENAME}"
