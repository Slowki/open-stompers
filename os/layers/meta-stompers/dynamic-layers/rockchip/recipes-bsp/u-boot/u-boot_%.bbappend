FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

DEPENDS:append = " u-boot-mkimage-native"

SRC_URI:append = " \
    file://stompers-ab.cfg \
    file://boot.cmd.in \
"

do_compile:append() {
    sed -e 's|@KERNEL_BOOTFILE@|${UBOOT_EXTLINUX_KERNEL_IMAGE}|' \
        -e 's|@BOOTARGS@|${UBOOT_EXTLINUX_KERNEL_ARGS} ${UBOOT_EXTLINUX_CONSOLE}|' \
        ${UNPACKDIR}/boot.cmd.in > ${B}/boot.cmd
    mkimage -A ${UBOOT_ARCH} -T script -C none -n "Stompers OS A/B boot" -d ${B}/boot.cmd ${B}/boot.scr
}

# Every rootfs slot carries the boot script, so either slot can start the device.
do_install:append() {
    install -D -m 0644 ${B}/boot.scr ${D}/boot/boot.scr
}

FILES:${PN}-extlinux += "/boot/boot.scr"
