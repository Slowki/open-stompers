SUMMARY = "AICSemi AIC8800 SDIO Wi-Fi and Bluetooth driver"
DESCRIPTION = "Out-of-tree aic8800 kernel modules and firmware for the AIC8800DS2 \
fitted to most Radxa ZERO 3W boards, plus a service that attaches its Bluetooth \
controller on UART1. Adapted from pollen-robotics/microduck_yocto (Apache-2.0)."
HOMEPAGE = "https://github.com/radxa-pkg/aic8800"
# src/* is AICSemi's GPL-2 driver; radxa's Debian packaging around it is GPL-3+.
LICENSE = "GPL-2.0-only & GPL-3.0-or-later"
LIC_FILES_CHKSUM = " \
    file://LICENSE;md5=1ebbd3e34237af26da5dc08a4e440464 \
    file://debian/copyright;md5=ccd22839cbff32b2bbe9a6ec05d11739 \
"

# Tag 5.0+git20260123.5f7be68d-9.
SRCREV = "d13d07963cd15d731e2895e8288a04cca6152ac9"
SRC_URI = " \
    git://github.com/radxa-pkg/aic8800.git;protocol=https;nobranch=1 \
    file://aic8800-bt-attach \
    file://aic8800-bt.service \
"

COMPATIBLE_MACHINE = "(rockchip)"

inherit module systemd

# The vendor Makefile is a kbuild fragment, so drive kbuild directly.
B = "${S}/src/SDIO/driver_fw/driver/aic8800"
MAKE_TARGETS = "-C ${STAGING_KERNEL_DIR} M=${B} modules"
MODULES_INSTALL_TARGET = "-C ${STAGING_KERNEL_DIR} M=${B} modules_install"
MODULES_MODULE_SYMVERS_LOCATION = "."

# Apply radxa's debian/patches/series: kernel compatibility fixes up to 7.3 and
# the /lib/firmware/aic8800_fw/SDIO firmware path. The vendor drop mixes CRLF
# and LF line endings while the patches are LF, so normalise the sources first.
# quilt tracks applied patches, so a re-run of do_patch is a no-op.
apply_debian_patches() {
    cd ${S}
    find src -type f \( -name '*.c' -o -name '*.h' -o -name 'Makefile*' -o -name 'Kconfig*' \) \
        -exec sed -i 's/\r$//' {} +
    status=0
    QUILT_PATCHES=debian/patches quilt --quiltrc /dev/null push -a || status=$?
    # 2 means every patch is already applied.
    if [ "$status" != 0 ] && [ "$status" != 2 ]; then
        bbfatal "quilt push -a failed with exit code $status"
    fi
}
do_patch[postfuncs] += "apply_debian_patches"

FIRMWARE_DIR = "${nonarch_base_libdir}/firmware/aic8800_fw/SDIO"

do_install:append() {
    # Every chip variant; the driver picks the matching directory at runtime.
    install -d ${D}${FIRMWARE_DIR}
    cp -R --no-preserve=ownership ${S}/src/SDIO/driver_fw/fw/* ${D}${FIRMWARE_DIR}/
    rm -f ${D}${FIRMWARE_DIR}/*.7z

    install -d ${D}${libexecdir}
    install -m 0755 ${UNPACKDIR}/aic8800-bt-attach ${D}${libexecdir}/
    install -d ${D}${systemd_system_unitdir}
    sed -e 's,@LIBEXECDIR@,${libexecdir},g' ${UNPACKDIR}/aic8800-bt.service > ${D}${systemd_system_unitdir}/aic8800-bt.service
    chmod 0644 ${D}${systemd_system_unitdir}/aic8800-bt.service
}

PACKAGES =+ "${PN}-firmware ${PN}-bt"
FILES:${PN}-firmware = "${FIRMWARE_DIR}"
FILES:${PN}-bt = "${libexecdir}/aic8800-bt-attach ${systemd_system_unitdir}/aic8800-bt.service"

# Both modules carry SDIO device tables, so udev loads them only when the chip is present.
RDEPENDS:${PN} += "${PN}-firmware"

SYSTEMD_PACKAGES = "${PN}-bt"
SYSTEMD_SERVICE:${PN}-bt = "aic8800-bt.service"
RDEPENDS:${PN}-bt = "bluez5"
