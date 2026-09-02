SUMMARY = "WuQi WQ9201S SDIO Wi-Fi driver"
DESCRIPTION = "Out-of-tree wq_wlan kernel module for the WuQi WQ9201S Wi-Fi and \
Bluetooth combo fitted to some Radxa ZERO 3W boards, plus a service that attaches \
its Bluetooth controller on UART1. Firmware is wq9201s-firmware. Adapted from \
pollen-robotics/microduck_yocto (Apache-2.0)."
HOMEPAGE = "https://github.com/Seeed-Studio/seeed_armbian_extension/tree/deb_source/wq9201s-wifi-bt-dkms"
LICENSE = "GPL-2.0-or-later"
LIC_FILES_CHKSUM = "file://debian/copyright;md5=3727876b7206ead678f0a134cddb542b"

# WuQi's SDK v2.2.3.94 as Seeed packages it for their Debian images.
SRCREV = "1f4bd1cbf6e844291a54ca55816c1ac827b6bd02"
SRC_URI = " \
    git://github.com/Seeed-Studio/seeed_armbian_extension.git;protocol=https;branch=deb_source \
    file://0001-wq_wlan-build-on-Linux-6.18.patch \
    file://wq9201s-bt-attach \
    file://wq9201s-bt.service \
"
S = "${UNPACKDIR}/${BP}/wq9201s-wifi-bt-dkms"

COMPATIBLE_MACHINE = "(rockchip)"

inherit module systemd

B = "${S}/src/wifi/rwnx_drv"
MAKE_TARGETS = "-C ${STAGING_KERNEL_DIR} M=${B} modules"
MODULES_INSTALL_TARGET = "-C ${STAGING_KERNEL_DIR} M=${B} modules_install"
MODULES_MODULE_SYMVERS_LOCATION = "."
# The Makefile's own targets, such as clean in do_configure, default to the host kernel.
EXTRA_OEMAKE += "KERNELDIR=${STAGING_KERNEL_DIR}"

# The SDK was built with GCC 12; GCC 15's new warnings must not fail the build.
drop_werror() {
    sed -i '/^ccflags-y += -Werror$/d' ${B}/Makefile
    if grep -q -- '-Werror' ${B}/Makefile; then
        bbfatal "-Werror is still set in the wq_wlan Makefile"
    fi
}
do_patch[postfuncs] += "drop_werror"

do_install:append() {
    install -d ${D}${libexecdir}
    install -m 0755 ${UNPACKDIR}/wq9201s-bt-attach ${D}${libexecdir}/
    install -d ${D}${systemd_system_unitdir}
    sed -e 's,@LIBEXECDIR@,${libexecdir},g' ${UNPACKDIR}/wq9201s-bt.service > ${D}${systemd_system_unitdir}/wq9201s-bt.service
    chmod 0644 ${D}${systemd_system_unitdir}/wq9201s-bt.service
}

# The SDIO function shows no matching modalias before firmware download, so load
# the module at boot. It is idle on boards with an AIC8800.
KERNEL_MODULE_AUTOLOAD += "wq_wlan"

RDEPENDS:${PN} += "wq9201s-firmware"

PACKAGES =+ "${PN}-bt"
FILES:${PN}-bt = "${libexecdir}/wq9201s-bt-attach ${systemd_system_unitdir}/wq9201s-bt.service"
SYSTEMD_PACKAGES = "${PN}-bt"
SYSTEMD_SERVICE:${PN}-bt = "wq9201s-bt.service"
RDEPENDS:${PN}-bt = "bluez5 procps"
