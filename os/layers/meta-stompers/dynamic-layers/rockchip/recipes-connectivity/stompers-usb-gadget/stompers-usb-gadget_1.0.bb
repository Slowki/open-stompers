SUMMARY = "NVIDIA JetPack style USB device mode"
DESCRIPTION = "RNDIS, ECM and ACM gadgets on the USB OTG port. The network \
gadgets are bridged at 192.168.55.1 with a DHCP server handing 192.168.55.100 \
to the host, and /dev/ttyGS0 runs a login console. Adapted from meta-tegra's \
l4t-usb-device-mode."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://70-usb-gadget.network \
    file://70-usbbr0.network \
    file://usbbr0.netdev \
    file://98-usb-gadget-tty.rules \
    file://serial-getty@ttyGS0.service.d/no-tty-reset.conf \
    file://stompers.schema.in \
    file://stompers-gadget-config-setup \
    file://usbgx-stompers.conf \
"

S = "${UNPACKDIR}"

inherit features_check

REQUIRED_DISTRO_FEATURES = "systemd"

do_configure[noexec] = "1"
do_compile[noexec] = "1"

do_install() {
    install -d ${D}${sysconfdir}/systemd/network
    install -m 0644 ${S}/usbbr0.netdev ${S}/70-usbbr0.network ${S}/70-usb-gadget.network ${D}${sysconfdir}/systemd/network/

    install -d ${D}${sysconfdir}/udev/rules.d
    install -m 0644 ${S}/98-usb-gadget-tty.rules ${D}${sysconfdir}/udev/rules.d/

    install -d ${D}${systemd_system_unitdir}/serial-getty@ttyGS0.service.d
    install -m 0644 ${S}/serial-getty@ttyGS0.service.d/no-tty-reset.conf ${D}${systemd_system_unitdir}/serial-getty@ttyGS0.service.d/

    install -d ${D}${datadir}/usbgx
    install -m 0644 ${S}/stompers.schema.in ${D}${datadir}/usbgx/
    install -d ${D}${sysconfdir}/usbgx
    ln -sf /run/usbgx/stompers.schema ${D}${sysconfdir}/usbgx/stompers.schema

    install -d ${D}${bindir}
    install -m 0755 ${S}/stompers-gadget-config-setup ${D}${bindir}/

    install -d ${D}${systemd_system_unitdir}/usbgx.service.d
    sed -e 's,@BINDIR@,${bindir},g' ${S}/usbgx-stompers.conf > ${D}${systemd_system_unitdir}/usbgx.service.d/usbgx-stompers.conf
}

FILES:${PN} += "${datadir}/usbgx ${systemd_system_unitdir}"

RDEPENDS:${PN} = "libusbgx-examples"
RRECOMMENDS:${PN} = " \
    kernel-module-bridge \
    kernel-module-libcomposite \
    kernel-module-usb-f-acm \
    kernel-module-usb-f-ecm \
    kernel-module-usb-f-rndis \
    kernel-module-usb-f-serial \
"
