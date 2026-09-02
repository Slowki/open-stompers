SUMMARY = "Fall back to USB device mode on the Jetson USB-C port"
DESCRIPTION = "Sets the USB2 port 0 role switch to device when the fusb301 Type-C \
controller fails to probe, so l4t-usb-device-mode still shows up on a connected host."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://stompers-usb-device-role \
    file://stompers-usb-device-role.service \
"

S = "${UNPACKDIR}"

COMPATIBLE_MACHINE = "(tegra)"

inherit systemd

SYSTEMD_SERVICE:${PN} = "stompers-usb-device-role.service"

do_install() {
    install -d ${D}${libexecdir}
    install -m 0755 ${S}/stompers-usb-device-role ${D}${libexecdir}/
    install -d ${D}${systemd_system_unitdir}
    sed -e 's,@LIBEXECDIR@,${libexecdir},g' ${S}/stompers-usb-device-role.service > ${D}${systemd_system_unitdir}/stompers-usb-device-role.service
    chmod 0644 ${D}${systemd_system_unitdir}/stompers-usb-device-role.service
}
