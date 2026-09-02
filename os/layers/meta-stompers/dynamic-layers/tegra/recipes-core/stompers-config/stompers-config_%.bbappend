FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI += "file://tegra-se-disable.conf"

do_install:append() {
    install -d ${D}${sysconfdir}/modprobe.d
    install -m 0644 ${UNPACKDIR}/tegra-se-disable.conf ${D}${sysconfdir}/modprobe.d/
}
