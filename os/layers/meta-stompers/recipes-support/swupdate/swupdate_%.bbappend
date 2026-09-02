FILESEXTRAPATHS:prepend := "${THISDIR}/${BPN}:"

SRC_URI += " \
    file://systemd.cfg \
    file://hash.cfg \
    file://part-format.cfg \
    file://archive.cfg \
"

# Jetson switches slots through UEFI; other platforms use the U-Boot environment.
SRC_URI:append:tegra = " file://disable-uboot.cfg"

DEPENDS += "e2fsprogs"

# The machine config package provides a slot-aware /etc/swupdate.cfg.
do_install:append() {
    rm -f ${D}${sysconfdir}/swupdate.cfg
}

RDEPENDS:${PN} += "stompers-swupdate-config"
