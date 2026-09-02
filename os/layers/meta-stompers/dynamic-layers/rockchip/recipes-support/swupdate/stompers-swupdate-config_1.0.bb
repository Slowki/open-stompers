SUMMARY = "Slot-aware swupdate configuration for Stompers OS on Rockchip"
DESCRIPTION = "Selects the swupdate target slot from the running slot and marks \
a slot as good once it reaches multi-user.target, resetting its U-Boot boot counter."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://swupdate.cfg.in \
    file://swupdate-genconfig.sh.in \
    file://swupdate-genconfig.conf.in \
    file://stompers-current-slot \
    file://stompers-boot-success.service.in \
"

SWUPDATE_BOARDNAME ??= "${MACHINE}"
SWUPDATE_HWREVISION ??= "1.0"

S = "${UNPACKDIR}"
B = "${WORKDIR}/build"

inherit systemd

SYSTEMD_SERVICE:${PN} = "stompers-boot-success.service"

do_compile() {
    echo "${SWUPDATE_BOARDNAME} ${SWUPDATE_HWREVISION}" > ${B}/hwrevision
    sed -e 's,@MODEL@,${SWUPDATE_BOARDNAME},g' ${S}/swupdate.cfg.in > ${B}/swupdate.cfg.in
    sed -e 's,@DATADIR@,${datadir},g' -e 's,@BINDIR@,${bindir},g' ${S}/swupdate-genconfig.sh.in > ${B}/swupdate-genconfig
    sed -e 's,@LIBEXECDIR@,${libexecdir},g' ${S}/swupdate-genconfig.conf.in > ${B}/swupdate-genconfig.conf
    sed -e 's,@BINDIR@,${bindir},g' ${S}/stompers-boot-success.service.in > ${B}/stompers-boot-success.service
}

do_install() {
    install -d ${D}${sysconfdir}
    install -m 0644 ${B}/hwrevision ${D}${sysconfdir}/
    ln -s /run/swupdate/swupdate.cfg ${D}${sysconfdir}/swupdate.cfg

    install -d ${D}${datadir}/swupdate
    install -m 0644 ${B}/swupdate.cfg.in ${D}${datadir}/swupdate/

    install -d ${D}${bindir}
    install -m 0755 ${S}/stompers-current-slot ${D}${bindir}/

    install -d ${D}${libexecdir}/swupdate
    install -m 0755 ${B}/swupdate-genconfig ${D}${libexecdir}/swupdate/

    install -d ${D}${systemd_system_unitdir}/swupdate.service.d
    install -m 0644 ${B}/swupdate-genconfig.conf ${D}${systemd_system_unitdir}/swupdate.service.d/
    install -m 0644 ${B}/stompers-boot-success.service ${D}${systemd_system_unitdir}/
}

FILES:${PN} += "${datadir}/swupdate ${systemd_system_unitdir}"

PACKAGE_ARCH = "${MACHINE_ARCH}"
RDEPENDS:${PN} = "u-boot-env u-boot-fw-utils"
