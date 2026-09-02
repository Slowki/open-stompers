SUMMARY = "Slot-aware swupdate configuration for Stompers OS"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

require stompers-swupdate.inc

SRC_URI = " \
    file://swupdate.cfg.in \
    file://swupdate-genconfig.sh.in \
    file://swupdate-genconfig.conf.in \
"

SWUPDATE_BOARDNAME ??= "${MACHINE}"
SWUPDATE_HWREVISION ??= "1.0"

S = "${UNPACKDIR}"
B = "${WORKDIR}/build"

do_compile() {
    echo "${SWUPDATE_BOARDNAME} ${SWUPDATE_HWREVISION}" > ${B}/hwrevision
    sed -e 's,@MODEL@,${SWUPDATE_BOARDNAME},g' ${S}/swupdate.cfg.in > ${B}/swupdate.cfg.in
    sed -e 's,@DATADIR@,${datadir},g' ${S}/swupdate-genconfig.sh.in > ${B}/swupdate-genconfig
    sed -e 's,@LIBEXECDIR@,${libexecdir},g' ${S}/swupdate-genconfig.conf.in > ${B}/swupdate-genconfig.conf
    echo "tegra-bootloader-capsule    ${STOMPERS_SWUPDATE_BOOTLOADER_VERSION}" > ${B}/sw-versions-thisslot
}

do_install() {
    install -d ${D}${sysconfdir}
    install -m 0644 ${B}/hwrevision ${D}${sysconfdir}/
    ln -s /run/swupdate/swupdate.cfg ${D}${sysconfdir}/swupdate.cfg
    ln -s /run/swupdate/sw-versions ${D}${sysconfdir}/sw-versions

    install -d ${D}${datadir}/swupdate
    install -m 0644 ${B}/swupdate.cfg.in ${B}/sw-versions-thisslot ${D}${datadir}/swupdate/

    install -d ${D}${libexecdir}/swupdate
    install -m 0755 ${B}/swupdate-genconfig ${D}${libexecdir}/swupdate/

    install -d ${D}${systemd_system_unitdir}/swupdate.service.d
    install -m 0644 ${B}/swupdate-genconfig.conf ${D}${systemd_system_unitdir}/swupdate.service.d/
}

FILES:${PN} += "${datadir}/swupdate ${systemd_system_unitdir}/swupdate.service.d"

PACKAGE_ARCH = "${MACHINE_ARCH}"
RDEPENDS:${PN} = "tegra-redundant-boot"
