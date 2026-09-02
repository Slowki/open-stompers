SUMMARY = "Persistent data partition shared by both rootfs slots"
DESCRIPTION = "Formats the `data` partition on first boot, mounts it at /data \
and bind-mounts /home, /var/lib/iwd and the SSH host keys onto it so they \
survive A/B updates. Without a usable data partition they stay on the rootfs."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://stompers-data-format \
    file://stompers-data-setup \
    file://stompers-data-format.service \
    file://stompers-data-setup.service \
    file://data.mount \
    file://sshdgenkeys-data.conf \
"

S = "${UNPACKDIR}"

inherit systemd features_check

REQUIRED_DISTRO_FEATURES = "systemd"

SYSTEMD_SERVICE:${PN} = "data.mount stompers-data-setup.service"

do_install() {
    install -d ${D}${libexecdir}
    install -m 0755 ${S}/stompers-data-format ${D}${libexecdir}/
    install -m 0755 ${S}/stompers-data-setup ${D}${libexecdir}/

    install -d ${D}${systemd_system_unitdir}
    for unit in stompers-data-format.service stompers-data-setup.service; do
        sed -e 's,@LIBEXECDIR@,${libexecdir},g' ${S}/$unit > ${D}${systemd_system_unitdir}/$unit
        chmod 0644 ${D}${systemd_system_unitdir}/$unit
    done
    install -m 0644 ${S}/data.mount ${D}${systemd_system_unitdir}/

    install -d ${D}${systemd_system_unitdir}/sshdgenkeys.service.d
    install -m 0644 ${S}/sshdgenkeys-data.conf ${D}${systemd_system_unitdir}/sshdgenkeys.service.d/

    install -d ${D}/data
    install -d ${D}${sysconfdir}/ssh/hostkeys
}

FILES:${PN} += "/data ${sysconfdir}/ssh/hostkeys ${systemd_system_unitdir}"

RDEPENDS:${PN} = "busybox util-linux-blkid util-linux-mount e2fsprogs-mke2fs e2fsprogs-e2fsck"
