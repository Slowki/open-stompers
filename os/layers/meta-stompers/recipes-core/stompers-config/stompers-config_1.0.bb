SUMMARY = "Stompers OS system configuration"
DESCRIPTION = "iwd network configuration, persistent journal, kernel panic logs in \
the journal and passwordless sudo for the sudo group."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://iwd-main.conf \
    file://journald-persistent.conf \
    file://pstore-journal.conf \
    file://sudoers-sudo-group \
"

S = "${UNPACKDIR}"

do_install() {
    install -d ${D}${sysconfdir}/iwd
    install -m 0644 ${S}/iwd-main.conf ${D}${sysconfdir}/iwd/main.conf

    install -d ${D}${nonarch_libdir}/systemd/journald.conf.d
    install -m 0644 ${S}/journald-persistent.conf ${D}${nonarch_libdir}/systemd/journald.conf.d/10-stompers-persistent.conf

    install -d ${D}${nonarch_libdir}/systemd/pstore.conf.d
    install -m 0644 ${S}/pstore-journal.conf ${D}${nonarch_libdir}/systemd/pstore.conf.d/10-stompers-journal.conf

    install -d -m 0750 ${D}${sysconfdir}/sudoers.d
    install -m 0440 ${S}/sudoers-sudo-group ${D}${sysconfdir}/sudoers.d/10-sudo-group
}

FILES:${PN} += "${nonarch_libdir}/systemd/journald.conf.d ${nonarch_libdir}/systemd/pstore.conf.d"

RDEPENDS:${PN} = "iwd sudo"
