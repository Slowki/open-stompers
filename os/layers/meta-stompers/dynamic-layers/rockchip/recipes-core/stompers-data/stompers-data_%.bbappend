FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI += "file://50-data.conf"

do_install:append() {
    install -d ${D}${nonarch_libdir}/repart.d
    install -m 0644 ${S}/50-data.conf ${D}${nonarch_libdir}/repart.d/
}

FILES:${PN} += "${nonarch_libdir}/repart.d"
