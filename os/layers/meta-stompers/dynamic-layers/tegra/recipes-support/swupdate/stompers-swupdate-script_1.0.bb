SUMMARY = "swupdate post-install script that stages the Tegra bootloader capsule"
DESCRIPTION = "Requests the UEFI capsule update on the next reboot. Applying \
the capsule switches the bootloader slot, which also switches the rootfs slot."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

require stompers-swupdate.inc

SRC_URI = "file://stompers-swupdate-script.lua.in"

S = "${UNPACKDIR}"
B = "${WORKDIR}/build"

inherit deploy

do_compile() {
    sed -e 's,@CAPSULE_INSTALL_PATH@,${STOMPERS_SWUPDATE_CAPSULE_INSTALL_PATH},g' \
        ${S}/stompers-swupdate-script.lua.in > ${B}/stompers-swupdate-script.lua
}

do_install[noexec] = "1"

do_deploy() {
    install -m 0644 ${B}/stompers-swupdate-script.lua ${DEPLOYDIR}/
}

addtask deploy after do_compile before do_build
