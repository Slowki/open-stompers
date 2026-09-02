SUMMARY = "Firmware and board settings for the WuQi WQ9201S"
DESCRIPTION = "wq9201_*.bin images and wq_wlan_settings.ini from Seeed's \
wq9201s-wifi-bt-dkms package."
# Seeed's debian/copyright: proprietary firmware redistributed with vendor permission.
LICENSE = "WuQi-firmware"
NO_GENERIC_LICENSE[WuQi-firmware] = "debian/copyright"
LIC_FILES_CHKSUM = "file://debian/copyright;md5=3727876b7206ead678f0a134cddb542b"

SRCREV = "1f4bd1cbf6e844291a54ca55816c1ac827b6bd02"
SRC_URI = "git://github.com/Seeed-Studio/seeed_armbian_extension.git;protocol=https;branch=deb_source"
S = "${UNPACKDIR}/${BP}/wq9201s-wifi-bt-dkms"

inherit allarch

do_configure[noexec] = "1"
do_compile[noexec] = "1"

# wq_wlan_settings.ini selects a single antenna shared between Wi-Fi and Bluetooth.
do_install() {
    install -d ${D}${nonarch_base_libdir}/firmware
    install -m 0644 ${S}/firmwares/wq9201_*.bin ${S}/configs/wq_wlan_settings.ini \
        ${D}${nonarch_base_libdir}/firmware/
}

FILES:${PN} = "${nonarch_base_libdir}/firmware"
