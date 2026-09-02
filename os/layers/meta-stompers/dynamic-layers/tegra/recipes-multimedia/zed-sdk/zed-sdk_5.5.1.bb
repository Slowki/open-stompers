SUMMARY = "Stereolabs ZED SDK for USB ZED cameras on Jetson"
DESCRIPTION = "ZED SDK runtime (libsl_zed), headers, CMake config and udev rules \
for USB ZED cameras such as the ZED Mini. GMSL (ZED X) drivers, the GUI tools \
and the TensorRT-based AI module (libsl_ai) are left out."
HOMEPAGE = "https://www.stereolabs.com/developers/release"
LICENSE = "Stereolabs-ZED-SDK"
NO_GENERIC_LICENSE[Stereolabs-ZED-SDK] = "zed-sdk/doc/license/LICENSE.txt"
LIC_FILES_CHKSUM = "file://zed-sdk/doc/license/LICENSE.txt;md5=a4c250c2c0063adc0e65b397db136810"

# Proprietary EULA. Builds must accept it with LICENSE_FLAGS_ACCEPTED += "stereolabs".
LICENSE_FLAGS = "stereolabs"

ZED_L4T_VERSION = "39.2"
SRC_URI = " \
    https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/${@'.'.join(d.getVar('PV').split('.')[:2])}/ZED_SDK_Tegra_L4T${ZED_L4T_VERSION}_v${PV}.zstd.run;unpack=0 \
    file://zed-sdk-settings.conf \
"
SRC_URI[sha256sum] = "f6e7fc00bd026f83c6155660e25fa5be7d26798054c929ff3fe3053f1de4ed20"

COMPATIBLE_MACHINE = "(tegra)"

DEPENDS = " \
    libglvnd \
    libusb1 \
    libx11 \
    tegra-libraries-camera \
    tegra-libraries-cuda \
    tegra-libraries-multimedia \
    tegra-libraries-nvsci \
    v4l-utils \
    zstd-native \
"

S = "${UNPACKDIR}"
ZED_PAYLOAD = "${UNPACKDIR}/zed-sdk"

# The installer is a makeself archive; take the payload without running its script.
do_unpack[postfuncs] += "extract_zed_payload"
extract_zed_payload() {
    installer="${UNPACKDIR}/ZED_SDK_Tegra_L4T${ZED_L4T_VERSION}_v${PV}.zstd.run"
    skip=$(head -c 20000 "$installer" | grep -a -m1 '^skip=' | tr -dc 0-9)
    [ -n "$skip" ] || bbfatal "Cannot find the makeself payload offset in $installer"
    rm -rf ${ZED_PAYLOAD}
    mkdir -p ${ZED_PAYLOAD}
    tail -n +$(expr $skip + 1) "$installer" | zstd -cd | tar -x --no-same-owner -C ${ZED_PAYLOAD}
}

do_configure[noexec] = "1"
do_compile[noexec] = "1"

ZED_DIR = "/usr/local/zed"

do_install() {
    install -d ${D}${ZED_DIR}/lib ${D}${ZED_DIR}/include ${D}${ZED_DIR}/doc
    install -m 0755 ${ZED_PAYLOAD}/lib/libsl_zed.so ${D}${ZED_DIR}/lib/
    cp -R --no-preserve=ownership ${ZED_PAYLOAD}/include/* ${D}${ZED_DIR}/include/
    install -m 0644 ${ZED_PAYLOAD}/zed-config.cmake ${ZED_PAYLOAD}/zed-config-version.cmake ${D}${ZED_DIR}/
    install -m 0644 ${ZED_PAYLOAD}/doc/license/LICENSE.txt ${D}${ZED_DIR}/doc/

    install -d ${D}${sysconfdir}/ld.so.conf.d
    echo "${ZED_DIR}/lib" > ${D}${sysconfdir}/ld.so.conf.d/001-zed.conf

    # USB and hidraw access plus autosuspend off for USB cameras; the I2C rule is GMSL only.
    install -d ${D}${sysconfdir}/udev/rules.d
    grep -v 'i2c' ${ZED_PAYLOAD}/99-slabs.rules > ${D}${sysconfdir}/udev/rules.d/99-slabs.rules
    chmod 0644 ${D}${sysconfdir}/udev/rules.d/99-slabs.rules
    install -d ${D}${sysconfdir}/sysctl.d
    install -m 0644 ${ZED_PAYLOAD}/60-zed-buffers.conf ${D}${sysconfdir}/sysctl.d/

    # Per-camera calibration is downloaded on first use; keep it on the data partition.
    ln -s /data/zed/settings ${D}${ZED_DIR}/settings
    install -d ${D}${nonarch_libdir}/tmpfiles.d
    install -m 0644 ${UNPACKDIR}/zed-sdk-settings.conf ${D}${nonarch_libdir}/tmpfiles.d/
}

PACKAGES = "${PN} ${PN}-dev"
FILES:${PN} = " \
    ${ZED_DIR}/lib \
    ${ZED_DIR}/settings \
    ${ZED_DIR}/doc \
    ${sysconfdir} \
    ${nonarch_libdir}/tmpfiles.d \
"
FILES:${PN}-dev = " \
    ${ZED_DIR}/include \
    ${ZED_DIR}/zed-config.cmake \
    ${ZED_DIR}/zed-config-version.cmake \
"
# sl/Camera.hpp includes cuda.h and cuda_runtime.h.
RDEPENDS:${PN}-dev = "${PN} cuda-driver-dev cuda-cudart-dev"
RRECOMMENDS:${PN} = "kernel-module-uvcvideo"

# Prebuilt, unversioned and already stripped by Stereolabs.
INSANE_SKIP:${PN} = "already-stripped dev-so ldflags libdir"
INSANE_SKIP:${PN}-dev = "dev-elf"
INHIBIT_PACKAGE_STRIP = "1"
INHIBIT_SYSROOT_STRIP = "1"
INHIBIT_PACKAGE_DEBUG_SPLIT = "1"
