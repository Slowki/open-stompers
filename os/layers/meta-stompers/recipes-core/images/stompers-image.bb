SUMMARY = "Stompers OS image"
DESCRIPTION = "Robot operating system image. Platform packages are added by \
the bbappends under dynamic-layers/."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COREBASE}/meta/COPYING.MIT;md5=3da9cfbcb788c80a0384361b4de20420"

inherit core-image extrausers

IMAGE_FEATURES += "ssh-server-openssh"

# The rootfs archive is the payload of stompers-image-swupdate.
IMAGE_FSTYPES:append = " tar.gz"

IMAGE_INSTALL = " \
    packagegroup-core-boot \
    ${CORE_IMAGE_EXTRA_INSTALL} \
    avahi-daemon \
    avahi-libnss-mdns \
    bash \
    iw \
    iwd \
    kernel-module-rtw88-8821cu \
    linux-firmware-rtl8821 \
    packagegroup-stompers-debug-tools \
    stompers-config \
    stompers-data \
    sudo \
    swupdate \
    swupdate-client \
    swupdate-progress \
    swupdate-www \
    vim \
    wireless-regdb-static \
"

# Default login: stomper / stomper. wheel grants iwctl access through iwd's D-Bus policy.
STOMPERS_USER_PASSWORD_HASH = "\$6\$stomperOS\$CZRBj2uHiNxf1fEWzOFTGWPw/WEeNx9RP//dTDyZafRI0jvZpBNNubRejbxboUMc09/aK9LlHK6SdzRnIoCy.0"
EXTRA_USERS_PARAMS = " \
    useradd --create-home --shell /bin/bash --groups sudo,wheel,video,audio,dialout,systemd-journal \
        --password '${STOMPERS_USER_PASSWORD_HASH}' stomper; \
"
