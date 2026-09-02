SUMMARY = "Debugging and diagnostic tools for the Stompers OS image"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COREBASE}/meta/COPYING.MIT;md5=3da9cfbcb788c80a0384361b4de20420"

inherit packagegroup

RDEPENDS:${PN} = " \
    bind-utils \
    ethtool \
    htop \
    iftop \
    iperf3 \
    iputils \
    iotop \
    lsof \
    mtr \
    netcat \
    perf \
    strace \
    sysstat \
    tcpdump \
    usbutils \
"
