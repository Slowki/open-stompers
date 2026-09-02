FILESEXTRAPATHS:prepend := "${THISDIR}/files:${STOMPERS_KERNEL_FRAGMENTS_DIR}:"

SRC_URI += " \
    file://rtl8821cu.cfg \
    file://iwd.cfg \
    file://zed-usb.cfg \
    file://lockup-debug.cfg \
"
