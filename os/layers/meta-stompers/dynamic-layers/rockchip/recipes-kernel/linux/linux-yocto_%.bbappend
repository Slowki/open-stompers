FILESEXTRAPATHS:prepend := "${THISDIR}/files:${STOMPERS_KERNEL_FRAGMENTS_DIR}:"

SRC_URI:append:rockchip = " \
    file://rtl8821cu.cfg \
    file://iwd.cfg \
    file://usb-gadget.cfg \
"

SRC_URI:append:radxa-zero-3w = " \
    file://0001-arm64-dts-rockchip-radxa-zero-3w-use-the-OTG-port-as-a-USB-device.patch \
    file://bluetooth-uart.cfg \
"
