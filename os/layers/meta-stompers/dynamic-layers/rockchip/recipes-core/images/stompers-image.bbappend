# NVIDIA JetPack style USB device mode at 192.168.55.1.
IMAGE_INSTALL:append = " stompers-usb-gadget"

# Onboard Wi-Fi and Bluetooth: the ZERO 3W ships with either an AIC8800DS2 or a
# WQ9201S. Both drivers are installed and each Bluetooth attach checks for its chip.
IMAGE_INSTALL:append:radxa-zero-3w = " \
    aic8800 \
    aic8800-bt \
    bluez5 \
    kernel-module-hci-uart \
    wq9201s \
    wq9201s-bt \
"
