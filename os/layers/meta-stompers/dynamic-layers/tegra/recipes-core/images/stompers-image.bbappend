# NVIDIA-style USB device mode at 192.168.55.1, with a device-mode fallback
# for when the fusb301 Type-C controller does not probe.
IMAGE_INSTALL:append = " l4t-usb-device-mode stompers-usb-device-role"

# ZED Mini and other USB ZED cameras.
IMAGE_INSTALL:append = " kernel-module-uvcvideo zed-sdk"
