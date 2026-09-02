# btattach cannot set a baud rate with glibc 2.42 or newer, which the WQ9201S
# Bluetooth attach needs. Fixed upstream after bluez 5.87; drop this once the
# bluez5 recipe carries it.
FILESEXTRAPATHS:prepend:rockchip := "${THISDIR}/files:"
SRC_URI:append:rockchip = " file://0001-btattach-Update-for-glibc-2.42-compatibility.patch"
