# Load the Stompers USB gadget schema from stompers-usb-gadget.
do_install:append() {
    sed -i -e 's,^IMPORT_SCHEMAS=.*,IMPORT_SCHEMAS="stompers",' ${D}${sysconfdir}/default/usbgx
}
