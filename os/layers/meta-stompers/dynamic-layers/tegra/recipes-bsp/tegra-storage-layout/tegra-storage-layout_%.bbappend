FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI += "file://data-partition.xml"

# NVIDIA's external layout with a fill-to-end `data` partition after the rootfs slots.
PARTITION_FILE_EXTERNAL = "${B}/stompers-external-layout.xml"

do_compile:prepend() {
    awk -v snippet="${UNPACKDIR}/data-partition.xml" '
        /<partition name="secondary_gpt"/ { while ((getline line < snippet) > 0) print line }
        { print }
    ' "${STAGING_DATADIR}/l4t-storage-layout/${PARTITION_LAYOUT_EXTERNAL}" > "${PARTITION_FILE_EXTERNAL}"
    if ! grep -q '<partition name="data"' "${PARTITION_FILE_EXTERNAL}"; then
        bbfatal "Failed to add the data partition to ${PARTITION_LAYOUT_EXTERNAL}"
    fi
}
