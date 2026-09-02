#!/bin/bash -e
# Build Stompers OS images and cross-compilation SDKs with kas-container.

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
platforms=(jetson-orin-nx radxa-zero-3w)

usage() {
  cat << EOF
Usage: $(basename "$0") PLATFORM COMMAND [ARGUMENTS...]

Platforms:
  jetson-orin-nx  Jetson Orin NX 16GB on the Orin Nano developer kit carrier.
  radxa-zero-3w   Radxa ZERO 3W.

Commands:
  image   Build the flashable image and the swupdate (.swu) update bundle.
  sdk     Build the cross-compilation SDK installer.
  shell   Open a bitbake shell in the build container.
  lock    Pin all layer revisions in kas/PLATFORM.lock.yml.

Outputs land in build/PLATFORM/tmp/deploy.
EOF
}

platform="${1:-}"
command="${2:-}"
if [[ ! " ${platforms[*]} " == *" ${platform} "* ]] || [[ -z "${command}" ]]; then
  usage > /dev/stderr
  exit 1
fi
shift 2

export KAS_WORK_DIR="${KAS_WORK_DIR:-${script_directory}/build}"
export KAS_BUILD_DIR="${KAS_BUILD_DIR:-${KAS_WORK_DIR}/${platform}}"
export DL_DIR="${DL_DIR:-${KAS_WORK_DIR}/downloads}"
export SSTATE_DIR="${SSTATE_DIR:-${KAS_WORK_DIR}/sstate-cache}"
export BB_HASHSERVE_DB_DIR="${BB_HASHSERVE_DB_DIR:-${SSTATE_DIR}}"
mkdir -p "${KAS_WORK_DIR}" "${KAS_BUILD_DIR}" "${DL_DIR}" "${SSTATE_DIR}"

kas_container="${script_directory}/kas-container"
kas_configuration="${script_directory}/kas/${platform}.yml"
kas_sdk_configuration="${kas_configuration}:${script_directory}/kas/sdk.yml"

case "${command}" in
  image)
    exec "${kas_container}" build "${kas_configuration}" "$@"
    ;;
  sdk)
    exec "${kas_container}" build "${kas_sdk_configuration}" "$@"
    ;;
  shell)
    exec "${kas_container}" shell "${kas_configuration}" "$@"
    ;;
  lock)
    exec "${kas_container}" lock --update "${kas_configuration}" "$@"
    ;;
  *)
    usage > /dev/stderr
    exit 1
    ;;
esac
