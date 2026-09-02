"""On-demand asset caching under the XDG cache directory.

Binary assets (meshes) are not stored in this repository. They are downloaded
on demand from their upstream sources into ``$XDG_CACHE_HOME/openstompers``
(falling back to ``~/.cache/openstompers``) and verified against SHA-256
checksums. The ``HARDWARE_MODELS_CACHE_ROOT`` environment variable overrides
the cache root entirely (useful for tests and CI).
"""

import hashlib
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CACHE_SUBDIR = "openstompers"
CACHE_ROOT_ENV_VAR = "HARDWARE_MODELS_CACHE_ROOT"
USER_AGENT = "openstompers-open_stompers_hardware_models"

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 0.5
_TIMEOUT_S = 120.0


class AssetError(RuntimeError):
    """Base error for asset resolution and downloading."""


class AssetNotFoundError(AssetError):
    """Raised when a resource is referenced but not present in the cache."""


class AssetChecksumError(AssetError):
    """Raised when a downloaded asset does not match its expected checksum."""


class AssetDownloadError(AssetError):
    """Raised when an asset cannot be downloaded."""


@dataclass(frozen=True, slots=True)
class AssetEntry:
    """A single downloadable asset.

    Attributes:
        resource: Cache-root-relative path under which the asset is stored and
            referenced from models.
        url: Upstream download URL.
        sha256: Expected SHA-256 hex digest of the file contents.
    """

    resource: str
    url: str
    sha256: str


def cache_root() -> Path:
    """Return the root directory used to cache downloaded assets."""
    if override := os.environ.get(CACHE_ROOT_ENV_VAR):
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        base = str(Path("~/.cache").expanduser())
    return Path(base) / CACHE_SUBDIR


def resource_path(resource: str) -> Path:
    """Return the cache path for a resource (whether or not it exists)."""
    return cache_root() / resource


def is_cached(resource: str) -> bool:
    """Return True if the resource exists in the cache with the expected name."""
    return resource_path(resource).is_file()


def resolve(resource: str) -> Path:
    """Resolve a cache-relative resource to an existing file.

    Raises:
        AssetNotFoundError: If the file is not present in the cache yet.
    """
    path = resource_path(resource)
    if not path.is_file():
        msg = (
            f"Asset '{resource}' is not present in the cache at '{path}'. "
            "Download it first with: hardware_models fetch"
        )
        raise AssetNotFoundError(msg)
    return path


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read_url(url: str) -> bytes:
    if urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise AssetDownloadError(f"Unsupported asset URL scheme: {url!r}")
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": USER_AGENT}
    )
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=_TIMEOUT_S
            ) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    msg = f"Failed to download '{url}'"
    raise AssetDownloadError(msg) from last_error


def fetch(entries: Iterable[AssetEntry]) -> list[Path]:
    """Download any missing or corrupted assets into the cache.

    Files already present with the expected checksum are left untouched.
    Returns the paths of all requested assets (downloaded or pre-existing).
    """
    paths: list[Path] = []
    for entry in entries:
        dest = resource_path(entry.resource)
        if dest.is_file() and sha256_of(dest) == entry.sha256:
            paths.append(dest)
            continue
        data = _read_url(entry.url)
        actual = hashlib.sha256(data).hexdigest()
        if actual != entry.sha256:
            msg = f"Checksum mismatch for '{entry.resource}': expected {entry.sha256}, got {actual}"
            raise AssetChecksumError(msg)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        partial.write_bytes(data)
        partial.replace(dest)
        paths.append(dest)
    return paths
