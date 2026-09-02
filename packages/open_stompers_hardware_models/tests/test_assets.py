"""Tests for the asset cache, manifest integrity, and downloading."""

import hashlib
import os
import urllib.error
from dataclasses import replace
from pathlib import Path
from typing import Never

import pytest

from open_stompers_hardware_models import assets
from open_stompers_hardware_models.robots import mini_pi_plus
from open_stompers_hardware_models.robots.mini_pi_plus import MiniPiPlusVariant

NETWORK_ENV_VAR = "HARDWARE_MODELS_NETWORK"
TEST_VARIANT = MiniPiPlusVariant.TWELVE_LEG_EIGHT_ARM_WITHOUT_GRIPPERS_WITH_HEAD_WITHOUT_WAIST


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(assets.CACHE_ROOT_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    return tmp_path / "cache"


def test_cache_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(assets.CACHE_ROOT_ENV_VAR, str(tmp_path / "override"))
    assert assets.cache_root() == tmp_path / "override"


def test_cache_root_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(assets.CACHE_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert assets.cache_root() == tmp_path / "xdg" / assets.CACHE_SUBDIR


def test_cache_root_default_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(assets.CACHE_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert assets.cache_root() == tmp_path / ".cache" / assets.CACHE_SUBDIR


def test_resolve_missing_raises(cache_dir: Path) -> None:
    del cache_dir
    with pytest.raises(assets.AssetNotFoundError, match="hardware_models fetch"):
        assets.resolve("hardware_models/mini_pi_plus/missing/mesh.STL")


def test_resolve_returns_existing_file(cache_dir: Path) -> None:
    del cache_dir
    path = assets.resource_path("some/resource.STL")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"data")
    assert assets.resolve("some/resource.STL") == path


def test_fetch_downloads_and_caches(cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del cache_dir
    entry = mini_pi_plus.mesh_entries(TEST_VARIANT)[0]
    payload = b"fake mesh payload"
    entry = replace(entry, sha256=hashlib.sha256(payload).hexdigest())

    calls: list[str] = []
    monkeypatch.setattr(assets, "_read_url", lambda url: calls.append(url) or payload)

    paths = assets.fetch([entry])
    assert paths == [assets.resource_path(entry.resource)]
    assert paths[0].read_bytes() == payload
    assert calls == [entry.url]

    # Second call hits the cache and does not download again.
    assert assets.fetch([entry]) == paths
    assert calls == [entry.url]


def test_fetch_rejects_checksum_mismatch(cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del cache_dir
    entry = mini_pi_plus.mesh_entries(TEST_VARIANT)[0]
    monkeypatch.setattr(assets, "_read_url", lambda _url: b"corrupted bytes")

    with pytest.raises(assets.AssetChecksumError, match="Checksum mismatch"):
        assets.fetch([entry])
    assert not assets.resource_path(entry.resource).exists()


def test_fetch_replaces_corrupted_cache_file(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del cache_dir
    entry = mini_pi_plus.mesh_entries(TEST_VARIANT)[0]
    payload = b"good bytes"
    entry = replace(entry, sha256=hashlib.sha256(payload).hexdigest())

    dest = assets.resource_path(entry.resource)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"stale corrupted content")

    monkeypatch.setattr(assets, "_read_url", lambda _url: payload)
    paths = assets.fetch([entry])
    assert paths[0].read_bytes() == payload


def test_download_failure_raises(cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del cache_dir
    entry = mini_pi_plus.mesh_entries(TEST_VARIANT)[0]

    def failing(*_args: object, **_kwargs: object) -> Never:
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(assets.urllib.request, "urlopen", failing)
    monkeypatch.setattr(assets.time, "sleep", lambda _seconds: None)

    with pytest.raises(assets.AssetDownloadError, match="Failed to download"):
        assets.fetch([entry])


def test_resource_paths_follow_cache_layout() -> None:
    variant = TEST_VARIANT
    for filename in mini_pi_plus.MESH_MANIFESTS[variant]:
        resource = mini_pi_plus.get_cache_relative_path(variant, filename)
        assert resource == f"hardware_models/mini_pi_plus/{variant}/meshes/{filename}"
        assert mini_pi_plus.upstream_url(variant, filename) == (
            f"https://raw.githubusercontent.com/{mini_pi_plus.UPSTREAM_REPO}/"
            f"{mini_pi_plus.UPSTREAM_COMMIT}/ht_urdf/{variant}/meshes/{filename}"
        )


@pytest.mark.skipif(
    os.environ.get(NETWORK_ENV_VAR) != "1",
    reason=f"set {NETWORK_ENV_VAR}=1 to run network-dependent asset tests",
)
def test_ensure_meshes_downloads_all_variants(cache_dir: Path) -> None:
    del cache_dir
    paths = mini_pi_plus.ensure_meshes()
    assert len(paths) == sum(len(m) for m in mini_pi_plus.MESH_MANIFESTS.values())
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
