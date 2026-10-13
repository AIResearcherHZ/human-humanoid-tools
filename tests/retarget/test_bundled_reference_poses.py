"""Bundled zero-frame reference BVHs must ship inside the hhtools package.

Regression for issue 13: the desktop installer installs hhtools from a wheel,
and when these files lived outside the package the lookup silently returned
None, so LAFAN calibration fell back to a synthetic bind rest.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import tomllib

import pytest

from hhtools.retarget.newton_basic.rest_pose import (
    _BUNDLED_REFERENCE_BVH,
    bundled_reference_bvh_path,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("reference", sorted(_BUNDLED_REFERENCE_BVH))
def test_bundled_reference_resolves_inside_package(reference: str) -> None:
    path = bundled_reference_bvh_path(reference)
    assert path is not None, reference
    pkg_root = Path(str(resources.files("hhtools"))).resolve()
    assert pkg_root in path.parents


def test_reference_poses_declared_as_package_data() -> None:
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    globs = cfg["tool"]["setuptools"]["package-data"]["hhtools"]
    pkg = REPO / "hhtools"
    shipped = {p for g in globs for p in pkg.glob(g)}
    for name in _BUNDLED_REFERENCE_BVH.values():
        assert pkg / "data" / "reference_poses" / name in shipped, name


def test_lafan_calibration_reference_uses_bundled_bvh() -> None:
    from hhtools.retarget.calibration import reference as ref

    ref._lafan17_from_zero_bvh.cache_clear()
    assert ref._lafan17_from_zero_bvh() is not None
