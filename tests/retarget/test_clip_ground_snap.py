"""Ground snapping for legged and mobile-base contact links."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hhtools.retarget.clip_ground_snap import (
    measure_clip_min_foot_world_z,
    snap_joint_q_clip_floor,
)


class _Graph:
    nodes_geometry = ()

    def __init__(self, z_by_link: dict[str, float]) -> None:
        self._z_by_link = z_by_link

    def get(self, link: str):
        transform = np.eye(4, dtype=np.float64)
        transform[2, 3] = self._z_by_link[link]
        return transform, None


class _Robot:
    def __init__(self) -> None:
        self.preset = SimpleNamespace(
            name="mobile_base",
            feet={},
            ik_map={"hips": "base_link"},
        )
        self.urdf = SimpleNamespace(
            scene=SimpleNamespace(
                graph=_Graph({
                    "wheel_fl": -0.38,
                    "wheel_fr": -0.40,
                    "wheel_rl": -0.39,
                    "wheel_rr": -0.37,
                }),
            ),
        )

    def link_names(self):
        return ("base_link", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")

    def dof_names(self):
        return ()

    def zero_configuration(self):
        return {}

    def apply_configuration(self, _config):
        return None


def test_clip_floor_snap_uses_wheel_link_origins_when_meshes_are_absent() -> None:
    robot = _Robot()
    q = np.array([[0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    min_z = measure_clip_min_foot_world_z(robot, q, upright_only=True)

    assert min_z == pytest.approx(0.02, abs=1e-6)


def test_clip_floor_snap_translates_mobile_base_root_only() -> None:
    robot = _Robot()
    q = np.array([[0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    snapped, delta = snap_joint_q_clip_floor(robot, q)

    assert delta == pytest.approx(0.02, abs=1e-6)
    np.testing.assert_allclose(snapped[0, 2], 0.40, atol=1e-6)
    np.testing.assert_allclose(snapped[0, 7:], q[0, 7:])
