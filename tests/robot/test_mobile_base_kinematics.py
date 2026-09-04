"""Wheel odometry synthesis for mobile-base retarget outputs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hhtools.robot.mobile_base import apply_mobile_base_kinematics

_DOF_NAMES = (
    "base_x", "base_y", "base_yaw",
    "wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint",
    "lift_mid_joint", "lift_joint",
)


def _mobile_urdf() -> str:
    wheel_specs = (
        ("fl", 0.21, 0.30, 1.5708),
        ("fr", 0.21, -0.30, -1.5708),
        ("rl", -0.21, 0.30, -1.5708),
        ("rr", -0.21, -0.30, 1.5708),
    )
    wheels = []
    for suffix, x, y, roller_rpy in wheel_specs:
        wheels.append(f"""
  <joint name="wheel_{suffix}_joint" type="continuous">
    <parent link="chassis"/><child link="wheel_{suffix}"/>
    <origin xyz="{x} {y} 0.0762"/><axis xyz="0 1 0"/>
  </joint>
  <link name="wheel_{suffix}">
    <visual name="roller_{suffix}"><origin rpy="0 0 {roller_rpy}"/>
      <geometry><cylinder radius="0.016" length="0.06"/></geometry>
    </visual>
    <collision><geometry><cylinder radius="0.0762" length="0.06"/></geometry></collision>
  </link>""")
    return f"""<robot name="mobile">
  <link name="world"/><link name="base_x_link"/><link name="base_y_link"/>
  <link name="chassis"/><link name="lift_carriage"/>
  <joint name="base_x" type="prismatic">
    <parent link="world"/><child link="base_x_link"/><axis xyz="1 0 0"/>
  </joint>
  <joint name="base_y" type="prismatic">
    <parent link="base_x_link"/><child link="base_y_link"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="base_yaw" type="continuous">
    <parent link="base_y_link"/><child link="chassis"/><axis xyz="0 0 1"/>
  </joint>
  {''.join(wheels)}
  <joint name="lift_mid_joint" type="prismatic">
    <parent link="chassis"/><child link="lift_mid"/><axis xyz="0 0 1"/>
    <mimic joint="lift_joint" multiplier="0.5" offset="0"/>
  </joint>
  <link name="lift_mid"/>
  <joint name="lift_joint" type="prismatic">
    <parent link="chassis"/><child link="lift_carriage"/><axis xyz="0 0 1"/>
  </joint>
</robot>"""


class _Model:
    def __init__(self, urdf_path) -> None:
        self.preset = SimpleNamespace(
            urdf_path=urdf_path,
            meta={},
            feet={},
        )
        self.actuated_joints = tuple(
            SimpleNamespace(
                name=name,
                joint_type=(
                    "continuous" if name.startswith("wheel_") or name == "base_yaw"
                    else "prismatic"
                ),
                child_link=name.removesuffix("_joint") if name.startswith("wheel_") else "",
                axis=(0.0, 1.0, 0.0) if name.startswith("wheel_") else (0.0, 0.0, 1.0),
                velocity_limit=14.1732 if name.startswith("wheel_") else None,
                limit_lower=0.0 if name == "lift_joint" else None,
                limit_upper=0.38 if name == "lift_joint" else None,
            )
            for name in _DOF_NAMES
        )

    def dof_names(self):
        return _DOF_NAMES


@pytest.fixture
def mobile_model(tmp_path):
    path = tmp_path / "mobile.urdf"
    path.write_text(_mobile_urdf(), encoding="utf-8")
    return _Model(path)


def _trajectory(frame_count: int) -> np.ndarray:
    q = np.zeros((frame_count, 7 + len(_DOF_NAMES)), dtype=np.float32)
    q[:, 6] = 1.0
    return q


def _dof_column(name: str) -> int:
    return 7 + _DOF_NAMES.index(name)


def test_planar_helpers_are_folded_into_upright_root_and_wheels(mobile_model) -> None:
    q = _trajectory(3)
    q[:, 2] = (0.02, 0.07, 0.12)
    q[:, _dof_column("base_x")] = (0.0, 0.0762, 0.1524)
    q[:, 3] = np.sin(0.1)
    q[:, 6] = np.cos(0.1)

    out, meta = apply_mobile_base_kinematics(
        mobile_model, q, sample_rate=10.0,
    )

    np.testing.assert_allclose(out[:, 0], (0.0, 0.0762, 0.1524), atol=1e-6)
    np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, 3:5], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, _dof_column("base_x")], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, _dof_column("lift_joint")], (0.02, 0.07, 0.12), atol=1e-6)
    np.testing.assert_allclose(
        out[:, _dof_column("lift_mid_joint")], (0.01, 0.035, 0.06), atol=1e-6,
    )
    for name in ("wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint"):
        np.testing.assert_allclose(out[:, _dof_column(name)], (0.0, 1.0, 2.0), atol=1e-5)
    assert meta["mobile_base_applied"] is True
    assert meta["mobile_base_drive"] == "mecanum"


def test_yaw_uses_opposite_left_and_right_wheel_speeds(mobile_model) -> None:
    q = _trajectory(2)
    q[1, _dof_column("base_yaw")] = 0.1

    out, _meta = apply_mobile_base_kinematics(
        mobile_model, q, sample_rate=10.0,
    )

    expected = 0.51 * 0.1 / 0.0762
    assert out[1, _dof_column("wheel_fl_joint")] == pytest.approx(-expected)
    assert out[1, _dof_column("wheel_rl_joint")] == pytest.approx(-expected)
    assert out[1, _dof_column("wheel_fr_joint")] == pytest.approx(expected)
    assert out[1, _dof_column("wheel_rr_joint")] == pytest.approx(expected)
    assert out[1, 5] == pytest.approx(np.sin(0.05))
    assert out[1, _dof_column("base_yaw")] == 0.0


def test_mecanum_lateral_motion_uses_diagonal_wheel_pairs(mobile_model) -> None:
    q = _trajectory(2)
    q[1, _dof_column("base_y")] = 0.0762

    out, _meta = apply_mobile_base_kinematics(
        mobile_model, q, sample_rate=10.0,
    )

    assert out[1, _dof_column("wheel_fl_joint")] == pytest.approx(-1.0)
    assert out[1, _dof_column("wheel_rr_joint")] == pytest.approx(-1.0)
    assert out[1, _dof_column("wheel_fr_joint")] == pytest.approx(1.0)
    assert out[1, _dof_column("wheel_rl_joint")] == pytest.approx(1.0)
