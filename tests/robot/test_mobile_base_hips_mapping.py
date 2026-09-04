"""Anatomical hips mapping for mobile robots with planar-base helper links."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hhtools.robot.foot_geometry import _ground_contact_links
from hhtools.robot.kinematics import (
    infer_ik_map_from_kinematics,
    repair_ik_map,
    validate_ik_map,
)
from hhtools.robot.scaffold import scaffold_preset

_MOBILE_UPPER_BODY_URDF = """<?xml version="1.0"?>
<robot name="mobile_upper_body">
  <link name="world"/>
  <link name="base_x_link"/>
  <joint name="base_x" type="prismatic">
    <parent link="world"/><child link="base_x_link"/><axis xyz="1 0 0"/>
    <limit lower="-10" upper="10" effort="1" velocity="1"/>
  </joint>
  <link name="base_y_link"/>
  <joint name="base_y" type="prismatic">
    <parent link="base_x_link"/><child link="base_y_link"/><axis xyz="0 1 0"/>
    <limit lower="-10" upper="10" effort="1" velocity="1"/>
  </joint>
  <link name="chassis"/>
  <joint name="base_yaw" type="continuous">
    <parent link="base_y_link"/><child link="chassis"/><axis xyz="0 0 1"/>
    <limit effort="1" velocity="1"/>
  </joint>
  <link name="lift_carriage"/>
  <joint name="lift_joint" type="prismatic">
    <parent link="chassis"/><child link="lift_carriage"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="base_link"/>
  <joint name="carriage_to_body" type="fixed">
    <parent link="lift_carriage"/><child link="base_link"/>
  </joint>
  <link name="waist_roll_link"/>
  <joint name="waist_roll_joint" type="revolute">
    <parent link="base_link"/><child link="waist_roll_link"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _write_mobile_urdf(tmp_path: Path) -> Path:
    urdf = tmp_path / "mobile_upper_body.urdf"
    urdf.write_text(_MOBILE_UPPER_BODY_URDF, encoding="utf-8")
    return urdf


def test_inference_prefers_anatomical_base_link_over_planar_helpers(tmp_path: Path) -> None:
    urdf = _write_mobile_urdf(tmp_path)

    inferred = infer_ik_map_from_kinematics(urdf)

    assert inferred["hips"] == "base_link"
    assert scaffold_preset(tmp_path, urdf).preset.ik_map["hips"] == "base_link"


def test_validation_repairs_planar_helper_used_as_hips(tmp_path: Path) -> None:
    urdf = _write_mobile_urdf(tmp_path)
    stale = {"hips": "base_x_link", "chest": "waist_roll_link"}

    issues = validate_ik_map(urdf, stale)
    repaired, changes = repair_ik_map(urdf, stale)

    assert any(issue.slot == "hips" and "planar-base" in issue.message for issue in issues)
    assert repaired["hips"] == "base_link"
    assert any("hips" in change and "base_link" in change for change in changes)


class _ContactModel:
    def __init__(self, *, feet=None, links=()):
        self.preset = SimpleNamespace(feet=feet or {}, ik_map={})
        self._links = tuple(links)

    def link_names(self):
        return self._links


def test_mobile_base_wheels_are_ground_contacts_without_leg_slots() -> None:
    model = _ContactModel(
        links=("world", "chassis", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"),
    )

    assert _ground_contact_links(model) == (
        "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr",
    )


def test_explicit_ground_contacts_override_wheel_name_inference() -> None:
    model = _ContactModel(
        feet={"ground_contact_links": ["caster_left", "caster_right"]},
        links=("wheel_fl", "caster_left", "caster_right"),
    )

    assert _ground_contact_links(model) == ("caster_left", "caster_right")
