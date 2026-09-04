# SPDX-License-Identifier: Apache-2.0
"""Post-process retargeted trajectories for wheeled mobile bases.

Generic IK can solve both Newton's floating root and scalar planar helper
joints from a converted URDF.  A mobile robot then gets two overlapping base
motions while its wheel joints remain at rest.  This module consolidates that
motion into one upright floating-base pose and derives wheel rotation from
planar odometry.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

__all__ = [
    "MobileBaseKinematics",
    "WheelKinematics",
    "apply_mobile_base_kinematics",
    "resolve_mobile_base_kinematics",
]


@dataclass(frozen=True)
class WheelKinematics:
    joint_name: str
    x: float
    y: float
    z: float
    radius: float
    axis_sign: float = 1.0
    velocity_limit: float | None = None


@dataclass(frozen=True)
class MobileBaseKinematics:
    wheels: tuple[WheelKinematics, ...]
    drive: str
    x_joint: str | None
    y_joint: str | None
    yaw_joint: str | None
    lift_joint: str | None
    lift_mimics: tuple[tuple[str, float, float], ...] = ()
    ground_z: float = 0.0


def _mobile_base_block(preset) -> tuple[dict[str, Any], bool]:
    """Return ``(config, explicit)`` from robot preset metadata."""
    top = getattr(preset, "meta", {}).get("mobile_base")
    retarget = getattr(preset, "meta", {}).get("retarget")
    nested = retarget.get("mobile_base") if isinstance(retarget, dict) else None
    raw = nested if nested is not None else top
    if raw is False:
        return {"enabled": False}, True
    if isinstance(raw, dict):
        return dict(raw), True
    if raw is True:
        return {}, True
    return {}, False


def _xyz(text: str | None) -> tuple[float, float, float]:
    values = [float(v) for v in str(text or "0 0 0").split()]
    values.extend([0.0] * (3 - len(values)))
    return float(values[0]), float(values[1]), float(values[2])


def _wheel_radius(link_el: ET.Element | None) -> float | None:
    if link_el is None:
        return None
    for section in ("collision", "visual"):
        radii = [
            float(cylinder.get("radius"))
            for parent in link_el.findall(section)
            for cylinder in parent.findall("./geometry/cylinder")
            if cylinder.get("radius") is not None
        ]
        if radii:
            return max(radii)
    return None


def _joint_config_name(
    block: dict[str, Any], key: str, default: str, dof_set: set[str],
) -> str | None:
    planar = block.get("planar_joints")
    raw = planar.get(key) if isinstance(planar, dict) else block.get(f"{key}_joint")
    name = str(raw or default)
    return name if name in dof_set else None


def resolve_mobile_base_kinematics(
    model: URDFRobotModel,
) -> MobileBaseKinematics | None:
    """Infer planar helpers and wheel geometry from a robot URDF.

    Automatic activation is deliberately strict: the robot must contain all
    three conventional ``base_x/base_y/base_yaw`` helpers and at least two
    wheel joints.  Other robots can opt in through ``retarget.mobile_base``.
    """
    preset = model.preset
    block, explicit = _mobile_base_block(preset)
    if block.get("enabled", True) is False:
        return None
    urdf_path = getattr(preset, "urdf_path", None)
    if urdf_path is None:
        return None
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError):
        return None

    dof_names = tuple(model.dof_names())
    dof_set = set(dof_names)
    joints_info = {joint.name: joint for joint in model.actuated_joints}
    joint_elements = {
        str(el.get("name")): el
        for el in root.findall("joint")
        if el.get("name")
    }
    link_elements = {
        str(el.get("name")): el
        for el in root.findall("link")
        if el.get("name")
    }

    x_joint = _joint_config_name(block, "x", "base_x", dof_set)
    y_joint = _joint_config_name(block, "y", "base_y", dof_set)
    yaw_joint = _joint_config_name(block, "yaw", "base_yaw", dof_set)

    raw_wheels = block.get("wheel_joints")
    if isinstance(raw_wheels, str):
        wheel_names = [raw_wheels]
    elif isinstance(raw_wheels, (list, tuple)):
        wheel_names = [str(name) for name in raw_wheels]
    else:
        wheel_names = [
            name
            for name in dof_names
            if "wheel" in name.lower()
            and joints_info.get(name) is not None
            and joints_info[name].joint_type in ("continuous", "revolute")
        ]

    radius_cfg = block.get("wheel_radius")
    wheels: list[WheelKinematics] = []
    has_rollers = False
    for name in wheel_names:
        info = joints_info.get(name)
        joint_el = joint_elements.get(name)
        if info is None or joint_el is None:
            continue
        origin = joint_el.find("origin")
        x, y, z = _xyz(origin.get("xyz") if origin is not None else None)
        link_el = link_elements.get(info.child_link)
        if isinstance(radius_cfg, dict):
            radius_raw = radius_cfg.get(name)
        else:
            radius_raw = radius_cfg
        radius = float(radius_raw) if radius_raw is not None else _wheel_radius(link_el)
        if radius is None or radius <= 1e-6:
            continue
        axis = np.asarray(info.axis, dtype=np.float64)
        axis_sign = 1.0 if float(axis[1]) >= 0.0 else -1.0
        if link_el is not None:
            has_rollers = has_rollers or any(
                "roller" in str(visual.get("name") or "").lower()
                for visual in link_el.findall("visual")
            )
        wheels.append(WheelKinematics(
            joint_name=name,
            x=x,
            y=y,
            z=z,
            radius=float(radius),
            axis_sign=axis_sign,
            velocity_limit=(
                float(info.velocity_limit)
                if info.velocity_limit is not None and info.velocity_limit > 0.0
                else None
            ),
        ))

    auto_planar = x_joint is not None and y_joint is not None and yaw_joint is not None
    if len(wheels) < 2 or (not explicit and not auto_planar):
        return None

    drive = str(block.get("drive") or ("mecanum" if has_rollers else "differential"))
    if drive not in ("mecanum", "differential"):
        _log.warning("unknown mobile-base drive %r; using differential", drive)
        drive = "differential"

    lift_raw = block.get("lift_joint")
    lift_name = str(lift_raw or "lift_joint")
    lift_joint = lift_name if lift_name in dof_set else None
    lift_mimics: list[tuple[str, float, float]] = []
    if lift_joint is not None:
        for mimic_name, joint_el in joint_elements.items():
            mimic = joint_el.find("mimic")
            if (
                mimic_name not in dof_set
                or mimic is None
                or mimic.get("joint") != lift_joint
            ):
                continue
            lift_mimics.append((
                mimic_name,
                float(mimic.get("multiplier", "1")),
                float(mimic.get("offset", "0")),
            ))
    feet = dict(getattr(preset, "feet", None) or {})
    ground_z = float(block.get("ground_z", feet.get("ground_z", 0.0)))
    return MobileBaseKinematics(
        wheels=tuple(wheels),
        drive=drive,
        x_joint=x_joint,
        y_joint=y_joint,
        yaw_joint=yaw_joint,
        lift_joint=lift_joint,
        lift_mimics=tuple(lift_mimics),
        ground_z=ground_z,
    )


def _quat_to_rotmat(q_xyzw: NDArray[np.floating]) -> NDArray[np.float64]:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _yaw_from_xyzw(q_xyzw: NDArray[np.floating]) -> float:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-9:
        return 0.0
    x, y, z, w = q / norm
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _joint_limits(model: URDFRobotModel, name: str) -> tuple[float, float]:
    joint = next((j for j in model.actuated_joints if j.name == name), None)
    if joint is None:
        return -float("inf"), float("inf")
    lo = float(joint.limit_lower) if joint.limit_lower is not None else -float("inf")
    hi = float(joint.limit_upper) if joint.limit_upper is not None else float("inf")
    return lo, hi


def _wheel_increments(
    xy: NDArray[np.float64],
    yaw: NDArray[np.float64],
    wheels: tuple[WheelKinematics, ...],
    drive: str,
    sample_rate: float,
) -> tuple[NDArray[np.float64], int]:
    frame_count = int(xy.shape[0])
    increments = np.zeros((max(frame_count - 1, 0), len(wheels)), dtype=np.float64)
    if frame_count < 2:
        return increments, 0
    arm = max(abs(wheel.x) for wheel in wheels) + max(abs(wheel.y) for wheel in wheels)
    clipped = 0
    fps = max(float(sample_rate), 1e-6)
    for frame in range(1, frame_count):
        dxy = xy[frame] - xy[frame - 1]
        dyaw = float(yaw[frame] - yaw[frame - 1])
        mid_yaw = 0.5 * float(yaw[frame] + yaw[frame - 1])
        c, s = math.cos(mid_yaw), math.sin(mid_yaw)
        forward = c * float(dxy[0]) + s * float(dxy[1])
        lateral = -s * float(dxy[0]) + c * float(dxy[1])
        for wi, wheel in enumerate(wheels):
            if drive == "mecanum":
                lateral_sign = -1.0 if wheel.x * wheel.y >= 0.0 else 1.0
                side_sign = 1.0 if wheel.y >= 0.0 else -1.0
                travel = (
                    forward
                    + lateral_sign * lateral
                    - side_sign * arm * dyaw
                )
            else:
                side_sign = 1.0 if wheel.y >= 0.0 else -1.0
                travel = forward - side_sign * abs(wheel.y) * dyaw
            increments[frame - 1, wi] = wheel.axis_sign * travel / wheel.radius

        scale = 1.0
        for wi, wheel in enumerate(wheels):
            if wheel.velocity_limit is None:
                continue
            speed = abs(float(increments[frame - 1, wi])) * fps
            if speed > wheel.velocity_limit:
                scale = min(scale, wheel.velocity_limit / speed)
        if scale < 1.0:
            increments[frame - 1] *= scale
            clipped += 1
    return increments, clipped


def apply_mobile_base_kinematics(
    model: URDFRobotModel,
    joint_q: NDArray,
    *,
    sample_rate: float,
    root_coord_count: int = 7,
    dof_names: tuple[str, ...] | None = None,
) -> tuple[NDArray, dict[str, Any]]:
    """Consolidate planar motion and synthesize wheel angles for mobile bases."""
    spec = resolve_mobile_base_kinematics(model)
    q = np.asarray(joint_q)
    if spec is None or q.ndim != 2 or q.shape[0] == 0 or q.shape[1] < root_coord_count:
        return q, {}

    names = tuple(dof_names or model.dof_names())
    available = min(len(names), q.shape[1] - root_coord_count)
    index = {name: root_coord_count + i for i, name in enumerate(names[:available])}
    wheel_columns = [index.get(wheel.joint_name) for wheel in spec.wheels]
    if any(column is None for column in wheel_columns):
        return q, {}

    out = q.astype(np.float32, copy=True)
    frame_count = out.shape[0]
    base_x = (
        out[:, index[spec.x_joint]].astype(np.float64)
        if spec.x_joint in index else np.zeros(frame_count)
    )
    base_y = (
        out[:, index[spec.y_joint]].astype(np.float64)
        if spec.y_joint in index else np.zeros(frame_count)
    )
    base_yaw = (
        out[:, index[spec.yaw_joint]].astype(np.float64)
        if spec.yaw_joint in index else np.zeros(frame_count)
    )

    xy = np.empty((frame_count, 2), dtype=np.float64)
    source_root_z = out[:, 2].astype(np.float64).copy()
    planar_offset_z = np.zeros(frame_count, dtype=np.float64)
    root_yaw = np.empty(frame_count, dtype=np.float64)
    for frame in range(frame_count):
        rotation = _quat_to_rotmat(out[frame, 3:7])
        offset = rotation @ np.array([base_x[frame], base_y[frame], 0.0])
        xy[frame] = out[frame, :2].astype(np.float64) + offset[:2]
        planar_offset_z[frame] = float(offset[2])
        root_yaw[frame] = _yaw_from_xyzw(out[frame, 3:7]) + base_yaw[frame]
    yaw = np.unwrap(root_yaw)

    out[:, :2] = xy.astype(np.float32)
    out[:, 3] = 0.0
    out[:, 4] = 0.0
    out[:, 5] = np.sin(0.5 * yaw).astype(np.float32)
    out[:, 6] = np.cos(0.5 * yaw).astype(np.float32)
    for helper in (spec.x_joint, spec.y_joint, spec.yaw_joint):
        if helper in index:
            out[:, index[helper]] = 0.0

    contact_z = min(wheel.z - wheel.radius for wheel in spec.wheels)
    grounded_root_z = float(spec.ground_z - contact_z)
    if spec.lift_joint in index:
        lift_col = index[spec.lift_joint]
        lo, hi = _joint_limits(model, spec.lift_joint)
        transferred = (
            out[:, lift_col].astype(np.float64)
            + source_root_z
            + planar_offset_z
            - grounded_root_z
        )
        out[:, lift_col] = np.clip(transferred, lo, hi).astype(np.float32)
        for mimic_name, multiplier, offset in spec.lift_mimics:
            mimic_col = index.get(mimic_name)
            if mimic_col is not None:
                out[:, mimic_col] = out[:, lift_col] * multiplier + offset
    out[:, 2] = np.float32(grounded_root_z)

    increments, clipped_frames = _wheel_increments(
        xy, yaw, spec.wheels, spec.drive, sample_rate,
    )
    max_speed = 0.0
    if increments.size:
        max_speed = float(np.max(np.abs(increments))) * max(float(sample_rate), 1e-6)
    for wi, column in enumerate(wheel_columns):
        assert column is not None
        initial = float(out[0, column])
        out[0, column] = np.float32(initial)
        if frame_count > 1:
            out[1:, column] = (
                initial + np.cumsum(increments[:, wi])
            ).astype(np.float32)

    meta: dict[str, Any] = {
        "mobile_base_applied": True,
        "mobile_base_drive": spec.drive,
        "mobile_base_wheel_joints": [wheel.joint_name for wheel in spec.wheels],
        "mobile_base_wheel_max_speed_rad_s": max_speed,
        "mobile_base_wheel_speed_clipped_frames": int(clipped_frames),
        "mobile_base_grounded_root_z": grounded_root_z,
    }
    if spec.lift_joint is not None:
        meta["mobile_base_vertical_transfer_joint"] = spec.lift_joint
    _log.info(
        "mobile base: consolidated planar root, grounded z=%.4f, drive=%s, wheels=%s",
        grounded_root_z,
        spec.drive,
        meta["mobile_base_wheel_joints"],
    )
    return out, meta
