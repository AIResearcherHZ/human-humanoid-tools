# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""R2R yellow overlay must foot-ground even when wrists dip below the floor."""

from __future__ import annotations

import numpy as np
import pytest

from hhtools.retarget.retarget_result import RetargetedMotion
from hhtools.robot.loader import load_robot
from hhtools.robot.registry import get, refresh
from hhtools.web.serialize import (
    _scaled_overlay_foot_z,
    serialize_robot_trajectory,
)


@pytest.fixture(scope="module")
def g1_rp1():
    refresh()
    try:
        return (
            load_robot(get("g1"), compile_mjcf=False),
            load_robot(get("rp1"), compile_mjcf=False),
        )
    except KeyError:
        pytest.skip("g1/rp1 robots not registered in this environment")


def test_r2r_scaled_preview_snaps_feet_when_wrist_is_lowest(g1_rp1):
    from hhtools.web.server import (
        _align_scaled_preview_to_robot_playback,
        _compute_r2r_scaled_preview,
    )

    src, tgt = g1_rp1
    # Synthetic source FK motion: feet above wrists (get-up contact).
    from hhtools.core.hierarchy import Hierarchy
    from hhtools.core.motion import Motion

    names = [
        "hips", "spine", "chest", "neck", "head",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    ]
    parents = [-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 0, 14, 15]
    parent_names = [None] + [names[p] for p in parents[1:]]
    hier = Hierarchy(
        bone_names=names, parent_indices=parents, parent_names=parent_names,
    )
    pos = np.zeros((4, len(names), 3), dtype=np.float32)
    pos[:, names.index("hips"), 2] = 0.9
    pos[:, names.index("left_ankle"), 2] = 0.05
    pos[:, names.index("right_ankle"), 2] = 0.05
    # Wrist dips below the feet later in the clip.
    pos[:, names.index("left_wrist"), 2] = 0.8
    pos[-1, names.index("left_wrist"), 2] = -0.08
    quat = np.zeros((4, len(names), 4), dtype=np.float32)
    quat[..., 3] = 1.0
    motion = Motion(
        name="getup_like",
        hierarchy=hier,
        positions=pos,
        quaternions=quat,
        framerate=50.0,
    )

    calib = {
        j.name: 0.0 for j in tgt.actuated_joints if j.joint_type != "fixed"
    }
    scaled = _compute_r2r_scaled_preview(src, tgt, motion, calib)
    yellow_foot = _scaled_overlay_foot_z(scaled, 0)
    assert yellow_foot is not None
    assert abs(yellow_foot) < 0.02, f"yellow feet should rest near z=0, got {yellow_foot}"

    # Playback must not float the mesh sole just because a wrist was lower.
    dof_names = list(tgt.dof_names())
    F = 4
    root = np.zeros((F, 7), dtype=np.float32)
    root[:, 2] = 0.75
    root[:, 6] = 1.0
    dof = np.zeros((F, len(dof_names)), dtype=np.float32)
    ret = RetargetedMotion(
        name="fake",
        joint_q=np.concatenate([root, dof], axis=1),
        sample_rate=50.0,
        dof_names=tuple(dof_names),
        root_coord_count=7,
    )
    traj = serialize_robot_trajectory(
        tgt, ret, scaled_preview=scaled, ground_follow=False,
    )
    # Align-to-sole must keep feet near the grounded mesh sole (not +sole_depth).
    aligned = _align_scaled_preview_to_robot_playback(tgt, ret, scaled, traj)
    yellow_after = _scaled_overlay_foot_z(aligned, 0)
    assert yellow_after is not None
    assert abs(yellow_after) < 0.02
    assert np.asarray(aligned["positions"])[:, :, 2].min() < 0.03
