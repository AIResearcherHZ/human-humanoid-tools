# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import numpy as np

from hhtools.retarget.robot_to_robot import (
    DEFAULT_SOURCE_FRAMERATE,
    _load_csv_trajectory,
    load_source_trajectory,
)


def test_load_csv_trajectory_accepts_motiondecode_running_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "motiondecode.csv"
    csv_path.write_text(
        (
            "root_pos_x(m),root_pos_y(m),root_pos_z(m),"
            "root_rot_w,root_rot_x,root_rot_y,root_rot_z,"
            "dof_left_knee_joint(rad),dof_right_knee_joint(rad)\n"
            "1.0,2.0,3.0,0.5,0.1,0.2,0.3,0.4,0.5\n"
        ),
        encoding="utf-8",
    )

    traj = _load_csv_trajectory(csv_path, fallback_dof_names=None)

    assert traj.joint_q.shape == (1, 9)
    np.testing.assert_allclose(
        traj.joint_q[0, :7],
        np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5], dtype=np.float32),
    )
    assert traj.dof_names == ("left_knee_joint", "right_knee_joint")
    assert traj.framerate == DEFAULT_SOURCE_FRAMERATE
    assert traj.meta["source_format"] == "motiondecode_running_csv"
    assert traj.meta["root_quat_format"] == "wxyz"


def test_motiondecode_source_fps_override(tmp_path: Path) -> None:
    csv_path = tmp_path / "motiondecode_120.csv"
    csv_path.write_text(
        (
            "root_pos_x(m),root_pos_y(m),root_pos_z(m),"
            "root_rot_w,root_rot_x,root_rot_y,root_rot_z,"
            "dof_left_knee_joint(rad)\n"
            "0,0,1,1,0,0,0,0.1\n"
            "0,0,1,1,0,0,0,0.2\n"
        ),
        encoding="utf-8",
    )

    traj = load_source_trajectory(csv_path, source_fps=120.0)
    assert traj.framerate == 120.0
    assert traj.joint_q.shape == (2, 8)
