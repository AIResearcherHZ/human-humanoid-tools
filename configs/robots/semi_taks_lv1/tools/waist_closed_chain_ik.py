from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

MODEL_XML = Path(__file__).resolve().parents[1] / "Semi_Taks_LV1.xml"
MOTOR_JOINTS = ("waist_right_motor_joint", "waist_left_motor_joint")
DRIVEN_JOINTS = ("waist_right_link_driven_joint", "waist_left_link_driven_joint")
LOOP_EQUALITIES = ("waist_right_parallel_loop", "waist_left_parallel_loop")


def _rotation_vector_quat(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.array((1.0, 0.0, 0.0, 0.0))
    return np.concatenate(
        (
            np.array((np.cos(angle * 0.5),)),
            np.sin(angle * 0.5) * vector / angle,
        )
    )


class WaistClosedChainIK:
    def __init__(self, model_xml: Path | str = MODEL_XML):
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        self.motor_qpos = np.array(
            [self._qpos_address(name) for name in MOTOR_JOINTS], dtype=np.intp
        )
        self.roll_qpos = self._qpos_address("waist_roll_joint")
        self.pitch_qpos = self._qpos_address("waist_pitch_joint")
        self.driven_qpos = np.array(
            [self._qpos_address(name) for name in DRIVEN_JOINTS], dtype=np.intp
        )
        self.equality_ids = tuple(
            self.model.equality(name).id for name in LOOP_EQUALITIES
        )
        self.motor_limits = self.model.jnt_range[
            [self.model.joint(name).id for name in MOTOR_JOINTS]
        ].copy()
        self._seed = np.zeros(8)

    def _qpos_address(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self.model.joint(name).id])

    def _residual(self, state: np.ndarray, roll: float, pitch: float) -> np.ndarray:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.roll_qpos] = roll
        self.data.qpos[self.pitch_qpos] = pitch
        self.data.qpos[self.motor_qpos] = state[:2]
        for address, rotation_vector in zip(self.driven_qpos, (state[2:5], state[5:8])):
            self.data.qpos[address : address + 4] = _rotation_vector_quat(
                rotation_vector
            )
        mujoco.mj_kinematics(self.model, self.data)
        residual = []
        for equality_id in self.equality_ids:
            body1 = self.model.eq_obj1id[equality_id]
            body2 = self.model.eq_obj2id[equality_id]
            point1 = (
                self.data.xpos[body1]
                + self.data.xmat[body1].reshape(3, 3)
                @ self.model.eq_data[equality_id, :3]
            )
            point2 = (
                self.data.xpos[body2]
                + self.data.xmat[body2].reshape(3, 3)
                @ self.model.eq_data[equality_id, 3:6]
            )
            residual.extend(point1 - point2)
        return np.asarray(residual)

    def inverse(self, roll: float, pitch: float) -> tuple[float, float]:
        if not np.all(
            np.array((roll, pitch))
            >= self.model.jnt_range[
                [
                    self.model.joint("waist_roll_joint").id,
                    self.model.joint("waist_pitch_joint").id,
                ],
                0,
            ]
        ) or not np.all(
            np.array((roll, pitch))
            <= self.model.jnt_range[
                [
                    self.model.joint("waist_roll_joint").id,
                    self.model.joint("waist_pitch_joint").id,
                ],
                1,
            ]
        ):
            raise ValueError(
                f"腰部目标超出主关节限位: roll={roll:.6f}, pitch={pitch:.6f}"
            )
        solution = least_squares(
            self._residual,
            self._seed,
            args=(roll, pitch),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        residual = float(np.linalg.norm(solution.fun))
        if residual > 1e-7:
            raise ValueError(
                f"腰部闭链目标不可达: roll={roll:.6f}, pitch={pitch:.6f}, residual={residual:.3e}"
            )
        motors = solution.x[:2]
        if np.any(motors < self.motor_limits[:, 0]) or np.any(
            motors > self.motor_limits[:, 1]
        ):
            raise ValueError(
                f"腰部闭链电机目标超限: right={motors[0]:.6f}, left={motors[1]:.6f}"
            )
        self._seed = solution.x
        return float(motors[0]), float(motors[1])


_DEFAULT_IK: WaistClosedChainIK | None = None


def cranks_from_pose(roll: float, pitch: float) -> tuple[float, float]:
    global _DEFAULT_IK
    if _DEFAULT_IK is None:
        _DEFAULT_IK = WaistClosedChainIK()
    return _DEFAULT_IK.inverse(roll, pitch)
