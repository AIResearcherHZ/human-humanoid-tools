import os

import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.dirname(HERE)
M_XML = os.path.join(ASSET_DIR, "scene_Semi_Taks_LV1.xml")


def load():
    m = mujoco.MjModel.from_xml_path(M_XML)
    m.opt.gravity[:] = 0
    d = mujoco.MjData(m)
    return m, d


def jid(m, nm):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)


def connect_residuals(m, d, eq_ids):
    out = []
    for e in eq_ids:
        a1 = m.eq_data[e, 0:3]
        a2 = m.eq_data[e, 3:6]
        b1 = m.eq_obj1id[e]
        b2 = m.eq_obj2id[e]
        p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ a1
        p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ a2
        out.append(p1 - p2)
    return np.concatenate(out)


def rod_residuals(m, d, eq_ids):
    out = []
    for e in eq_ids:
        a1 = m.eq_data[e, 0:3]
        rod = m.eq_data[e, 3:6]
        b1 = m.eq_obj1id[e]
        b2 = m.eq_obj2id[e]
        p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ a1
        center = d.xpos[b2]
        out.append([np.linalg.norm(p1 - center) - np.linalg.norm(rod)])
    return np.concatenate(out)


def make_solver(motor_specs, eq_ids, free_joints, fixed, elim, residual_fn):
    m, d = load()
    motor_qadr = [(m.jnt_qposadr[jid(m, nm)], coef) for nm, coef in motor_specs]
    fixed_qadr = [(m.jnt_qposadr[jid(m, nm)], val) for nm, val in fixed]
    free_qadr = [m.jnt_qposadr[jid(m, nm)] for nm in free_joints]
    [(nm, m.jnt_qposadr[jid(m, nm)]) for nm, _ in fixed if False]
    elim_qadr = [
        (m.jnt_qposadr[jid(m, nm)], coef, m.jnt_qposadr[jid(m, src)])
        for nm, coef, src in elim
    ]

    def fk(x):
        d.qpos[:] = 0
        for adr, val in fixed_qadr:
            d.qpos[adr] = val
        for adr, coef in motor_qadr:
            d.qpos[adr] = coef * fk.th
        for adr, val in zip(free_qadr, x):
            d.qpos[adr] = val
        for adr, coef, src_adr in elim_qadr:
            d.qpos[adr] = coef * d.qpos[src_adr]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return residual_fn(m, d, eq_ids)

    fk.th = 0.0
    return fk, m, d


def scan(
    motor_specs,
    eq_ids,
    free_joints,
    fixed,
    elim,
    residual_fn,
    out_limits,
    theta_max=4.0,
    n=800,
    sign=1.0,
):
    fk, m, d = make_solver(motor_specs, eq_ids, free_joints, fixed, elim, residual_fn)
    out_qadr = {nm: (m.jnt_qposadr[jid(m, nm)], lim) for nm, lim in out_limits}
    ths = np.linspace(0, theta_max, n)
    x0 = np.zeros(len(free_joints))
    rows = []
    for th in ths:
        fk.th = sign * th
        sol = least_squares(fk, x0, method="lm", xtol=1e-14, ftol=1e-14, max_nfev=8000)
        r = float(np.linalg.norm(sol.fun))
        outs = {nm: float(d.qpos[adr]) for nm, (adr, _) in out_qadr.items()}
        limhit = None
        for nm, (adr, lim) in out_qadr.items():
            if abs(d.qpos[adr]) >= lim - 1e-4:
                limhit = nm
                break
        rows.append((sign * th, r, outs, limhit))
        if r > 5e-3 or limhit:
            break
        x0 = sol.x
    return rows


ARM_LOCK = [
    ("right_shoulder_pitch_joint", 0.0),
    ("right_shoulder_roll_joint", 0.0),
    ("right_shoulder_yaw_joint", 0.0),
    ("right_elbow_joint", 0.0),
    ("right_wrist_roll_joint", 0.0),
    ("right_arm_long_link_motor_joint", 0.0),
    ("right_arm_short_link_motor_joint", 0.0),
    ("left_shoulder_pitch_joint", 0.0),
    ("left_shoulder_roll_joint", 0.0),
    ("left_shoulder_yaw_joint", 0.0),
    ("left_elbow_joint", 0.0),
    ("left_wrist_roll_joint", 0.0),
    ("left_arm_long_link_motor_joint", 0.0),
    ("left_arm_short_link_motor_joint", 0.0),
]

NECK_FIXED = [("waist_yaw_joint", 0.0), ("head_yaw_joint", 0.0)] + ARM_LOCK
NECK_FREE = [
    "head_front_link_active_joint",
    "head_front_link_active_aux_joint",
    "head_rear_link_active_joint",
    "head_rear_link_active_aux_joint",
    "head_roll_joint",
    "head_pitch_joint",
]
NECK_ELIM = [
    ("head_front_link_bevel_gear_joint", -0.8, "head_pitch_joint"),
    ("head_rear_link_bevel_gear_joint", 0.8, "head_pitch_joint"),
]
NECK_EQS = [10, 11, 12, 13]
NECK_OUT_LIMITS = [("head_pitch_joint", 1.57)]

WAIST_FIXED = [("waist_yaw_joint", 0.0)] + ARM_LOCK
WAIST_FREE = ["waist_roll_joint", "waist_pitch_joint"]
WAIST_EQS = [0, 1]
WAIST_OUT_LIMITS = [("waist_roll_joint", 0.7854), ("waist_pitch_joint", 0.7854)]


def report(label, rows):
    ok = [r for r in rows if r[1] < 5e-3 and r[3] is None]
    if not ok:
        print(f"    {label}: NO valid config", flush=True)
        return None
    tmax = ok[-1][0]
    cause = "no-break"
    if len(rows) > len(ok):
        bad = rows[len(ok)]
        cause = (
            f"toggle(res={bad[1] * 1e3:.4f}mm)" if bad[1] > 5e-3 else f"limit[{bad[3]}]"
        )
    print(
        f"    {label}: range=±{abs(tmax):.5f} rad ({np.degrees(abs(tmax)):.3f} deg)  cause={cause}  last_out={ {k: round(v, 4) for k, v in ok[-1][2].items()} }",
        flush=True,
    )
    return abs(tmax)


if __name__ == "__main__":
    print("### NECK (head_front/rear link motor, bevel diff) ###", flush=True)
    neck_ranges = []
    for mode, specs in [
        (
            "pitch(front=-rear)",
            [
                ("head_front_link_motor_joint", 1.0),
                ("head_rear_link_motor_joint", -1.0),
            ],
        ),
        (
            "roll(front=+rear)",
            [("head_front_link_motor_joint", 1.0), ("head_rear_link_motor_joint", 1.0)],
        ),
    ]:
        print(f"  mode={mode}", flush=True)
        for sgn in (1.0, -1.0):
            rows = scan(
                specs,
                NECK_EQS,
                NECK_FREE,
                NECK_FIXED,
                NECK_ELIM,
                connect_residuals,
                NECK_OUT_LIMITS,
                sign=sgn,
            )
            r = report(f"sign={'+' if sgn > 0 else '-'}", rows)
            if r:
                neck_ranges.append(r)
    print(
        f"  => NECK motor range = min over modes = ±{min(neck_ranges):.5f} rad ({np.degrees(min(neck_ranges)):.3f} deg)"
        if neck_ranges
        else "  => no range found",
        flush=True,
    )

    print("\n### WAIST (left/right motor, 2RSS parallel) ###", flush=True)
    waist_ranges = []
    for mode, specs in [
        (
            "roll(R=+L)",
            [("waist_right_motor_joint", 1.0), ("waist_left_motor_joint", 1.0)],
        ),
        (
            "pitch(R=-L)",
            [("waist_right_motor_joint", 1.0), ("waist_left_motor_joint", -1.0)],
        ),
    ]:
        print(f"  mode={mode}", flush=True)
        for sgn in (1.0, -1.0):
            rows = scan(
                specs,
                WAIST_EQS,
                WAIST_FREE,
                WAIST_FIXED,
                [],
                rod_residuals,
                WAIST_OUT_LIMITS,
                sign=sgn,
            )
            r = report(f"sign={'+' if sgn > 0 else '-'}", rows)
            if r:
                waist_ranges.append(r)
    print(
        f"  => WAIST motor range = min over modes = ±{min(waist_ranges):.5f} rad ({np.degrees(min(waist_ranges)):.3f} deg)"
        if waist_ranges
        else "  => no range found",
        flush=True,
    )
