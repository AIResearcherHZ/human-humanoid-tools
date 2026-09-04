#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.dirname(HERE)
ROBOT_XML = os.path.join(ASSET_DIR, "Semi_Taks_LV1.xml")
SCENE_XML = os.path.join(ASSET_DIR, "scene_Semi_Taks_LV1.xml")

SIDE = "right"
TCP_SITE = "right_tcp"
ARM_ACTUATORS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_arm_long_link_motor_joint",
    "right_arm_short_link_motor_joint",
]

ELBOW_POS_Y = 0.0356
ELBOW_POS_Z = 0.0552
ELBOW_YZ_LEN = math.hypot(ELBOW_POS_Y, ELBOW_POS_Z)
ELBOW_BASE_X = 0.003
BASE_ECC_DEG = math.degrees(math.atan2(ELBOW_BASE_X, ELBOW_YZ_LEN))

TASK_BOX = ((0.05, 0.55), (-0.55, 0.15), (-0.25, 0.45))
VOXEL = 0.05
ROT_SCALE = 0.25
EQ_TOL = 2e-3
TRACK_TOL = 0.05
SETTLE_STEPS = 1000
CLEARANCE_MAX = 0.08
CLEARANCE_TARGET = 0.02

METRICS = {
    "manip": ("manip_p10", 1.0),
    "accel": ("accel_vol_p10", 1.0),
    "torque": ("torque_util_p90", -1.0),
    "reach": ("reach_voxels", 1.0),
    "clearance": ("clearance_score", 1.0),
}


def make_scene(ecc_deg, workdir):
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    x_offset = ELBOW_YZ_LEN * math.tan(math.radians(ecc_deg))
    found = 0
    for body in root.iter("body"):
        name = body.get("name")
        if name == "right_elbow_link":
            body.set("pos", f"{x_offset:.9f} {-ELBOW_POS_Y:.9f} {-ELBOW_POS_Z:.9f}")
            found += 1
        elif name == "left_elbow_link":
            body.set("pos", f"{x_offset:.9f} {ELBOW_POS_Y:.9f} {-ELBOW_POS_Z:.9f}")
            found += 1
    if found != 2:
        raise RuntimeError(f"elbow bodies not found in {ROBOT_XML} (got {found})")
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("compiler element missing")
    compiler.set("meshdir", os.path.join(ASSET_DIR, "meshes") + os.sep)
    robot_name = f"_eccscan_robot_{ecc_deg:.6f}.xml"
    tree.write(
        os.path.join(workdir, robot_name), encoding="utf-8", xml_declaration=True
    )
    with open(SCENE_XML, "r", encoding="utf-8") as f:
        scene = f.read()
    tag = 'file="Semi_Taks_LV1.xml"'
    if tag not in scene:
        raise RuntimeError(f"include tag {tag} not found in {SCENE_XML}")
    scene_name = f"_eccscan_scene_{ecc_deg:.6f}.xml"
    with open(os.path.join(workdir, scene_name), "w", encoding="utf-8") as f:
        f.write(scene.replace(tag, f'file="{robot_name}"'))
    return os.path.join(workdir, scene_name)


def build(ecc_deg, workdir):
    m = mujoco.MjModel.from_xml_path(make_scene(ecc_deg, workdir))
    m.opt.gravity[:] = 0
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    return m


def dof_count(m, j):
    return {
        mujoco.mjtJoint.mjJNT_FREE: 6,
        mujoco.mjtJoint.mjJNT_BALL: 3,
    }.get(m.jnt_type[j], 1)


class ArmIndex:
    def __init__(self, m):
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self.site < 0:
            raise RuntimeError(f"site {TCP_SITE} missing")
        self.act = []
        for name in ARM_ACTUATORS:
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a < 0:
                raise RuntimeError(f"actuator {name} missing")
            self.act.append(a)
        self.act = np.asarray(self.act, dtype=int)
        self.ctrl_range = np.asarray(m.actuator_ctrlrange[self.act])
        self.torque_limit = np.abs(m.actuator_forcerange[self.act]).max(axis=1)
        self.act_qposadr = np.asarray(
            [m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in self.act], dtype=int
        )
        self.act_dofadr = np.asarray(
            [m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in self.act], dtype=int
        )

        arm_dofs = []
        arm_bodies = set()
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith(SIDE + "_"):
                adr = m.jnt_dofadr[j]
                arm_dofs.extend(range(adr, adr + dof_count(m, j)))
                arm_bodies.add(int(m.jnt_bodyid[j]))
        self.arm_dofs = np.asarray(sorted(arm_dofs), dtype=int)
        arm_set = set(self.arm_dofs.tolist())
        self.lock_dofs = np.asarray(
            [d for d in range(m.nv) if d not in arm_set], dtype=int
        )

        self.selection = np.zeros((len(self.act), m.nv))
        self.selection[np.arange(len(self.act)), self.act_dofadr] = 1.0

        self.arm_geoms = {
            g
            for g in range(m.ngeom)
            if m.geom_group[g] == 0 and int(m.geom_bodyid[g]) in arm_bodies
        }
        self.other_geoms = {
            g
            for g in range(m.ngeom)
            if m.geom_group[g] == 0 and int(m.geom_bodyid[g]) not in arm_bodies
        }
        self.shoulder_jid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, SIDE + "_shoulder_pitch_joint"
        )
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.signs = np.array(
            [
                [1 if (v >> i) & 1 else -1 for i in range(len(self.act))]
                for v in range(2 ** len(self.act))
            ],
            dtype=float,
        )


def settle_one(args):
    m, idx, ctrl, state0 = args
    d = mujoco.MjData(m)
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_setState(m, d, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    d.ctrl[idx.act] = ctrl
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(m, d)
        if np.linalg.norm(d.qvel) < 1e-3:
            break
    out = np.zeros(nstate)
    mujoco.mj_getState(m, d, out, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return out


def settle_batch(m, idx, ctrls, nthread):
    ref = mujoco.MjData(m)
    mujoco.mj_resetData(m, ref)
    mujoco.mj_forward(m, ref)
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    s0 = np.zeros(nstate)
    mujoco.mj_getState(m, ref, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    tasks = [(m, idx, ctrl, s0) for ctrl in ctrls]
    if nthread <= 1:
        return [settle_one(t) for t in tasks]
    with ThreadPoolExecutor(max_workers=nthread) as ex:
        return list(ex.map(settle_one, tasks))


def equality_jacobian(m, d):
    if d.nefc == 0:
        return np.zeros((0, m.nv))
    efc_j = np.asarray(d.efc_J).reshape(d.nefc, m.nv)
    mask = np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    return efc_j[mask]


def reduced_basis(m, d, idx):
    lock = np.zeros((len(idx.lock_dofs), m.nv))
    lock[np.arange(len(idx.lock_dofs)), idx.lock_dofs] = 1.0
    basis = null_space(np.vstack([equality_jacobian(m, d), lock]))
    if basis.shape[1] == 0:
        raise RuntimeError("empty constraint null space")
    return basis


def site_jacobian(m, d, idx):
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, idx.site)
    return jacp, jacr


def dense_mass(m, d):
    mass = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, mass)
    return mass


def accel_polytope_volume(jacp, basis, mass, idx):
    j_r = jacp @ basis
    m_r = basis.T @ mass @ basis
    gain = basis.T @ idx.selection.T
    transfer = j_r @ np.linalg.solve(m_r, gain)
    verts = (transfer @ (idx.signs * idx.torque_limit).T).T
    if np.linalg.matrix_rank(transfer, tol=1e-9) < 3:
        return 0.0
    try:
        return float(ConvexHull(verts).volume)
    except Exception:
        return 0.0


def torque_utilization(m, d, basis, idx):
    gain = basis.T @ idx.selection.T
    bias = basis.T @ d.qfrc_bias
    tau, *_ = np.linalg.lstsq(gain, bias, rcond=None)
    return float(np.max(np.abs(tau) / idx.torque_limit))


def min_clearance(m, d, idx):
    arm_set = set(idx.arm_geoms)
    other_set = set(idx.other_geoms)
    saved = m.geom_margin.copy()
    m.geom_margin[:] = CLEARANCE_MAX
    mujoco.mj_forward(m, d)
    m.geom_margin[:] = saved
    worst = CLEARANCE_MAX
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if idx.floor in (g1, g2):
            continue
        if g1 in arm_set and g2 in other_set or g2 in arm_set and g1 in other_set:
            worst = min(worst, float(con.dist))
    return worst


def equality_violation(m, d):
    worst = 0.0
    for e in range(m.neq):
        if not d.eq_active[e] or m.eq_type[e] != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        b1, b2 = m.eq_obj1id[e], m.eq_obj2id[e]
        p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[e, 0:3]
        p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[e, 3:6]
        worst = max(worst, float(np.linalg.norm(p1 - p2)))
    return worst


def in_task_box(p):
    return all(lo <= v <= hi for v, (lo, hi) in zip(p, TASK_BOX))


def evaluate(ecc_deg, ctrl_unit, workdir, nthread):
    m = build(ecc_deg, workdir)
    idx = ArmIndex(m)
    ctrls = qmc.scale(ctrl_unit, idx.ctrl_range[:, 0], idx.ctrl_range[:, 1])
    states = settle_batch(m, idx, ctrls, nthread)

    d = mujoco.MjData(m)
    manip, accel, torque, clear = [], [], [], []
    voxels = set()
    n_unconverged = 0
    for i, state in enumerate(states):
        mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(m, d)
        track = np.max(np.abs(d.qpos[idx.act_qposadr] - ctrls[i]))
        if track > TRACK_TOL or equality_violation(m, d) > EQ_TOL:
            n_unconverged += 1
            continue
        basis = reduced_basis(m, d, idx)
        jacp, jacr = site_jacobian(m, d, idx)
        task = np.vstack([jacp, ROT_SCALE * jacr]) @ basis
        sv = np.linalg.svd(task, compute_uv=False)
        manip.append(float(np.prod(sv)))
        accel.append(accel_polytope_volume(jacp, basis, dense_mass(m, d), idx))
        torque.append(torque_utilization(m, d, basis, idx))
        clear.append(min_clearance(m, d, idx))
        p = d.site_xpos[idx.site] - d.xanchor[idx.shoulder_jid]
        if in_task_box(p):
            voxels.add(tuple(np.floor(p / VOXEL).astype(int)))

    n_valid = len(manip)
    if n_valid == 0:
        raise RuntimeError(f"no valid sample at eccentricity {ecc_deg:.3f} deg")
    manip = np.asarray(manip)
    accel = np.asarray(accel)
    torque = np.asarray(torque)
    clear = np.asarray(clear)
    return {
        "eccentric_deg": float(ecc_deg),
        "samples": len(ctrls),
        "valid": n_valid,
        "unconverged": n_unconverged,
        "manip_p10": float(np.percentile(manip, 10)),
        "manip_mean": float(manip.mean()),
        "accel_vol_p10": float(np.percentile(accel, 10)),
        "accel_vol_mean": float(accel.mean()),
        "torque_util_p90": float(np.percentile(torque, 90)),
        "reach_voxels": float(len(voxels)),
        "clearance_score": float(np.mean(np.clip(clear / CLEARANCE_TARGET, 0.0, 1.0))),
        "clearance_min": float(clear.min()),
    }


def objective(row, baseline, weights):
    total = 0.0
    terms = {}
    for key, (field, sign) in METRICS.items():
        ratio = row[field] / max(baseline[field], 1e-12)
        terms[key] = sign * math.log(max(ratio, 1e-6))
        total += weights[key] * terms[key]
    row["log_ratio"] = terms
    row["score"] = total / sum(weights.values())
    return row["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nthread", type=int, default=os.cpu_count())
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--w-manip", type=float, default=0.3)
    ap.add_argument("--w-accel", type=float, default=0.25)
    ap.add_argument("--w-torque", type=float, default=0.2)
    ap.add_argument("--w-reach", type=float, default=0.15)
    ap.add_argument("--w-clearance", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.join(HERE, "elbow_eccentric_scan.json"))
    args = ap.parse_args()

    weights = {
        "manip": args.w_manip,
        "accel": args.w_accel,
        "torque": args.w_torque,
        "reach": args.w_reach,
        "clearance": args.w_clearance,
    }
    sampler = qmc.Sobol(d=len(ARM_ACTUATORS), scramble=True, seed=args.seed)
    ctrl_unit = sampler.random(args.samples)

    workdir = tempfile.mkdtemp(dir=ASSET_DIR, prefix="_eccscan_")
    cache = {}

    def run(ecc):
        key = round(float(ecc), 4)
        if key not in cache:
            row = evaluate(key, ctrl_unit, workdir, args.nthread)
            cache[key] = row
            print(
                f"ecc={key:6.2f}  valid={row['valid']:4d}  "
                f"manip_p10={row['manip_p10']:.3e}  accel_p10={row['accel_vol_p10']:.3e}  "
                f"tau_p90={row['torque_util_p90']:.3f}  vox={int(row['reach_voxels']):4d}  "
                f"clr={row['clearance_score']:.3f}",
                flush=True,
            )
        return cache[key]

    try:
        baseline = run(BASE_ECC_DEG)
        grid = [float(t) for t in np.arange(args.start, args.stop + 1e-9, args.step)]
        for ecc in grid:
            run(ecc)
        if args.refine:
            best_grid = max(
                (t for t in cache), key=lambda t: objective(cache[t], baseline, weights)
            )
            span = args.step
            res = minimize_scalar(
                lambda t: -objective(run(t), baseline, weights),
                bounds=(
                    max(args.start, best_grid - span),
                    min(args.stop, best_grid + span),
                ),
                method="bounded",
                options={"xatol": 0.05},
            )
            run(float(res.x))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    rows = [cache[k] for k in sorted(cache)]
    for row in rows:
        objective(row, baseline, weights)
    best = max(rows, key=lambda r: r["score"])

    print("\necc_deg    score    manip   accel   torque   reach   clearance")
    for r in rows:
        t = r["log_ratio"]
        print(
            f"{r['eccentric_deg']:8.2f}  {r['score']:+.4f}  {t['manip']:+.3f}  "
            f"{t['accel']:+.3f}  {t['torque']:+.3f}  {t['reach']:+.3f}  {t['clearance']:+.3f}"
        )
    print(f"\nbaseline {baseline['eccentric_deg']:.2f} deg (score 0 by definition)")
    print(f"best     {best['eccentric_deg']:.2f} deg -> score {best['score']:+.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_eccentric_deg": BASE_ECC_DEG,
                "weights": weights,
                "metrics": {k: v[0] for k, v in METRICS.items()},
                "task_box_shoulder_frame": TASK_BOX,
                "voxel": VOXEL,
                "rot_scale": ROT_SCALE,
                "clearance_target": CLEARANCE_TARGET,
                "settle_steps": SETTLE_STEPS,
                "samples_per_eccentric": args.samples,
                "rows": rows,
                "best_eccentric_deg": best["eccentric_deg"],
            },
            f,
            indent=2,
        )
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
