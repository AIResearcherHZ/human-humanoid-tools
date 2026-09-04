from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "backend", ROOT / "backend" / "libs" / "SDK"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import mujoco
import numpy as np
from libs.drivers.rate_limiter import perf_counter, sleep

SCENE_XML = Path(__file__).with_name("scene_Semi_Taks_LV1.xml")
CAN_POLL_INTERVAL = 0.02

SHOULDER = "RP50H"
WRIST = "RP40C"
WAIST = "RP70H"
HEAD = "RP40C"

TITLE = "real2sim 从机整机 (all.png, 22 电机)"

JOINTS: dict[tuple[str, int], str] = {
    ("can0", 1): "right_shoulder_pitch_joint",
    ("can0", 2): "right_shoulder_roll_joint",
    ("can0", 3): "right_shoulder_yaw_joint",
    ("can0", 4): "right_elbow_joint",
    ("can0", 5): "right_wrist_roll_joint",
    ("can0", 6): "right_arm_long_link_motor_joint",
    ("can0", 7): "right_arm_short_link_motor_joint",
    ("can1", 1): "left_shoulder_pitch_joint",
    ("can1", 2): "left_shoulder_roll_joint",
    ("can1", 3): "left_shoulder_yaw_joint",
    ("can1", 4): "left_elbow_joint",
    ("can1", 5): "left_wrist_roll_joint",
    ("can1", 6): "left_arm_long_link_motor_joint",
    ("can1", 7): "left_arm_short_link_motor_joint",
    ("can2", 1): "waist_yaw_joint",
    ("can2", 2): "waist_right_motor_joint",
    ("can2", 3): "waist_left_motor_joint",
    ("can2", 4): "head_roll_joint",
    ("can2", 5): "head_pitch_joint",
    ("can2", 6): "head_yaw_joint",
}

MOTOR_TYPE: dict[tuple[str, int], str] = {
    **{
        (bus, did): SHOULDER
        for bus, ids in (("can0", range(1, 5)), ("can1", range(1, 5)))
        for did in ids
    },
    **{
        (bus, did): WRIST
        for bus, ids in (("can0", range(5, 8)), ("can1", range(5, 8)))
        for did in ids
    },
    **{("can2", did): WAIST for did in (1, 2, 3)},
    **{("can2", did): HEAD for did in (4, 5, 6)},
}

MOTOR_SIGN: dict[tuple[str, int], float] = {}

DEFAULT_CAN_MAP: dict[str, list[int]] = {
    "can0": [1, 2, 3, 4, 5, 6, 7],
    "can1": [1, 2, 3, 4, 5, 6, 7],
    "can2": [1, 2, 3, 4, 5, 6],
}


@dataclass(frozen=True)
class MotorSpec:
    bus: str
    device_id: int
    joint: str
    motor_type: str
    sign: float = 1.0
    scale: float = 1.0


def expand_ids(ids: str) -> list[int]:
    out: list[int] = []
    for tok in ids.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a, 0), int(b, 0) + 1))
        else:
            out.append(int(tok, 0))
    return out


def parse_can_map(spec, default: dict[str, list[int]]) -> dict[str, list[int]]:
    if not spec:
        return {bus: list(ids) for bus, ids in default.items()}
    result: dict[str, list[int]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        bus, _, ids = item.partition(":")
        result[bus.strip()] = expand_ids(ids)
    return result


def build_specs(can_map: dict[str, list[int]]) -> list[MotorSpec]:
    can_set = {(bus, did) for bus, ids in can_map.items() for did in ids}
    specs = [
        MotorSpec(
            bus,
            device_id,
            joint,
            MOTOR_TYPE[(bus, device_id)],
            MOTOR_SIGN.get((bus, device_id), 1.0),
        )
        for (bus, device_id), joint in JOINTS.items()
        if (bus, device_id) in can_set
    ]
    for bus in can_map:
        device_ids = [s.device_id for s in specs if s.bus == bus]
        if len(device_ids) > 8 or len(device_ids) != len(set(device_ids)):
            raise SystemExit(f"{bus} 的 EYou RP 设备 ID 必须唯一且位于 1..8")
    return specs


class MotorDataCache:
    __slots__ = ("_lock", "_snap")

    def __init__(self):
        self._snap: dict = {}
        self._lock = threading.Lock()

    def update(self, batch: dict) -> None:
        with self._lock:
            self._snap.update(batch)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)


def _can_up(interface: str) -> bool:
    path = f"/sys/class/net/{interface}"
    if not os.path.exists(path):
        return False
    try:
        with open(f"{path}/operstate") as f:
            return f.read().strip() in ("up", "unknown")
    except OSError:
        return True


class CanReader(threading.Thread):
    def __init__(self, interface: str, specs: list, cache: MotorDataCache):
        super().__init__(daemon=True)
        self.interface = interface
        self.specs = specs
        self.cache = cache
        self.kd = 0.0
        self._stop_event = threading.Event()
        self._mc = None
        self._motors: dict = {}

    def set_damping(self, on: bool) -> None:
        self.kd = 1.0 if on else 0.0

    def run(self) -> None:
        if not _can_up(self.interface):
            print(f"⚠️  CAN {self.interface} 不在线, 跳过 ({len(self.specs)} 电机)")
            return
        from libs.drivers.EYou_RP_CANFD import (
            EYouRp_Motor_Type,
            EYouRpCanfdControl,
            Motor,
        )

        try:
            motors = []
            for s in self.specs:
                motor = Motor(getattr(EYouRp_Motor_Type, s.motor_type), s.device_id)
                self._motors[s.device_id] = motor
                motors.append(motor)
            self._mc = EYouRpCanfdControl(self.interface, motors, silent=True)
        except Exception as exc:
            print(f"CAN {self.interface} 初始化失败: {exc}")
            return
        try:
            self._mc.enable()
        except Exception as exc:
            print(f"CAN {self.interface} 使能失败: {exc}")
            self._safe_close()
            return
        self._mc.controlMIT(
            [(m, 0.0, 0.0, 0.0, 0.0, 0.0) for m in self._motors.values()]
        )
        try:
            while not self._stop_event.is_set():
                kd = self.kd
                self._mc.controlMIT(
                    [(m, 0.0, kd, 0.0, 0.0, 0.0) for m in self._motors.values()]
                )
                self.cache.update(
                    {
                        (self.interface, did): (
                            m.getPosition(),
                            m.getVelocity(),
                            m.getTorque(),
                            m.getFeedbackAge() < 0.1,
                        )
                        for did, m in self._motors.items()
                    }
                )
                self._stop_event.wait(CAN_POLL_INTERVAL)
        finally:
            self._safe_close()

    def _safe_close(self) -> None:
        if self._mc is None:
            return
        try:
            self._mc.disable()
        except Exception:
            pass
        try:
            self._mc.close()
        except Exception:
            pass

    def stop(self) -> None:
        self._stop_event.set()


class RateLimiter:
    __slots__ = ("_next", "period")

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self._next = perf_counter() + self.period

    def sleep(self) -> None:
        now = perf_counter()
        wait = self._next - now
        if wait > 0:
            sleep(wait)
            self._next += self.period
        else:
            self._next = now + self.period


def run(
    specs: list,
    title: str,
    scene: Path,
    rate: float,
    damping: bool,
    startup: float,
    headless: bool,
    list_only: bool,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    act_id, sign, scale = {}, {}, {}
    for s in specs:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, s.joint)
        if aid < 0:
            raise SystemExit(
                f"模型缺少执行器: {s.joint} (电机 {s.bus}:ID{s.device_id})"
            )
        key = (s.bus, s.device_id)
        act_id[key] = aid
        sign[key] = s.sign
        scale[key] = s.scale
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    act_qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]

    rpy_joints = [
        ("腰", ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")),
        ("头", ("head_roll_joint", "head_pitch_joint", "head_yaw_joint")),
        (
            "右腕",
            (
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ),
        ),
        (
            "左腕",
            ("left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"),
        ),
    ]
    rpy_qadr = {}
    for label, names in rpy_joints:
        adrs = []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise SystemExit(f"模型缺少关节: {name}")
            adrs.append(model.jnt_qposadr[jid])
        rpy_qadr[label] = adrs

    def rpy_text() -> str:
        return " ".join(
            f"{label}RPY=[{', '.join(f'{np.degrees(data.qpos[a]):+.1f}' for a in adrs)}]°"
            for label, adrs in rpy_qadr.items()
        )

    print(f"=== {title} ===")
    by_bus: dict = {}
    for s in specs:
        by_bus.setdefault(s.bus, []).append(s)
    for bus in sorted(by_bus):
        ids = ", ".join(f"{s.bus}:ID{s.device_id}" for s in by_bus[bus])
        print(f"  {bus}: {len(by_bus[bus])} 电机  [{ids}]")
    if list_only:
        for s in specs:
            print(
                f"  {s.bus}:ID{s.device_id} {s.motor_type:8s} "
                f"sign={s.sign:+.0f} -> {s.joint}"
            )
        unassigned = [key for key in JOINTS if key not in act_id]
        if unassigned:
            print(
                "  未分配总线(保持零位): "
                + ", ".join(f"{bus}:ID{did}" for bus, did in unassigned)
            )
        return

    cache = MotorDataCache()
    readers = [CanReader(bus, group, cache) for bus, group in by_bus.items()]
    for r in readers:
        r.set_damping(damping)
        r.start()

    cmd = np.zeros(model.nu)
    substeps = max(1, round((1.0 / rate) / model.opt.timestep))

    viewer = None
    if not headless:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=True
        )

    rl = RateLimiter(rate)
    t_first = None
    next_sync = 0.0
    next_log = perf_counter() + 2.0
    ticks = 0
    print(f"运行中 ({'阻尼' if damping else '自由'}模式, {rate:.0f}Hz). Ctrl+C 退出。")
    try:
        while viewer is None or viewer.is_running():
            snap = cache.snapshot()
            online = 0
            for s in specs:
                key = (s.bus, s.device_id)
                d = snap.get(key)
                if d and d[3]:
                    aid = act_id[key]
                    cmd[aid] = min(
                        hi[aid],
                        max(lo[aid], sign[key] * scale[key] * d[0]),
                    )
                    online += 1
            now = perf_counter()
            if online and t_first is None:
                t_first = now
            f = 1.0
            if t_first is not None and startup > 0.0:
                f = min(1.0, (now - t_first) / startup)
            elif t_first is None:
                f = 0.0
            q_now = data.qpos[act_qadr]
            data.ctrl[:] = q_now + f * (cmd - q_now)
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            if viewer is not None and now >= next_sync:
                viewer.sync()
                next_sync = now + 1.0 / 60.0
            ticks += 1
            if now >= next_log:
                deg = np.degrees(data.ctrl[: min(4, model.nu)])
                print(
                    f"[{title.split()[0]}] {ticks / (now - (next_log - 2.0)):.0f}Hz "
                    f"online={online}/{len(specs)} ctrl[:4]={np.round(deg, 1).tolist()} "
                    f"{rpy_text()}"
                )
                ticks = 0
                next_log = now + 2.0
            rl.sleep()
    except KeyboardInterrupt:
        print("\n中断退出。")
    finally:
        for r in readers:
            r.stop()
        for r in readers:
            r.join(timeout=0.5)
        if viewer is not None:
            viewer.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--scene", type=Path, default=SCENE_XML)
    p.add_argument("--rate", type=float, default=200.0, help="控制循环 Hz")
    p.add_argument("--damping", action="store_true", help="阻尼模式 (kd=1) 而非自由")
    p.add_argument("--startup", type=float, default=1.0, help="上电缓启动秒数")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--list", action="store_true", help="仅打印电机->关节映射")
    p.add_argument(
        "--cans",
        default=None,
        help="CAN 总线分配, 如 'can0:1-7;can1:1-7;can2:1-6'; 默认 DEFAULT_CAN_MAP",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.scene.exists():
        raise SystemExit(f"找不到场景文件: {args.scene}")
    specs = build_specs(parse_can_map(args.cans, DEFAULT_CAN_MAP))
    run(
        specs,
        TITLE,
        args.scene,
        args.rate,
        args.damping,
        args.startup,
        args.headless,
        args.list,
    )


if __name__ == "__main__":
    main()
