from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "taks_level1", ROOT / "taks_level1" / "libs" / "SDK"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
import mujoco
import numpy as np

SCENE_XML = Path(__file__).with_name("scene_Semi_Taks_LV1.xml")
CAN_POLL_INTERVAL = 0.02
LEADER = "DM3507"
TITLE = "real2sim 主机遥操作臂 (semi.png, 8 电机)"
JOINTS: dict[int, str] = {
    0x31: "right_shoulder_pitch_joint",
    0x32: "right_shoulder_roll_joint",
    0x33: "right_shoulder_yaw_joint",
    0x34: "right_elbow_joint",
    0x39: "left_shoulder_pitch_joint",
    0x3A: "left_shoulder_roll_joint",
    0x3B: "left_shoulder_yaw_joint",
    0x3C: "left_elbow_joint",
}
MOTOR_TYPE: dict[int, str] = {mid: LEADER for mid in JOINTS}
DEFAULT_CAN_MAP: dict[str, list[int]] = {
    "can3": [0x31, 0x32, 0x33, 0x34],
    "can4": [0x39, 0x3A, 0x3B, 0x3C],
}


@dataclass(frozen=True)
class MotorSpec:
    motor_id: int
    joint: str
    bus: str
    motor_type: str


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
    id_to_bus = {mid: bus for bus, ids in can_map.items() for mid in ids}
    return [
        MotorSpec(mid, joint, id_to_bus[mid], MOTOR_TYPE[mid])
        for mid, joint in JOINTS.items()
        if mid in id_to_bus
    ]


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

    def run(self):
        try:
            from libs.drivers.DM_CANFD import DM_Motor_Type, Motor, MotorControl

            self._mc = MotorControl(self.interface, silent=True)
            for s in self.specs:
                mt = getattr(DM_Motor_Type, s.motor_type, DM_Motor_Type.DM4340P)
                motor = Motor(mt, s.motor_id, s.motor_id + 0x80)
                self._motors[s.motor_id] = motor
                self._mc.add_motor(motor)
            for m in self._motors.values():
                self._mc.enable(m)
            for m in self._motors.values():
                self._mc.control_mit(m, 0.0, 0.0, 0.0, 0.0, 0.0)
            while not self._stop_event.is_set():
                kd = self.kd
                for m in self._motors.values():
                    self._mc.control_mit(m, 0.0, kd, 0.0, 0.0, 0.0)
                self.cache.update(
                    {
                        mid: (
                            m.get_status()[0],
                            m.get_status()[1],
                            m.get_status()[2],
                            m.get_status()[3] < 0.1,
                        )
                        for mid, m in self._motors.items()
                    }
                )
                self._stop_event.wait(CAN_POLL_INTERVAL)
        except Exception as exc:
            print(f"{self.interface} 不可用，已跳过：{exc}")
        finally:
            self._safe_close()

    def _safe_close(self) -> None:
        if self._mc is None:
            return
        try:
            for m in self._motors.values():
                self._mc.disable(m)
        except Exception:
            pass
        try:
            self._mc.close()
        except Exception:
            pass

    def stop(self) -> None:
        self._stop_event.set()


def configure_graphics():
    import glfw

    if sys.platform != "linux":
        return
    if os.environ.get("DISPLAY"):
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    if os.path.exists("/proc/driver/nvidia/version"):
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        if os.environ["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia" and mujoco.get_mju_user_warning() is None:
            mujoco.set_mju_user_warning(_graphics_warning)


def _graphics_warning(message):
    if message == "OpenGL error 0x502 in or before mjr_makeContext":
        return
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def run_simulation(scene, rate, headless):
    import time
    from contextlib import nullcontext
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if not headless:
        configure_graphics()
    context = nullcontext(None) if headless else viewer.launch_passive(model, data)
    substeps = max(1, round(1.0 / (rate * model.opt.timestep)))
    try:
        with context as handle:
            while handle is None or handle.is_running():
                started = time.perf_counter()
                mujoco.mj_step(model, data, nstep=substeps)
                if handle is not None:
                    handle.sync()
                delay = 1.0 / rate - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass


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
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    act_id = {}
    for s in specs:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, s.joint)
        if aid < 0:
            raise SystemExit(f"模型缺少执行器: {s.joint} (电机 0x{s.motor_id:02X})")
        act_id[s.motor_id] = aid
    act_qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    rpy_joints = [
        ("腰", ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")),
        (
            "右腕",
            (
                "right_wrist_yaw_joint",
                "right_wrist_pitch_joint",
                "right_wrist_roll_joint",
            ),
        ),
        (
            "左腕",
            ("left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"),
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
            (
                f"{label}YPR=[{', '.join((f'{data.qpos[a]:+.3f}' for a in adrs))}] rad"
                for label, adrs in rpy_qadr.items()
            )
        )

    print(f"=== {title} ===")
    by_bus: dict = {}
    for s in specs:
        by_bus.setdefault(s.bus, []).append(s)
    for bus in sorted(by_bus):
        ids = ", ".join(f"0x{s.motor_id:02X}" for s in by_bus[bus])
        print(f"  {bus}: {len(by_bus[bus])} 电机  [{ids}]")
    if list_only:
        for s in specs:
            print(f"  0x{s.motor_id:02X} {s.bus:5s} {s.motor_type:8s} -> {s.joint}")
        unassigned = [mid for mid in JOINTS if mid not in act_id]
        if unassigned:
            print(
                "  未分配总线(保持零位): " + ", ".join(f"0x{m:02X}" for m in unassigned)
            )
        return
    from libs.drivers.rate_limiter import RateLimiter, perf_counter

    cache = MotorDataCache()
    readers = [CanReader(bus, group, cache) for bus, group in by_bus.items()]
    cmd = np.zeros(model.nu)
    substeps = max(1, round(1.0 / rate / model.opt.timestep))
    viewer = None
    t_first = None
    next_sync = 0.0
    next_log = perf_counter() + 2.0
    ticks = 0
    print(
        f"运行中 ({('阻尼' if damping else '自由')}模式, {rate:.0f}Hz). Ctrl+C 退出。"
    )
    try:
        rl = RateLimiter(rate)
        if not headless:
            from mujoco.viewer import launch_passive

            configure_graphics()
            viewer = launch_passive(model, data)
        for r in readers:
            r.set_damping(damping)
            r.start()
        while viewer is None or viewer.is_running():
            snap = cache.snapshot()
            online = 0
            for s in specs:
                d = snap.get(s.motor_id)
                if d and d[3]:
                    aid = act_id[s.motor_id]
                    cmd[aid] = d[0]
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
            mujoco.mj_step(model, data, nstep=substeps)
            if viewer is not None and now >= next_sync:
                viewer.sync()
                next_sync = now + 1.0 / 60.0
            ticks += 1
            if now >= next_log:
                pos_rad = data.ctrl[act_id[specs[0].motor_id]] if specs else 0.0
                print(
                    f"[{title.split()[0]}] {ticks / (now - (next_log - 2.0)):.0f}Hz online={online}/{len(specs)} r_shldrP={pos_rad:+.3f} rad {rpy_text()}"
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
            if r.ident is not None:
                r.join(timeout=0.5)
        if viewer is not None:
            viewer.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=TITLE)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sim", dest="real", action="store_false", help="仿真模式（默认）")
    mode.add_argument("--real", action="store_true", help="连接真机读取电机")
    p.set_defaults(real=False)
    p.add_argument("--scene", type=Path, default=SCENE_XML)
    p.add_argument("--rate", type=float, default=100.0, help="控制循环 Hz")
    p.add_argument(
        "--damping", action="store_true", help="阻尼模式 (kd=1) 而非自由", default=False
    )
    p.add_argument("--startup", type=float, default=1.0, help="上电缓启动秒数")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument(
        "--list", action="store_true", help="仅打印电机->关节映射", default=False
    )
    p.add_argument(
        "--cans",
        default=None,
        help="CAN 总线分配, 如 'can0:0x31-0x34;can2:0x39-0x3C'; 默认 DEFAULT_CAN_MAP",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.scene.exists():
        raise SystemExit(f"找不到场景文件: {args.scene}")
    if not np.isfinite(args.rate) or args.rate <= 0:
        raise SystemExit("rate 必须为正数")
    if not args.real and not args.list:
        run_simulation(args.scene, args.rate, args.headless)
        return
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
