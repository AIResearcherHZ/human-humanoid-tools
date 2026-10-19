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
SHOULDER = "RP50H"
WRIST = "RP40C"
WAIST = "RP70H"
HEAD = "RP40C"
TITLE = "real2sim 从机整机 (all.png, 19 电机)"
JOINTS: dict[tuple[str, int], str] = {
    ("right_arm", 0x01): "right_shoulder_pitch_joint",
    ("right_arm", 0x02): "right_shoulder_roll_joint",
    ("right_arm", 0x03): "right_shoulder_yaw_joint",
    ("right_arm", 0x04): "right_elbow_joint",
    ("right_arm", 0x05): "right_wrist_roll_joint",
    ("right_arm", 0x06): "right_wrist_yaw_joint",
    ("right_arm", 0x07): "right_wrist_pitch_joint",
    ("left_arm", 0x01): "left_shoulder_pitch_joint",
    ("left_arm", 0x02): "left_shoulder_roll_joint",
    ("left_arm", 0x03): "left_shoulder_yaw_joint",
    ("left_arm", 0x04): "left_elbow_joint",
    ("left_arm", 0x05): "left_wrist_roll_joint",
    ("left_arm", 0x06): "left_wrist_yaw_joint",
    ("left_arm", 0x07): "left_wrist_pitch_joint",
    ("waist", 0x01): "waist_yaw_joint",
    ("waist", 0x02): "waist_pitch_joint",
    ("waist", 0x03): "waist_roll_joint",
    ("head", 0x01): "head_pitch_joint",
    ("head", 0x02): "head_yaw_joint",
}
MOTOR_TYPE: dict[tuple[str, int], str] = {
    **{
        (group, did): SHOULDER
        for group, ids in (
            ("right_arm", range(0x01, 0x05)),
            ("left_arm", range(0x01, 0x05)),
        )
        for did in ids
    },
    **{
        (group, did): WRIST
        for group, ids in (
            ("right_arm", range(0x05, 0x08)),
            ("left_arm", range(0x05, 0x08)),
        )
        for did in ids
    },
    **{("waist", did): WAIST for did in (0x01, 0x02, 0x03)},
    **{("head", did): HEAD for did in (0x01, 0x02)},
}


@dataclass(frozen=True)
class MotorSpec:
    group: str
    bus: str
    device_id: int
    joint: str
    motor_type: str


def parse_can_map(spec, default: dict[str, str]) -> dict[str, str]:
    if not spec:
        return dict(default)
    result = dict(default)
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        group, sep, bus = item.partition(":")
        group = group.strip()
        bus = bus.strip()
        if not sep or group not in default or (not bus):
            raise SystemExit(f"无效 CAN 分组 {item!r}，格式应为 group:canX")
        result[group] = bus
    return result


def build_specs(can_map: dict[str, str]) -> list[MotorSpec]:
    specs = [
        MotorSpec(group, can_map[group], device_id, joint, MOTOR_TYPE[group, device_id])
        for (group, device_id), joint in JOINTS.items()
    ]
    for bus in set(can_map.values()):
        device_ids = [s.device_id for s in specs if s.bus == bus]
        if len(device_ids) > 8 or len(device_ids) != len(set(device_ids)):
            raise SystemExit(f"{bus} 上的 EYou RP 设备 ID 必须唯一且总数不能超过 8")
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


class CanReader(threading.Thread):
    def __init__(
        self, interface: str, specs: list, cache: MotorDataCache, address=None
    ):
        super().__init__(daemon=True)
        self.interface = interface
        self.address = address
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
            from libs.drivers.EYou_RP_CANFD import (
                EYouRp_Motor_Type,
                EYouRpCanfdControl,
                EYouRpCanfdGroup,
                Motor,
            )

            motors = []
            for s in self.specs:
                motor = Motor(
                    getattr(EYouRp_Motor_Type, s.motor_type),
                    s.device_id,
                    getattr(
                        EYouRpCanfdGroup,
                        {
                            "right_arm": "RightArm",
                            "left_arm": "LeftArm",
                            "waist": "Waist",
                            "head": "Head",
                        }[s.group],
                    ),
                )
                self._motors[s.device_id] = motor
                motors.append(motor)
            self._mc = EYouRpCanfdControl(
                self.interface, motors, silent=True, address=self.address
            )
            self._mc.enable()
            self._mc.control_mit(
                [(m, 0.0, 0.0, 0.0, 0.0, 0.0) for m in self._motors.values()]
            )
            while not self._stop_event.is_set():
                kd = self.kd
                self._mc.control_mit(
                    [(m, 0.0, kd, 0.0, 0.0, 0.0) for m in self._motors.values()]
                )
                self.cache.update(
                    {
                        (self.interface, did): (
                            m.get_status()[0],
                            m.get_status()[1],
                            m.get_status()[2],
                            m.get_status()[3] < 0.1,
                        )
                        for did, m in self._motors.items()
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
            self._mc.disable()
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
    address=None,
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
            raise SystemExit(
                f"模型缺少执行器: {s.joint} (电机 {s.bus}:ID 0x{s.device_id:02X})"
            )
        key = (s.bus, s.device_id)
        act_id[key] = aid
    act_qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    rpy_joints = [
        ("腰", ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")),
        ("头", ("head_pitch_joint", "head_yaw_joint")),
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
        ids = ", ".join(f"{s.group}:ID 0x{s.device_id:02X}" for s in by_bus[bus])
        print(f"  {bus}: {len(by_bus[bus])} 电机  [{ids}]")
    if list_only:
        for s in specs:
            print(
                f"  {s.group}->{s.bus}:ID 0x{s.device_id:02X} {s.motor_type:8s} -> {s.joint}"
            )
        return
    from libs.drivers.rate_limiter import RateLimiter, perf_counter

    cache = MotorDataCache()
    readers = [
        CanReader(bus, group, cache, address=address) for bus, group in by_bus.items()
    ]
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
                key = (s.bus, s.device_id)
                d = snap.get(key)
                if d and d[3]:
                    aid = act_id[key]
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
                print(
                    f"[{title.split()[0]}] {ticks / (now - (next_log - 2.0)):.0f}Hz online={online}/{len(specs)} ctrl[:4]={np.round(data.ctrl[: min(4, model.nu)], 3).tolist()} rad {rpy_text()}"
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
    p.add_argument(
        "--address",
        dest="address",
        default=None,
        help="远程 QUIC 地址 HOST[:PORT]，省略时使用本地 SHM，默认端口 5555",
    )
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
        help="分组到 CAN 的映射，如 'right_arm:can0;left_arm:can1;waist:can2;head:can3'",
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
    from backend.services.config import DEFAULT_CAN_MAP

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
        address=args.address,
    )


if __name__ == "__main__":
    main()
