import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
LINKS_PAYLOAD = json.load(open(os.path.join(HERE, "links.json")))
LINKS = LINKS_PAYLOAD["links"]

O_CAD = np.array([0.0, -4.8, -383.3])
M_ROB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)


def to_robot(p, mirror=False):
    p = np.array(p, float)
    if mirror:
        p = p * np.array([-1, 1, 1])
    return (M_ROB @ (p - O_CAD)) / 1000.0


def O(link):
    return np.array(LINKS[link]["origin"])


def fmt(v, nd=8):
    out = []
    for x in np.atleast_1d(v):
        s = f"{x:.{nd}g}"
        if s == "-0":
            s = "0"
        out.append(s)
    return " ".join(out)


BALL_WR = to_robot([-10.5, -38.95, -226.0])
BALL_WL = to_robot([10.5, -38.95, -226.0])
PIN_LONG = to_robot([-179.0, -229.9, -208.22])
PIN_LONG_SECOND_LOWER = to_robot([-155.0, -130.4, -208.22])
PIN_LONG_SECOND = to_robot([-155.0, -229.9, -208.22])
PIN_SHORT = to_robot([-155.0, -229.9, -261.22])
PIN_SHORT_SECOND_LOWER = to_robot([-179.0, -177.4, -261.22])
PIN_SHORT_SECOND = to_robot([-179.0, -229.9, -261.22])
PIN_LONG_L = to_robot([179.0, -229.9, -208.22])
PIN_LONG_L_SECOND_LOWER = to_robot([155.0, -130.4, -208.22])
PIN_LONG_L_SECOND = to_robot([155.0, -229.9, -208.22])
PIN_SHORT_L = to_robot([155.0, -229.9, -261.22])
PIN_SHORT_L_SECOND_LOWER = to_robot([179.0, -177.4, -261.22])
PIN_SHORT_L_SECOND = to_robot([179.0, -229.9, -261.22])
PIN_HF = to_robot([10.77, -29.2, 134.36])
PIN_HF_SECOND_LOWER = to_robot([12.33, -29.2, 71.64])
PIN_HF_SECOND = to_robot([-11.17, -29.2, 124.64])
PIN_HR = to_robot([-11.17, 29.2, 134.36])
PIN_HR_SECOND_LOWER = to_robot([-12.73, 29.2, 71.64])
PIN_HR_SECOND = to_robot([10.77, 29.2, 124.64])
TCP_R = to_robot([-180.58, -303.91, -235.5])
TCP_L = to_robot([180.58, -303.91, -235.5])
WAIST_RIGHT_CRANK_RANGE = (-0.75, 1.1619)
WAIST_LEFT_CRANK_RANGE = (-1.1619, 0.75)

BODY_TREE = {
    "base_link": (None, None),
    "waist_yaw_link": (
        "base_link",
        {"axis": "0 0 1", "range": "-2.618 2.618", "cls": "torso_motor"},
    ),
    "waist_right_link_driven_link": (
        "waist_yaw_link",
        {"ball": True, "cls": "torso_motor"},
    ),
    "waist_left_link_driven_link": (
        "waist_yaw_link",
        {"ball": True, "cls": "torso_motor"},
    ),
    "waist_pitch_link": (
        "waist_yaw_link",
        {"axis": "0 1 0", "range": "-0.7854 0.7854", "cls": "torso_motor"},
    ),
    "waist_roll_link": (
        "waist_pitch_link",
        {"axis": "1 0 0", "range": "-0.7854 0.7854", "cls": "torso_motor"},
    ),
    "waist_right_motor_link": (
        "waist_roll_link",
        {
            "axis": "1 0 0",
            "range": f"{WAIST_RIGHT_CRANK_RANGE[0]:.8f} {WAIST_RIGHT_CRANK_RANGE[1]:.8f}",
            "cls": "torso_motor",
        },
    ),
    "waist_left_motor_link": (
        "waist_roll_link",
        {
            "axis": "1 0 0",
            "range": f"{WAIST_LEFT_CRANK_RANGE[0]:.8f} {WAIST_LEFT_CRANK_RANGE[1]:.8f}",
            "cls": "torso_motor",
        },
    ),
}

for side, sp_axis, roll_range in [
    ("right", "0 0.94832 -0.3173", "-2.2515 1.5882"),
    ("left", "0 0.94832 0.3173", "-1.5882 2.2515"),
]:
    s = side
    BODY_TREE.update(
        {
            f"{s}_shoulder_pitch_link": (
                "waist_roll_link",
                {"axis": sp_axis, "range": "-3.1415 1.0472", "cls": "shoulder_motor"},
            ),
            f"{s}_shoulder_roll_link": (
                f"{s}_shoulder_pitch_link",
                {"axis": "1 0 0", "range": roll_range, "cls": "shoulder_motor"},
            ),
            f"{s}_shoulder_yaw_link": (
                f"{s}_shoulder_roll_link",
                {"axis": "0 0 1", "range": "-2.618 2.618", "cls": "shoulder_motor"},
            ),
            f"{s}_elbow_link": (
                f"{s}_shoulder_yaw_link",
                {"axis": "0 1 0", "range": "-1.0472 1.5708", "cls": "shoulder_motor"},
            ),
            f"{s}_wrist_roll_link": (
                f"{s}_elbow_link",
                {"axis": "1 0 0", "range": "-2.67 2.67", "cls": "wrist_motor"},
            ),
            f"{s}_arm_long_link_motor_link": (
                f"{s}_wrist_roll_link",
                {"axis": "0 0 1", "range": "-1.2217 1.2217", "cls": "wrist_motor"},
            ),
            f"{s}_arm_long_link_active_link": (
                f"{s}_arm_long_link_motor_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_arm_long_link_active_aux_link": (
                f"{s}_arm_long_link_motor_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_arm_short_link_motor_link": (
                f"{s}_wrist_roll_link",
                {"axis": "0 0 1", "range": "-1.2217 1.2217", "cls": "wrist_motor"},
            ),
            f"{s}_arm_short_link_active_link": (
                f"{s}_arm_short_link_motor_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_arm_short_link_active_aux_link": (
                f"{s}_arm_short_link_motor_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_wrist_yaw_link": (
                f"{s}_wrist_roll_link",
                {"axis": "0 0 1", "range": "-1.5708 1.5708", "cls": "wrist_motor"},
            ),
            f"{s}_wrist_pitch_link": (
                f"{s}_wrist_yaw_link",
                {"axis": "0 1 0", "range": "-1.5708 1.5708", "cls": "wrist_motor"},
            ),
            f"{s}_arm_long_link_bevel_gear_link": (
                f"{s}_wrist_yaw_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_arm_short_link_bevel_gear_link": (
                f"{s}_wrist_yaw_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
        }
    )

BODY_TREE.update(
    {
        "head_roll_link": (
            "waist_roll_link",
            {"axis": "1 0 0", "free": True, "cls": "wrist_motor"},
        ),
        "head_pitch_link": (
            "head_roll_link",
            {"axis": "0 1 0", "range": "-1.57 1.57", "cls": "wrist_motor"},
        ),
        "head_yaw_link": (
            "head_pitch_link",
            {"axis": "0 0 1", "range": "-2.618 2.618", "cls": "wrist_motor"},
        ),
    }
)

JOINT_NAME = {
    link: link.replace("_link", "_joint") if not link.endswith("_link") else None
    for link in BODY_TREE
}
for link in BODY_TREE:
    if link == "base_link":
        continue
    if link.endswith("_motor_link"):
        JOINT_NAME[link] = link[: -len("_link")] + "_joint"
    else:
        JOINT_NAME[link] = link[: -len("_link")] + "_joint"

CHILDREN = {}
for link, (parent, _) in BODY_TREE.items():
    if parent:
        CHILDREN.setdefault(parent, []).append(link)

ORDER = [
    "waist_yaw_link",
    "waist_right_link_driven_link",
    "waist_left_link_driven_link",
    "waist_pitch_link",
    "waist_roll_link",
    "waist_right_motor_link",
    "waist_left_motor_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_arm_long_link_motor_link",
    "right_arm_long_link_active_link",
    "right_arm_long_link_active_aux_link",
    "right_arm_short_link_motor_link",
    "right_arm_short_link_active_link",
    "right_arm_short_link_active_aux_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_arm_long_link_bevel_gear_link",
    "right_arm_short_link_bevel_gear_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_arm_long_link_motor_link",
    "left_arm_long_link_active_link",
    "left_arm_long_link_active_aux_link",
    "left_arm_short_link_motor_link",
    "left_arm_short_link_active_link",
    "left_arm_short_link_active_aux_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_arm_long_link_bevel_gear_link",
    "left_arm_short_link_bevel_gear_link",
    "head_roll_link",
    "head_pitch_link",
    "head_yaw_link",
]
for link in CHILDREN:
    CHILDREN[link].sort(key=lambda l: ORDER.index(l))

SITES = {
    "right_elbow_link": [("right_elbow", "0 0 0", 'size="0.006"')],
    "left_elbow_link": [("left_elbow", "0 0 0", 'size="0.006"')],
    "right_wrist_pitch_link": [
        ("right_tcp", fmt(TCP_R - O("right_wrist_pitch_link"), 6), 'size="0.004"')
    ],
    "left_wrist_pitch_link": [
        ("left_tcp", fmt(TCP_L - O("left_wrist_pitch_link"), 6), 'size="0.004"')
    ],
}

CAMERAS = {
    "head_yaw_link": [
        ("head_left_camera", "0.08812307593 0.02999965544 0.05311159739"),
        ("head_right_camera", "0.08812307593 -0.03000034456 0.05311159739"),
    ],
}
CAMERA_XYAXES = "0 -1 0 0.1719291 0 0.985109326"
CAMERA_URDF_RPY = "0 0.172787596 0"
COLLISION_NAME_OVERRIDES = {
    "base_link": "pelvis",
    "waist_roll_link": "waist_roll",
    "right_shoulder_pitch_link": "right_shoulder_pitch",
    "right_shoulder_roll_link": "right_shoulder_roll",
    "right_shoulder_yaw_link": "right_shoulder_yaw",
    "right_elbow_link": "right_elbow",
    "right_wrist_roll_link": "right_wrist_roll",
    "right_wrist_pitch_link": "right_hand",
    "right_wrist_yaw_link": "right_wrist_yaw",
    "left_shoulder_pitch_link": "left_shoulder_pitch",
    "left_shoulder_roll_link": "left_shoulder_roll",
    "left_shoulder_yaw_link": "left_shoulder_yaw",
    "left_elbow_link": "left_elbow",
    "left_wrist_roll_link": "left_wrist_roll",
    "left_wrist_pitch_link": "left_hand",
    "left_wrist_yaw_link": "left_wrist_yaw",
    "head_pitch_link": "head",
}


def collision_attributes(link, index):
    if link == "base_link":
        contype, conaffinity = 1, 2
    elif link.startswith("waist_"):
        contype, conaffinity = 4, 2
    elif link.startswith("right_"):
        contype, conaffinity = 8, 50
    elif link.startswith("left_"):
        contype, conaffinity = 16, 42
    else:
        contype, conaffinity = 32, 26
    name = COLLISION_NAME_OVERRIDES.get(link, link) + "_collision"
    if index:
        name += f"_{index}"
    return name, contype, conaffinity


def inertial_xml(link, indent):
    d = LINKS[link]
    I = np.array(d["inertia_com"])
    com = np.array(d["com_link"])
    principal, axes = np.linalg.eigh(I)
    if np.linalg.det(axes) < 0.0:
        axes[:, 0] *= -1.0
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, axes.reshape(-1))
    return (
        f'{indent}<inertial pos="{fmt(com, 16)}" mass="{d["mass"]:.16g}" '
        f'diaginertia="{fmt(principal, 16)}" quat="{fmt(quat, 16)}" />'
    )


def body_xml(link, parent, lines, indent):
    pos = O(link) - (O(parent) if parent != "world" else 0)
    jn = JOINT_NAME[link]
    spec = BODY_TREE[link][1]
    lines.append(f'{indent}<body name="{link}" pos="{fmt(pos, 7)}">')
    lines.append(inertial_xml(link, indent + "  "))
    if spec is not None:
        cls = spec["cls"]
        if spec.get("ball"):
            lines.append(
                f'{indent}  <joint name="{jn}" class="{cls}" type="ball" limited="false" />'
            )
        elif spec.get("free"):
            lines.append(
                f'{indent}  <joint name="{jn}" class="{cls}" type="hinge" axis="{spec["axis"]}" limited="false" />'
            )
        else:
            lines.append(
                f'{indent}  <joint name="{jn}" class="{cls}" type="hinge" axis="{spec["axis"]}" range="{spec["range"]}" />'
            )
    for name, pos_s, extra in SITES.get(link, []):
        lines.append(f'{indent}  <site name="{name}" pos="{pos_s}" {extra} />')
    for name, pos_s in CAMERAS.get(link, []):
        lines.append(
            f'{indent}  <camera name="{name}" pos="{pos_s}" xyaxes="{CAMERA_XYAXES}" fovy="125" resolution="1920 1080" />'
        )
    for index, _ in enumerate(LINKS[link]["mesh_files"]):
        mesh_name = link if index == 0 else f"{link}__{index}"
        lines.append(f'{indent}  <geom type="mesh" mesh="{mesh_name}" />')
        collision_name, contype, conaffinity = collision_attributes(link, index)
        lines.append(
            f'{indent}  <geom name="{collision_name}" type="mesh" mesh="{mesh_name}" contype="{contype}" conaffinity="{conaffinity}" group="0" />'
        )
    for ch in CHILDREN.get(link, []):
        body_xml(ch, link, lines, indent + "  ")
    lines.append(f"{indent}</body>")


lines = []
lines.append("<?xml version='1.0' encoding='utf-8'?>")
lines.append('<mujoco model="Semi_Taks_LV1">')
lines.append('  <compiler angle="radian" meshdir="meshes/" autolimits="true" />')
lines.append(
    '  <option timestep="0.002" iterations="150" solver="Newton" tolerance="1e-10" integrator="implicit" gravity="0 0 -9.81" />'
)
lines.append("  <default>")
lines.append('    <joint limited="true" />')
lines.append(
    '    <geom contype="0" conaffinity="0" condim="1" group="1" density="0" friction="1 0.005 0.0001" />'
)
lines.append('    <equality solref="0.01 2" solimp="0.95 0.999 0.001 0.5 2" />')
lines.append('    <default class="torso_motor">')
lines.append('      <joint armature="0.00985321" damping="0.01" frictionloss="0.1" />')
lines.append("    </default>")
lines.append('    <default class="shoulder_motor">')
lines.append(
    '      <joint armature="0.0030254460886951994" damping="0.01" frictionloss="0.1" />'
)
lines.append("    </default>")
lines.append('    <default class="wrist_motor">')
lines.append('      <joint armature="0.00074568" damping="0.01" frictionloss="0.01" />')
lines.append("    </default>")
lines.append("  </default>")
lines.append("  <asset>")
for link in sorted(LINKS):
    for index, filename in enumerate(LINKS[link]["mesh_files"]):
        mesh_name = link if index == 0 else f"{link}__{index}"
        lines.append(f'    <mesh name="{mesh_name}" file="{filename}" />')
lines.append("  </asset>")
lines.append("  <worldbody>")
base_lines = []
lines.append('    <body name="base_link" pos="0 0 0">')
lines.append(inertial_xml("base_link", "      "))
for index, _ in enumerate(LINKS["base_link"]["mesh_files"]):
    mesh_name = "base_link" if index == 0 else f"base_link__{index}"
    collision_name, contype, conaffinity = collision_attributes("base_link", index)
    lines.append(f'      <geom type="mesh" mesh="{mesh_name}" />')
    lines.append(
        f'      <geom name="{collision_name}" type="mesh" mesh="{mesh_name}" contype="{contype}" conaffinity="{conaffinity}" group="0" />'
    )
sub = []
body_xml("waist_yaw_link", "base_link", sub, "      ")
lines += sub
lines.append("    </body>")
lines.append("  </worldbody>")
eq = []
eq.append(
    '    <connect name="waist_right_parallel_loop" body1="waist_right_motor_link" body2="waist_right_link_driven_link" anchor="0.0082 0.0321 0" />'
)
eq.append(
    '    <connect name="waist_left_parallel_loop" body1="waist_left_motor_link" body2="waist_left_link_driven_link" anchor="0.0082 -0.0321 0" />'
)
eq.append(
    f'    <connect name="right_arm_long_parallel_loop" body1="right_arm_long_link_active_link" body2="right_arm_long_link_bevel_gear_link" anchor="{fmt(PIN_LONG - O("right_arm_long_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="right_arm_long_parallel_second_loop" body1="right_arm_long_link_active_aux_link" body2="right_arm_long_link_bevel_gear_link" anchor="{fmt(PIN_LONG_SECOND - O("right_arm_long_link_active_aux_link"), 7)}" />'
)
eq.append(
    f'    <connect name="right_arm_short_parallel_loop" body1="right_arm_short_link_active_link" body2="right_arm_short_link_bevel_gear_link" anchor="{fmt(PIN_SHORT - O("right_arm_short_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="right_arm_short_parallel_second_loop" body1="right_arm_short_link_active_aux_link" body2="right_arm_short_link_bevel_gear_link" anchor="{fmt(PIN_SHORT_SECOND - O("right_arm_short_link_active_aux_link"), 7)}" />'
)
eq.append(
    f'    <connect name="left_arm_long_parallel_loop" body1="left_arm_long_link_active_link" body2="left_arm_long_link_bevel_gear_link" anchor="{fmt(PIN_LONG_L - O("left_arm_long_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="left_arm_long_parallel_second_loop" body1="left_arm_long_link_active_aux_link" body2="left_arm_long_link_bevel_gear_link" anchor="{fmt(PIN_LONG_L_SECOND - O("left_arm_long_link_active_aux_link"), 7)}" />'
)
eq.append(
    f'    <connect name="left_arm_short_parallel_loop" body1="left_arm_short_link_active_link" body2="left_arm_short_link_bevel_gear_link" anchor="{fmt(PIN_SHORT_L - O("left_arm_short_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="left_arm_short_parallel_second_loop" body1="left_arm_short_link_active_aux_link" body2="left_arm_short_link_bevel_gear_link" anchor="{fmt(PIN_SHORT_L_SECOND - O("left_arm_short_link_active_aux_link"), 7)}" />'
)

eq.append(
    '    <joint name="right_arm_long_bevel_mesh" joint1="right_arm_long_link_bevel_gear_joint" joint2="right_wrist_pitch_joint" polycoef="0 0.8 0 0 0" />'
)
eq.append(
    '    <joint name="right_arm_short_bevel_mesh" joint1="right_arm_short_link_bevel_gear_joint" joint2="right_wrist_pitch_joint" polycoef="0 -0.8 0 0 0" />'
)
eq.append(
    '    <joint name="left_arm_long_bevel_mesh" joint1="left_arm_long_link_bevel_gear_joint" joint2="left_wrist_pitch_joint" polycoef="0 -0.8 0 0 0" />'
)
eq.append(
    '    <joint name="left_arm_short_bevel_mesh" joint1="left_arm_short_link_bevel_gear_joint" joint2="left_wrist_pitch_joint" polycoef="0 0.8 0 0 0" />'
)

lines.append("  <equality>")
lines += eq
lines.append("  </equality>")

ACT = [
    (
        "waist_yaw_joint",
        "-2.618 2.618",
        "38.89891391234306",
        "2.4763817720221986",
        "-60 60",
    ),
    (
        "waist_right_motor_joint",
        f"{WAIST_RIGHT_CRANK_RANGE[0]:.8f} {WAIST_RIGHT_CRANK_RANGE[1]:.8f}",
        "77.79782782468612",
        "4.952763544044397",
        "-60 60",
    ),
    (
        "waist_left_motor_joint",
        f"{WAIST_LEFT_CRANK_RANGE[0]:.8f} {WAIST_LEFT_CRANK_RANGE[1]:.8f}",
        "77.79782782468612",
        "4.952763544044397",
        "-60 60",
    ),
    (
        "right_shoulder_pitch_joint",
        "-3.1415 1.0472",
        "11.943982412897888",
        "0.7603775364861449",
        "-30 30",
    ),
    (
        "right_shoulder_roll_joint",
        "-2.2515 1.5882",
        "11.943982412897888",
        "0.7603775364861449",
        "-30 30",
    ),
    (
        "right_shoulder_yaw_joint",
        "-2.618 2.618",
        "11.943982412897888",
        "0.7603775364861449",
        "-30 30",
    ),
    (
        "right_elbow_joint",
        "-0.78 1.5708",
        "11.943982412897888",
        "0.7603775364861449",
        "-30 30",
    ),
    (
        "right_wrist_roll_joint",
        "-2.67 2.67",
        "2.943826643921725",
        "0.18740982479430696",
        "-6 6",
    ),
    (
        "right_arm_long_link_motor_joint",
        "-1.2217 1.2217",
        "5.88765328784345",
        "0.3748196495886139",
        "-6 6",
    ),
    (
        "right_arm_short_link_motor_joint",
        "-1.2217 1.2217",
        "5.88765328784345",
        "0.3748196495886139",
        "-6 6",
    ),
    (
        "left_shoulder_pitch_joint",
        "-3.1415 1.0472",
        "11.943982412897888",
        "0.7603775364861449",
        "-30 30",
    ),
    (
        "left_shoulder_roll_joint",
        "-1.5882 2.2515",
        "11.943982412897888",
        "0.7603775364861449",
        "-30 30",
    ),
    (
        "left_shoulder_yaw_joint",
        "-2.618 2.618",
        "11.943982412897888",
        "0.7603775364861449",
        "-30 30",
    ),
    (
        "left_elbow_joint",
        "-0.78 1.5708",
        "11.943982412897888",
        "0.7603775364861449",
        "-30 30",
    ),
    (
        "left_wrist_roll_joint",
        "-2.67 2.67",
        "2.943826643921725",
        "0.18740982479430696",
        "-6 6",
    ),
    (
        "left_arm_long_link_motor_joint",
        "-1.2217 1.2217",
        "5.88765328784345",
        "0.3748196495886139",
        "-6 6",
    ),
    (
        "left_arm_short_link_motor_joint",
        "-1.2217 1.2217",
        "5.88765328784345",
        "0.3748196495886139",
        "-6 6",
    ),
    (
        "head_yaw_joint",
        "-2.618 2.618",
        "2.943826643921725",
        "0.18740982479430696",
        "-6 6",
    ),
]
lines.append("  <actuator>")
for name, cr, kp, kv, fr in ACT:
    lines.append(
        f'    <position name="{name}" joint="{name}" ctrlrange="{cr}" kp="{kp}" kv="{kv}" forcerange="{fr}" />'
    )
lines.append("  </actuator>")
lines.append("</mujoco>")

with open(f"{OUT}/Semi_Taks_LV1.xml", "w") as f:
    f.write("\n".join(lines))

URDF_LIMITS = {
    "torso_motor": ("60", "4.19"),
    "shoulder_motor": ("30", "3.77"),
    "wrist_motor": ("6", "12.57"),
}
ARMATURE = {
    "torso_motor": "0.00985321",
    "shoulder_motor": "0.0030254460886951994",
    "wrist_motor": "0.00074568",
}
u = []
u.append("<?xml version='1.0' encoding='utf-8'?>")
u.append('<robot name="Semi_Taks_LV1">')
u.append("  <mujoco>")
u.append('    <compiler meshdir="." discardvisual="false" strippath="false"/>')
u.append("  </mujoco>")
u.append('  <link name="world" />')
u.append('  <joint name="world_to_base" type="fixed">')
u.append('    <origin xyz="0 0 0" rpy="0 0 0" />')
u.append('    <parent link="world" />')
u.append('    <child link="base_link" />')
u.append("  </joint>")


def urdf_link(link):
    d = LINKS[link]
    I = np.array(d["inertia_com"])
    com = np.array(d["com_link"])
    u.append(f'  <link name="{link}">')
    u.append("    <inertial>")
    u.append(f'      <origin xyz="{fmt(com, 16)}" rpy="0 0 0" />')
    u.append(f'      <mass value="{d["mass"]:.16g}" />')
    u.append(
        f'      <inertia ixx="{I[0, 0]:.16g}" ixy="{I[0, 1]:.16g}" ixz="{I[0, 2]:.16g}" iyy="{I[1, 1]:.16g}" iyz="{I[1, 2]:.16g}" izz="{I[2, 2]:.16g}" />'
    )
    u.append("    </inertial>")
    for filename in d["mesh_files"]:
        u.append("    <visual>")
        u.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
        u.append("      <geometry>")
        u.append(f'        <mesh filename="meshes/{filename}" />')
        u.append("      </geometry>")
        u.append("    </visual>")
        u.append("    <collision>")
        u.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
        u.append("      <geometry>")
        u.append(f'        <mesh filename="meshes/{filename}" />')
        u.append("      </geometry>")
        u.append("    </collision>")
    u.append("  </link>")


def urdf_joint(link):
    parent, spec = BODY_TREE[link]
    jn = JOINT_NAME[link]
    pos = O(link) - O(parent)
    cls = spec["cls"]
    axis = "1 0 0" if spec.get("ball") else spec["axis"]
    if spec.get("ball") or spec.get("free"):
        lower, upper, eff, vel = "-3.1415", "3.1415", "0", "0"
    else:
        lo, hi = spec["range"].split()
        lower, upper = lo, hi
        eff, vel = URDF_LIMITS[cls]
    if link in ("waist_roll_link", "waist_pitch_link", "waist_yaw_link"):
        eff, vel = URDF_LIMITS["torso_motor"]
    if link == "head_yaw_link":
        eff, vel = URDF_LIMITS["wrist_motor"]
    u.append(f'  <joint name="{jn}" type="revolute">')
    u.append(f'    <origin xyz="{fmt(pos, 7)}" rpy="0 0 0" />')
    u.append(f'    <parent link="{parent}" />')
    u.append(f'    <child link="{link}" />')
    u.append(f'    <axis xyz="{axis}" />')
    u.append(
        f'    <limit lower="{lower}" upper="{upper}" effort="{eff}" velocity="{vel}" />'
    )
    u.append(f'    <dynamics armature="{ARMATURE[cls]}" />')
    u.append("  </joint>")


urdf_link("base_link")


def urdf_walk(link):
    urdf_joint(link)
    urdf_link(link)
    for ch in CHILDREN.get(link, []):
        urdf_walk(ch)


urdf_walk("waist_yaw_link")
for parent, cams in CAMERAS.items():
    for name, pos_s in cams:
        u.append(f'  <joint name="{name}_joint" type="fixed">')
        u.append(f'    <origin xyz="{pos_s}" rpy="{CAMERA_URDF_RPY}" />')
        u.append(f'    <parent link="{parent}" />')
        u.append(f'    <child link="{name}_link" />')
        u.append("  </joint>")
        u.append(f'  <link name="{name}_link" />')
u.append("</robot>")


def gen_urdf():
    return "\n".join(u)


with open(f"{OUT}/Semi_Taks_LV1.urdf", "w") as f:
    f.write(gen_urdf())

ledger = {
    "robot": "Semi_Taks_LV1",
    "consumers": ["MuJoCo", "urdfdom", "Pinocchio", "RViz"],
    "units": {"length": "m", "mass": "kg", "angle": "rad", "mesh": "m"},
    "frame_map_from_cad": M_ROB.tolist(),
    "cad_origin_mm": O_CAD.tolist(),
    "top_assembly_scale": 1.0,
    "geometry_source": "0826临时版本.step",
    "dynamics_source": "s5激进单边骨架_materials.json",
    "topology_source": "Semi_Taks_LV1_STAGE4",
    "links": {
        link: {
            "role": "physical",
            "frame_origin_robot_m": LINKS[link]["origin"],
            "meshes": [f"meshes/{name}" for name in LINKS[link]["mesh_files"]],
            "inertial_source_paths": LINKS[link]["source_paths"],
        }
        for link in sorted(LINKS)
    },
    "joints": {
        JOINT_NAME[link]: {
            "parent": BODY_TREE[link][0],
            "child": link,
            "origin_parent_m": (O(link) - O(BODY_TREE[link][0])).tolist(),
            "axis_joint": (
                [1.0, 0.0, 0.0]
                if BODY_TREE[link][1].get("ball")
                else [float(x) for x in BODY_TREE[link][1]["axis"].split()]
            ),
            "positive_motion": "沿 STAGE3 关节轴右手定则",
            "source": "S4 CAD axis with STAGE3 sign convention",
        }
        for link in ORDER
    },
}
with open(os.path.join(HERE, "design_ledger.json"), "w", encoding="utf-8") as f:
    json.dump(ledger, f, ensure_ascii=False, indent=2)

print("joint origins (robot frame):")
for link in ORDER:
    parent = BODY_TREE[link][0]
    print(f"  {JOINT_NAME[link]:44s} pos={fmt(O(link) - O(parent), 6)}")
print(
    "anchors:",
    fmt(BALL_WR - O("waist_right_motor_link"), 6),
    "|",
    fmt(PIN_LONG - O("right_arm_long_link_active_link"), 6),
)
print("written:", OUT)
