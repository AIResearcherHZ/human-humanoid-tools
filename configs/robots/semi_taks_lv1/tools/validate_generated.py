import json
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from waist_closed_chain_ik import WaistClosedChainIK

WAIST_CRANK_RANGES = ((-0.75, 1.1619), (-1.1619, 0.75))
ROOT = os.path.dirname(HERE)
STAGE3 = os.path.join(
    os.path.dirname(os.path.dirname(ROOT)), "备份", "Semi_Taks_LV1_STAGE4"
)
SECOND_CONNECTS = {
    "right_arm_long_parallel_second_loop",
    "right_arm_short_parallel_second_loop",
    "left_arm_long_parallel_second_loop",
    "left_arm_short_parallel_second_loop",
    "head_front_parallel_second_loop",
    "head_rear_parallel_second_loop",
}
AUX_LINKS = {
    "right_arm_long_link_active_aux_link",
    "right_arm_short_link_active_aux_link",
    "left_arm_long_link_active_aux_link",
    "left_arm_short_link_active_aux_link",
    "head_front_link_active_aux_link",
    "head_rear_link_active_aux_link",
}
AUX_JOINTS = {name.removesuffix("_link") + "_joint" for name in AUX_LINKS}
CAMERA_POSITIONS = {
    "head_left_camera": np.array([0.08812307593, 0.02999965544, 0.05311159739]),
    "head_right_camera": np.array([0.08812307593, -0.03000034456, 0.05311159739]),
}
COLLISION_GROUPS = {
    "torso": (1, 4),
    "right": (8,),
    "left": (16,),
    "head": (32,),
}


def collides(model, first, second):
    return any(
        (model.geom_contype[left] & model.geom_conaffinity[right])
        or (model.geom_contype[right] & model.geom_conaffinity[left])
        for left in first
        for right in second
    )


def names(model, kind):
    count = getattr(model, "n" + kind)
    accessor = getattr(model, kind.removesuffix("s"))
    return {accessor(index).name for index in range(count)}


def named_set(model, count_name, accessor_name):
    accessor = getattr(model, accessor_name)
    return {accessor(index).name for index in range(getattr(model, count_name))}


def equality_residual(model, data):
    indices = np.flatnonzero(data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    return float(np.max(np.abs(data.efc_pos[indices]))) if len(indices) else 0.0


def validate_links():
    payload = json.load(open(os.path.join(HERE, "links.json"), encoding="utf-8"))
    metadata = payload["metadata"]
    links = payload["links"]
    assert (
        metadata["fusion_hits"] + len(metadata["geometry_only_paths"])
        == metadata["fusion_total"]
    )
    assert metadata["top_assembly_scale"] == 1.0
    assert abs(metadata["generated_mass_kg"] - metadata["source_mass_kg"]) < 1e-12
    assert all(
        path.rsplit("/", 1)[-1].startswith("=>")
        for path in metadata["geometry_only_paths"]
    )
    assert all(
        source["mass_kg"] >= 0.0 for source in metadata["residual_sources"].values()
    )
    assert (
        set(links)
        == {
            node.attrib["name"]
            for node in ET.parse(os.path.join(STAGE3, "Semi_Taks_LV1.urdf"))
            .getroot()
            .findall("link")
            if node.attrib["name"]
            not in {"world", "head_left_camera_link", "head_right_camera_link"}
        }
        | AUX_LINKS
    )
    assert all(
        "RP70H外形图" not in path
        for side in ("waist_right_motor_link", "waist_left_motor_link")
        for path in links[side]["source_paths"]
    )
    assert (
        sum("RP70H外形图" in path for path in links["waist_roll_link"]["source_paths"])
        == 2
    )
    assert links["waist_roll_link"]["mesh_files"][:5] == [
        "waist_roll_link.STL",
        "waist_roll_link_torso.STL",
        "waist_roll_link_right_shoulder_mount.STL",
        "waist_roll_link_left_shoulder_mount.STL",
        "waist_roll_link_neck_mount.STL",
    ]
    minimum_eigenvalue = np.inf
    maximum_bbox_error = 0.0
    maximum_faces = 0
    maximum_radius_ratio = 0.0
    for link, data in links.items():
        inertia = np.asarray(data["inertia_com"])
        eigenvalues = np.linalg.eigvalsh(inertia)
        assert np.all(eigenvalues > 0.0), (link, eigenvalues)
        assert eigenvalues[2] <= eigenvalues[0] + eigenvalues[1] + 1e-12, (
            link,
            eigenvalues,
        )
        minimum_eigenvalue = min(minimum_eigenvalue, float(eigenvalues.min()))
        bbox = np.asarray(data["bbox_step_robot"])
        com = np.asarray(data["origin"]) + np.asarray(data["com_link"])
        assert np.all(com >= bbox[0] - 2e-4) and np.all(com <= bbox[1] + 2e-4), (
            link,
            com,
            bbox,
        )
        corners = np.array(
            [[x, y, z] for x in bbox[:, 0] for y in bbox[:, 1] for z in bbox[:, 2]]
        )
        maximum_radius = float(np.max(np.linalg.norm(corners - com, axis=1)))
        equivalent_radius = float(np.sqrt(np.trace(inertia) / (2.0 * data["mass"])))
        radius_ratio = equivalent_radius / maximum_radius
        assert radius_ratio <= 1.0 + 1e-9, (link, equivalent_radius, maximum_radius)
        maximum_radius_ratio = max(maximum_radius_ratio, radius_ratio)
        maximum_bbox_error = max(maximum_bbox_error, data["bbox_max_error_m"])
        assert data["bbox_max_error_m"] < 2e-3, (link, data["bbox_max_error_m"])
        for filename in data["mesh_files"]:
            path = os.path.join(ROOT, "meshes", filename)
            assert os.path.isfile(path), path
            mesh = trimesh.load(path, process=False)
            assert len(mesh.faces) > 0, path
            assert len(mesh.faces) <= 180000, (path, len(mesh.faces))
            maximum_faces = max(maximum_faces, len(mesh.faces))
    assert metadata["part_inertia_reference"] == "part_local_origin"
    assert metadata["part_inertia_parallel_axis_shifted"] == metadata["fusion_hits"]
    assert all(
        np.isfinite(correction) and correction >= 0.0
        for correction in metadata["projected_part_inertias"].values()
    )
    return (
        metadata,
        links,
        minimum_eigenvalue,
        maximum_bbox_error,
        maximum_faces,
        maximum_radius_ratio,
    )


def validate_urdf(source_mass, links):
    path = os.path.join(ROOT, "Semi_Taks_LV1.urdf")
    root = ET.parse(path).getroot()
    assert root.tag == "robot"
    link_nodes = root.findall("link")
    joint_nodes = root.findall("joint")
    link_names = [node.attrib["name"] for node in link_nodes]
    assert len(link_names) == len(set(link_names))
    stage_root = ET.parse(os.path.join(STAGE3, "Semi_Taks_LV1.urdf")).getroot()
    assert (
        set(link_names)
        == {node.attrib["name"] for node in stage_root.findall("link")} | AUX_LINKS
    )
    assert {node.attrib["name"] for node in joint_nodes} == {
        node.attrib["name"] for node in stage_root.findall("joint")
    } | AUX_JOINTS
    parents = {}
    for joint in joint_nodes:
        child = joint.find("child").attrib["link"]
        parent = joint.find("parent").attrib["link"]
        assert child not in parents
        assert child in link_names and parent in link_names
        parents[child] = parent
        limit = joint.find("limit")
        if joint.attrib["type"] == "revolute":
            assert limit is not None
            assert float(limit.attrib["lower"]) <= 0.0 <= float(limit.attrib["upper"])
    roots = set(link_names) - set(parents)
    assert roots == {"world"}
    urdf_mass = sum(
        float(node.find("inertial/mass").attrib["value"])
        for node in link_nodes
        if node.find("inertial/mass") is not None
    )
    assert abs(urdf_mass - source_mass) < 1e-9, (urdf_mass, source_mass)
    for link, data in links.items():
        node = root.find(f"link[@name='{link}']")
        filenames = [
            mesh.attrib["filename"] for mesh in node.findall("visual/geometry/mesh")
        ]
        assert filenames == [f"meshes/{name}" for name in data["mesh_files"]]
        collision_filenames = [
            mesh.attrib["filename"] for mesh in node.findall("collision/geometry/mesh")
        ]
        assert collision_filenames == filenames
        for filename in filenames:
            assert os.path.isfile(os.path.join(ROOT, filename))
    for name, expected in CAMERA_POSITIONS.items():
        joint = root.find(f"joint[@name='{name}_joint']")
        assert np.allclose(
            np.fromstring(joint.find("origin").attrib["xyz"], sep=" "),
            expected,
            atol=1e-12,
        )
        assert np.allclose(
            np.fromstring(joint.find("origin").attrib["rpy"], sep=" "),
            [0.0, 0.172787596, 0.0],
            atol=1e-12,
        )
    return len(link_nodes), len(joint_nodes), urdf_mass


def validate_mujoco(source_mass):
    model_path = os.path.join(ROOT, "Semi_Taks_LV1.xml")
    scene_path = os.path.join(ROOT, "scene_Semi_Taks_LV1.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    scene = mujoco.MjModel.from_xml_path(scene_path)
    mujoco.MjModel.from_xml_path(os.path.join(ROOT, "Semi_Taks_LV1_chassis.xml"))
    mujoco.MjModel.from_xml_path(os.path.join(ROOT, "scene_Semi_Taks_LV1_chassis.xml"))
    model_equalities = named_set(model, "neq", "equality")
    assert {"waist_right_parallel_loop", "waist_left_parallel_loop"} <= model_equalities
    assert SECOND_CONNECTS <= model_equalities
    for index in range(model.njnt):
        joint = model.joint(index)
        if joint.name in AUX_JOINTS:
            assert not model.jnt_limited[index]
            continue
        if model.jnt_limited[index]:
            assert model.jnt_range[index, 0] <= 0.0 <= model.jnt_range[index, 1], (
                joint.name
            )
    for index in range(model.nu):
        model.actuator(index)
        joint_id = model.actuator_trnid[index, 0]
        assert joint_id >= 0
        if model.actuator_ctrllimited[index] and model.jnt_limited[joint_id]:
            control = model.actuator_ctrlrange[index]
            joint_range = model.jnt_range[joint_id]
            assert control[0] >= joint_range[0] - 1e-12
            assert control[1] <= joint_range[1] + 1e-12
    assert abs(float(model.body_mass.sum()) - source_mass) < 1e-9
    mesh_count = sum(
        len(link["mesh_files"])
        for link in json.load(open(os.path.join(HERE, "links.json"), encoding="utf-8"))[
            "links"
        ].values()
    )
    assert model.nmesh == mesh_count
    assert model.ngeom == 2 * mesh_count
    for mesh_id in range(model.nmesh):
        geoms = np.flatnonzero(model.geom_dataid == mesh_id)
        assert len(geoms) == 2
        assert min(model.geom_contype[geoms]) == 0
    collision_geoms = np.flatnonzero(model.geom_group == 0)
    assert len(collision_geoms) == mesh_count
    groups = {
        name: [
            geom_id
            for geom_id in collision_geoms
            if model.geom_contype[geom_id] in contypes
        ]
        for name, contypes in COLLISION_GROUPS.items()
    }
    assert all(groups.values())
    assert not any(collides(model, geoms, geoms) for geoms in groups.values())
    assert collides(model, groups["right"], groups["left"])
    assert collides(model, groups["right"], groups["head"])
    assert collides(model, groups["left"], groups["head"])
    assert not any(
        collides(model, groups["torso"], groups[name])
        for name in ("right", "left", "head")
    )
    floor = [scene.geom("floor").id]
    scene_groups = {
        name: [scene.geom(model.geom(geom_id).name).id for geom_id in geoms]
        for name, geoms in groups.items()
    }
    assert all(collides(scene, floor, geoms) for geoms in scene_groups.values())
    payload = json.load(open(os.path.join(HERE, "links.json"), encoding="utf-8"))
    urdf = ET.parse(os.path.join(ROOT, "Semi_Taks_LV1.urdf")).getroot()
    maximum_mjcf_inertia_error = 0.0
    maximum_urdf_inertia_error = 0.0
    for name, link in payload["links"].items():
        reference = np.asarray(link["inertia_com"])
        body_id = model.body(name).id
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
        rotation = rotation.reshape(3, 3)
        compiled = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        maximum_mjcf_inertia_error = max(
            maximum_mjcf_inertia_error, float(np.max(np.abs(compiled - reference)))
        )
        node = urdf.find(f"link[@name='{name}']/inertial/inertia")
        emitted = np.array(
            [
                [
                    float(node.get("ixx")),
                    float(node.get("ixy")),
                    float(node.get("ixz")),
                ],
                [
                    float(node.get("ixy")),
                    float(node.get("iyy")),
                    float(node.get("iyz")),
                ],
                [
                    float(node.get("ixz")),
                    float(node.get("iyz")),
                    float(node.get("izz")),
                ],
            ]
        )
        maximum_urdf_inertia_error = max(
            maximum_urdf_inertia_error, float(np.max(np.abs(emitted - reference)))
        )
    assert maximum_mjcf_inertia_error < 1e-8, maximum_mjcf_inertia_error
    assert maximum_urdf_inertia_error < 1e-10, maximum_urdf_inertia_error
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    residual_zero = equality_residual(model, data)
    assert residual_zero < 1e-10, residual_zero
    equality_rows = np.flatnonzero(
        data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    )
    equality_jacobian = data.efc_J.reshape(data.nefc, model.nv)[equality_rows]
    equality_singular_values = np.linalg.svd(equality_jacobian, compute_uv=False)
    equality_rank = int(np.sum(equality_singular_values > 1e-9))
    assert np.isfinite(equality_singular_values).all()
    assert equality_rank > 0
    for secondary in SECOND_CONNECTS:
        prefix = secondary.replace("_second_loop", "_loop")
        first_id = model.equality(prefix).id
        second_id = model.equality(secondary).id
        rows = []
        for equality_id in (first_id, second_id):
            rows.extend(
                np.flatnonzero(
                    (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                    & (data.efc_id == equality_id)
                ).tolist()
            )
        assert len(rows) == 6
        first_rank = np.linalg.matrix_rank(
            data.efc_J.reshape(data.nefc, model.nv)[rows[:3]], tol=1e-9
        )
        pair_rank = np.linalg.matrix_rank(
            data.efc_J.reshape(data.nefc, model.nv)[rows], tol=1e-9
        )
        assert pair_rank == first_rank + 1, (secondary, first_rank, pair_rank)
    expected_xyaxes = np.array([[0.0, -1.0, 0.0], [0.1719291, 0.0, 0.985109326]])
    for name, expected in CAMERA_POSITIONS.items():
        camera_id = model.camera(name).id
        assert np.allclose(model.cam_pos[camera_id], expected, atol=1e-12)
        camera_matrix = np.empty(9)
        mujoco.mju_quat2Mat(camera_matrix, model.cam_quat[camera_id])
        camera_matrix = camera_matrix.reshape(3, 3)
        assert np.allclose(camera_matrix[:, :2].T, expected_xyaxes, atol=1e-7)
    qpos_zero = data.qpos.copy()
    for _ in range(600):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert np.isfinite(data.qacc).all()
    drift = float(np.max(np.abs(data.qpos - qpos_zero)))
    velocity = float(np.max(np.abs(data.qvel)))
    residual_settled = equality_residual(model, data)
    assert drift < 0.1, drift
    assert velocity < 0.01, velocity
    assert residual_settled < 1e-3, residual_settled
    motion_joints = [
        "right_arm_long_link_motor_joint",
        "right_arm_short_link_motor_joint",
        "left_arm_long_link_motor_joint",
        "left_arm_short_link_motor_joint",
        "head_front_link_motor_joint",
        "head_rear_link_motor_joint",
    ]
    maximum_tracking_error = 0.0
    maximum_motion_residual = 0.0
    for target in (0.2, -0.2, 0.5):
        motion_data = mujoco.MjData(model)
        qpos_addresses = []
        for name in motion_joints:
            actuator_id = model.actuator(name).id
            motion_data.ctrl[actuator_id] = target
            joint_id = model.actuator_trnid[actuator_id, 0]
            qpos_addresses.append(model.jnt_qposadr[joint_id])
        for _ in range(1500):
            mujoco.mj_step(model, motion_data)
        tracking_error = float(
            np.max(np.abs(motion_data.qpos[qpos_addresses] - target))
        )
        motion_residual = equality_residual(model, motion_data)
        maximum_tracking_error = max(maximum_tracking_error, tracking_error)
        maximum_motion_residual = max(maximum_motion_residual, motion_residual)
        assert tracking_error < 0.01, (target, tracking_error)
        assert motion_residual < 1e-3, (target, motion_residual)
        assert motion_data.ncon == 0, (target, motion_data.ncon)
        assert not any(warning.number for warning in motion_data.warning)
    waist_actuators = (
        "waist_right_motor_joint",
        "waist_left_motor_joint",
    )
    for name, expected_range in zip(waist_actuators, WAIST_CRANK_RANGES):
        jid = model.joint(name).id
        assert np.allclose(model.jnt_range[jid], expected_range)
        aid = model.actuator(name).id
        assert np.allclose(model.actuator_ctrlrange[aid], expected_range)
    maximum_waist_tracking_error = 0.0
    maximum_waist_motion_residual = 0.0
    for right_target, left_target in (
        (0.14568, -0.14568),
        (-0.14568, 0.14568),
        (0.048, 0.048),
        (-0.048, -0.048),
        (0.152544, -0.080544),
    ):
        waist_data = mujoco.MjData(model)
        targets = (right_target, left_target)
        qpos_addresses = []
        for name, target in zip(waist_actuators, targets):
            actuator_id = model.actuator(name).id
            waist_data.ctrl[actuator_id] = target
            joint_id = model.actuator_trnid[actuator_id, 0]
            qpos_addresses.append(model.jnt_qposadr[joint_id])
        for _ in range(10000):
            mujoco.mj_step(model, waist_data)
        tracking_error = float(
            np.max(np.abs(waist_data.qpos[qpos_addresses] - targets))
        )
        motion_residual = equality_residual(model, waist_data)
        maximum_waist_tracking_error = max(maximum_waist_tracking_error, tracking_error)
        maximum_waist_motion_residual = max(
            maximum_waist_motion_residual, motion_residual
        )
        assert tracking_error < 0.015, (targets, tracking_error)
        assert motion_residual < 1e-3, (targets, motion_residual)
        assert waist_data.ncon == 0, (targets, waist_data.ncon)
        assert not any(warning.number for warning in waist_data.warning)
    waist_ik = WaistClosedChainIK(model_path)
    right_target, left_target = waist_ik.inverse(0.0, -0.7854)
    waist_data = mujoco.MjData(model)
    for name, target in zip(waist_actuators, (right_target, left_target)):
        waist_data.ctrl[model.actuator(name).id] = target
    for _ in range(10000):
        mujoco.mj_step(model, waist_data)
    pitch_qpos = model.jnt_qposadr[model.joint("waist_pitch_joint").id]
    roll_qpos = model.jnt_qposadr[model.joint("waist_roll_joint").id]
    limit_joint_ids = waist_data.efc_id[: waist_data.nefc][
        waist_data.efc_type[: waist_data.nefc]
        == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
    ]
    assert abs(waist_data.qpos[pitch_qpos] + 0.7854) < 0.01
    assert abs(waist_data.qpos[roll_qpos]) < 0.01
    assert waist_data.ncon == 0
    assert all(
        joint_id == model.joint("waist_pitch_joint").id for joint_id in limit_joint_ids
    )
    positive_singular_values = equality_singular_values[equality_singular_values > 1e-9]
    return (
        model,
        residual_zero,
        residual_settled,
        drift,
        velocity,
        maximum_mjcf_inertia_error,
        maximum_urdf_inertia_error,
        equality_rank,
        float(positive_singular_values[-1]),
        maximum_tracking_error,
        maximum_motion_residual,
        maximum_waist_tracking_error,
        maximum_waist_motion_residual,
    )


def main():
    (
        metadata,
        links,
        minimum_eigenvalue,
        maximum_bbox_error,
        maximum_faces,
        maximum_radius_ratio,
    ) = validate_links()
    urdf_links, urdf_joints, urdf_mass = validate_urdf(
        metadata["source_mass_kg"], links
    )
    (
        model,
        residual_zero,
        residual_settled,
        drift,
        velocity,
        mjcf_inertia_error,
        urdf_inertia_error,
        equality_rank,
        equality_min_singular,
        maximum_tracking_error,
        maximum_motion_residual,
        maximum_waist_tracking_error,
        maximum_waist_motion_residual,
    ) = validate_mujoco(metadata["source_mass_kg"])
    result = {
        "fusion_paths": f"{metadata['fusion_hits']}/{metadata['fusion_total']}",
        "geometry_only_paths": len(metadata["geometry_only_paths"]),
        "mass_kg": metadata["source_mass_kg"],
        "minimum_inertia_eigenvalue": minimum_eigenvalue,
        "maximum_bbox_error_m": maximum_bbox_error,
        "maximum_mesh_faces": maximum_faces,
        "maximum_equivalent_radius_ratio": maximum_radius_ratio,
        "projected_part_inertias": metadata["projected_part_inertias"],
        "maximum_mjcf_inertia_error": mjcf_inertia_error,
        "maximum_urdf_inertia_error": urdf_inertia_error,
        "mujoco": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq},
        "urdf": {"links": urdf_links, "joints": urdf_joints, "mass_kg": urdf_mass},
        "qpos0_equality_residual": residual_zero,
        "equality_jacobian_rank": equality_rank,
        "equality_min_singular_value": equality_min_singular,
        "settled_equality_residual": residual_settled,
        "settled_qpos_drift": drift,
        "settled_qvel_max": velocity,
        "closed_loop_tracking_error_max": maximum_tracking_error,
        "closed_loop_motion_residual_max": maximum_motion_residual,
        "waist_closed_loop_tracking_error_max": maximum_waist_tracking_error,
        "waist_closed_loop_motion_residual_max": maximum_waist_motion_residual,
        "status": "PASS",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
