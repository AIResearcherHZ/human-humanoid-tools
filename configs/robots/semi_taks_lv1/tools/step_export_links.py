import argparse
import json
import os
import struct

import numpy as np
import trimesh
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
STEP = os.path.join(HERE, "Semi_Taks_LV1.step")
MATERIALS = os.path.join(HERE, "Semi_Taks_LV1.json")
MESH_DIR = os.path.join(OUT, "meshes")
O_CAD = np.array([0.0, -4.8, -383.3])
M_ROB = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

ORIGINS_CAD = {
    "base_link": [0.0, -4.8, -383.3],
    "waist_yaw_link": [0.0, -4.8, -313.5],
    "waist_right_link_driven_link": [-22.1, -38.95, -297.0],
    "waist_left_link_driven_link": [22.0, -38.95, -297.0],
    "waist_pitch_link": [0.0, -4.8, -281.0],
    "waist_roll_link": [0.0, -4.8, -281.0],
    "waist_right_motor_link": [-42.5, -29.95, -229.0],
    "waist_left_motor_link": [42.5, -29.95, -229.0],
    "right_shoulder_yaw_link": [-161.8994, -3.7, -94.425],
    "right_shoulder_pitch_link": [-98.2676, 0.0, 32.8799],
    "right_shoulder_roll_link": [-146.9438, -42.0, 11.036],
    "right_elbow_link": [-197.5, -6.7, -209.9247],
    "right_wrist_yaw_link": [-167.0, -235.4, -234.9247],
    "right_wrist_pitch_link": [-167.0, -235.4, -234.9247],
    "right_wrist_roll_link": [-166.9994, -103.9, -234.9247],
    "left_shoulder_yaw_link": [161.8994, -3.7, -94.425],
    "left_shoulder_pitch_link": [98.2676, 0.0, 32.8799],
    "left_shoulder_roll_link": [146.9438, -42.0, 11.036],
    "left_elbow_link": [197.5, -6.7, -209.9247],
    "left_wrist_yaw_link": [167.0, -235.4, -234.9247],
    "left_wrist_pitch_link": [167.0, -235.4, -234.9247],
    "left_wrist_roll_link": [166.9994, -103.9, -234.9247],
    "head_yaw_link": [0.0, 0.0, 177.2],
    "head_pitch_link": [0.0, 0.0, 123.0],
}


def classify(path):
    if path == "/干涉区:1":
        return "waist_roll_link"
    if path.startswith("/胸腔:1/"):
        return "waist_roll_link"
    if path.startswith("/S6eyou腰:1/"):
        if "/底座:1/" in path:
            return "base_link"
        if "/RP70H外形图 整体step:3" in path:
            return "base_link"
        if "/限位补丁:1/yaw轴限位块:1" in path:
            return "base_link"
        if "/yaw轴与十字轴连接装配:1/鱼眼轴承:4" in path:
            return "waist_right_link_driven_link"
        if "/yaw轴与十字轴连接装配:1/鱼眼轴承:5" in path:
            return "waist_left_link_driven_link"
        if "/yaw轴与十字轴连接装配:1/" in path:
            return "waist_yaw_link"
        if "/雷霆大改真连杆:1" in path:
            return "waist_right_link_driven_link"
        if "/雷霆大改真连杆(镜像):1" in path:
            return "waist_left_link_driven_link"
        if "/新十字轴组件:1/" in path:
            return "waist_pitch_link"
        if "/雷霆大改电机输出:1/" in path:
            if "/鱼眼轴承:1" in path:
                return "waist_right_link_driven_link"
            return "waist_right_motor_link"
        if "/雷霆大改电机输出:2/" in path:
            if "/鱼眼轴承:1" in path:
                return "waist_left_link_driven_link"
            return "waist_left_motor_link"
        return "waist_roll_link"
    if path.startswith("/S5新右臂:1/"):
        return classify_arm_s5(path, "right", False)
    if path.startswith("/S5新右臂(镜像):1/"):
        return classify_arm_s5(path, "left", True)
    if path.startswith("/S6大花生脖子:1/"):
        return classify_head(path)
    if path.startswith("/S6无roll轴脖子:1/"):
        if "/脖子pitch轴运动部分:1/" in path:
            return "head_pitch_link"
        if "/S6中空40模块:1/" in path:
            return "head_yaw_link"
        return "waist_roll_link"
    return None


def classify_arm_s5(path, side, mirrored):
    mark = "(镜像)" if mirrored else ""
    if f"/S5-J1固定部分{mark}:1/" in path or f"/S5-J1固定部分 {mark}:1/" in path:
        return "waist_roll_link"
    if f"/S5-J1运动部分 {mark}:1/" in path:
        return f"{side}_shoulder_pitch_link"
    if f"/S5-J2运动部分{mark}:1/" in path:
        return f"{side}_shoulder_roll_link"
    if f"/S5-J3运动部分{mark}:1/" in path:
        return f"{side}_shoulder_yaw_link"
    serial_wrist = f"/新：手腕{mark}:1/"
    if serial_wrist in path:
        if path.rsplit("/", 1)[-1].startswith("末端电阻"):
            return f"{side}_wrist_yaw_link"
        if f"/J5{mark}:1/" in path:
            if path.endswith(
                (
                    f"/S6中空40模块{mark}:1/55X66X5薄壁交叉滚子轴承{mark}:1",
                    f"/S6中空40模块{mark}:1/内圈固定{mark}:1/内圈固定盖板{mark}:1",
                    f"/S6中空40模块{mark}:1/内圈固定{mark}:1/内圈固定底座{mark}:1",
                )
            ):
                return f"{side}_wrist_roll_link"
            if f"/外圈固定&传动{mark}:1/" in path or f"/J5运动部分{mark}:1/" in path:
                return f"{side}_wrist_roll_link"
            return f"{side}_elbow_link"
        if f"/J6{mark}:1/" in path:
            return f"{side}_wrist_roll_link"
        if f"/J7{mark}:1/" in path:
            if f"/运动部分{mark}:1/" in path or f"/运动部分{mark} (1):1/" in path:
                return f"{side}_wrist_pitch_link"
            if f"/S6-40c新抱箍模块{mark}:1/" in path:
                return f"{side}_wrist_yaw_link"
            return f"{side}_wrist_roll_link"
        return f"{side}_elbow_link"
    return f"{side}_elbow_link"


def classify_head(path):
    if "/大花生双臂组合A:1/" in path:
        if any(
            marker in path
            for marker in (
                "/RP40C行星关节模组_",
                "/大花生后端盖:1",
                "/花生壳:2/",
            )
        ):
            return "waist_roll_link"
        return "waist_roll_link"
    if "/大花生双臂组合A(镜像):2/" in path:
        if any(
            marker in path
            for marker in (
                "/RP40C行星关节模组_",
                "/大花生后端盖(镜像):1",
                "/花生壳(镜像):1/",
            )
        ):
            return "waist_roll_link"
        return "head_pitch_link"
    if "/S6中空40模块:1/" in path:
        return "head_yaw_link"
    return "waist_roll_link"


def waist_mesh_bucket(path):
    if path.startswith("/胸腔:1/"):
        return "torso"
    if path.startswith("/S5新右臂:1/S5-J1固定部分:1/"):
        return "right_shoulder_mount"
    if path.startswith("/S5新右臂(镜像):1/S5-J1固定部分(镜像):1/"):
        return "left_shoulder_mount"
    return "core"


def split_active_link(link, shape):
    return link


def residual_parent(path):
    if not path.rsplit("/", 1)[-1].startswith("=>"):
        return None
    if "/50H轴承后盖（J2处）" in path:
        return path.rsplit("/", 1)[0]
    if "/线束" in path:
        return path.rsplit("/", 1)[0]
    if "/雷霆大改前端固定铝件:1/" in path:
        return path.rsplit("/", 1)[0]
    return None


def get_name(label):
    value = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), value):
        return value.Get().ToExtString()
    return "?"


def matrix_from_location(location):
    transform = location.Transformation()
    rotation = np.array(
        [[transform.Value(i, j) for j in range(1, 4)] for i in range(1, 4)]
    )
    translation = transform.TranslationPart()
    return rotation, np.array([translation.X(), translation.Y(), translation.Z()])


def append_triangles(face, vertices, faces, origin_robot, linear, angular):
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)
    if triangulation is None:
        BRepMesh_IncrementalMesh(face, linear, False, angular, False)
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
    if triangulation is None:
        return
    transform = location.Transformation()
    base = len(vertices)
    for index in range(1, triangulation.NbNodes() + 1):
        point = triangulation.Node(index).Transformed(transform)
        cad = np.array([point.X(), point.Y(), point.Z()])
        vertices.append(M_ROB @ (cad - O_CAD) / 1000.0 - origin_robot)
    reverse = face.Orientation() == TopAbs_REVERSED
    for index in range(1, triangulation.NbTriangles() + 1):
        a, b, c = triangulation.Triangle(index).Get()
        if reverse:
            faces.append((base + a - 1, base + c - 1, base + b - 1))
        else:
            faces.append((base + a - 1, base + b - 1, base + c - 1))


def shape_mesh_robot(shape, origin_robot, coarse=False):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    lower, upper = box.CornerMin(), box.CornerMax()
    xmin, ymin, zmin = lower.X(), lower.Y(), lower.Z()
    xmax, ymax, zmax = upper.X(), upper.Y(), upper.Z()
    diagonal = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    factor = 0.008 if coarse else 0.004
    linear = min(
        2.4 if coarse else 1.2, max(0.3 if coarse else 0.15, factor * diagonal)
    )
    angular = 1.0 if coarse else (0.8 if diagonal < 60.0 else 0.5)
    BRepMesh_IncrementalMesh(shape, linear, False, angular, True)
    vertices, faces = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        append_triangles(
            (getattr(TopoDS, "Face_s", None) or TopoDS.Face)(explorer.Current()),
            vertices,
            faces,
            origin_robot,
            linear,
            angular,
        )
        explorer.Next()
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", nargs="+", choices=sorted(ORIGINS_CAD), default=None)
    args = parser.parse_args()
    os.makedirs(MESH_DIR, exist_ok=True)
    materials = json.load(open(MATERIALS, encoding="utf-8"))
    material_keys = list(materials)
    material_leaves = {
        key
        for key in material_keys
        if not any(x.startswith(key + "/") for x in material_keys)
    }
    material_roots = {key.split("/", 2)[1] for key in material_keys}
    document = TDocStd_Document(TCollection_ExtendedString("Semi_Taks_LV1"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(STEP) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {STEP}")
    reader.Transfer(document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    leaves = []
    empty_paths = []
    node_locations = {}

    def walk(label, location, path, inherited_material=None):
        accumulated = location.Multiplied(shape_tool.GetLocation_s(label))
        target = label
        if shape_tool.IsReference_s(label):
            referred = TDF_Label()
            shape_tool.GetReferredShape_s(label, referred)
            target = referred
        display = get_name(label)
        if display == "?":
            display = get_name(target)
        full_path = path + "/" + display
        material_path = full_path
        first = full_path.find("/", 1)
        if first >= 0 and full_path[first + 1 :].split("/", 1)[0] in material_roots:
            material_path = full_path[first:]
        target_location = accumulated.Multiplied(
            shape_tool.GetShape_s(target).Location()
        )
        if material_path in materials:
            node_locations[material_path] = target_location
        owner = inherited_material
        if material_path in material_leaves:
            owner = (material_path, target_location)
        if shape_tool.IsAssembly_s(target):
            children = TDF_LabelSequence()
            shape_tool.GetComponents_s(target, children)
            for index in range(1, children.Length() + 1):
                walk(children.Value(index), accumulated, full_path, owner)
            return
        shape = shape_tool.GetShape_s(target).Located(target_location)
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        if box.IsVoid():
            empty_paths.append(material_path)
            exact = material_path if material_path in material_leaves else None
            if exact is not None:
                leaves.append((material_path, None, target_location, exact, owner))
            return
        exact = material_path if material_path in material_leaves else None
        leaves.append((material_path, shape, target_location, exact, owner))

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    for index in range(1, roots.Length() + 1):
        walk(roots.Value(index), TopLoc_Location(), "")

    right_rod = next(
        shape
        for path, shape, _, _, _ in leaves
        if path == "/S6eyou腰:1/雷霆大改真连杆:1"
    )
    mirror = gp_Trsf()
    mirror.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))
    for index, (path, shape, location, exact, inherited) in enumerate(leaves):
        if path == "/S6eyou腰:1/雷霆大改真连杆(镜像):1" and shape is None:
            leaves[index] = (
                path,
                BRepBuilderAPI_Transform(right_rod, mirror, True).Shape(),
                location,
                exact,
                inherited,
            )
            empty_paths.remove(path)

    groups = {}
    geometry_only = []
    used_materials = set()
    projected_inertias = {}

    def project_inertia(label, inertia):
        inertia = (inertia + inertia.T) / 2.0
        principal = np.linalg.eigvalsh(inertia)
        if principal[0] >= 0.0 and principal[2] <= principal[0] + principal[1]:
            return inertia
        covariance = np.trace(inertia) / 2.0 * np.eye(3) - inertia
        values, vectors = np.linalg.eigh(covariance)
        covariance = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
        corrected = np.trace(covariance) * np.eye(3) - covariance
        projected_inertias[label] = float(np.linalg.norm(corrected - inertia))
        return corrected

    def json_props(key, location):
        source = materials[key]
        rotation, translation = matrix_from_location(location)
        mass = float(source["mass_kg"])
        com_local_m = np.asarray(source["com_mm"], dtype=float) / 1000.0
        com_cad = rotation @ np.asarray(source["com_mm"], dtype=float) + translation
        values = np.asarray(source["inertia_kg_m2"], dtype=float)
        inertia_origin = np.array(
            [
                [values[0], values[3], values[4]],
                [values[3], values[1], values[5]],
                [values[4], values[5], values[2]],
            ]
        )
        inertia_local = inertia_origin - mass * (
            np.dot(com_local_m, com_local_m) * np.eye(3)
            - np.outer(com_local_m, com_local_m)
        )
        inertia_local = project_inertia(key, inertia_local)
        com_robot = M_ROB @ (com_cad - O_CAD) / 1000.0
        inertia_robot = M_ROB @ rotation @ inertia_local @ rotation.T @ M_ROB.T
        return mass, com_robot, inertia_robot

    def step_props(key, shape):
        source = materials[key]
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        volume = props.Mass()
        if volume <= 0:
            return None
        mass = float(source["mass_kg"])
        center = props.CentreOfMass()
        com_cad = np.array([center.X(), center.Y(), center.Z()])
        matrix = props.MatrixOfInertia()
        inertia_cad = (
            np.array([[matrix.Value(i, j) for j in range(1, 4)] for i in range(1, 4)])
            * (mass / volume)
            * 1e-6
        )
        com_robot = M_ROB @ (com_cad - O_CAD) / 1000.0
        inertia_robot = M_ROB @ inertia_cad @ M_ROB.T
        return mass, com_robot, project_inertia(key, inertia_robot)

    for path, shape, location, exact, inherited in leaves:
        link = classify(path)
        link = split_active_link(link, shape)
        if link is None:
            raise RuntimeError(f"unclassified STEP path: {path}")
        material_path = exact
        material_location = location
        if (
            material_path is None
            and inherited is not None
            and inherited[0] not in used_materials
        ):
            material_path, material_location = inherited
        groups.setdefault(link, []).append(
            {
                "path": path,
                "shape": shape,
                "material_path": material_path,
                "material_location": material_location,
                "resolved_props": None,
            }
        )
        if material_path is not None:
            used_materials.add(material_path)

    residual_sources = {}
    for parts in groups.values():
        for part in parts:
            if part["material_path"] is not None:
                continue
            parent = residual_parent(part["path"])
            if parent is None:
                geometry_only.append(part["path"])
                continue
            parent_mass, parent_com, parent_inertia = json_props(
                parent, node_locations[parent]
            )
            parent_origin_inertia = parent_inertia + parent_mass * (
                np.dot(parent_com, parent_com) * np.eye(3)
                - np.outer(parent_com, parent_com)
            )
            mass = parent_mass
            first_moment = parent_mass * parent_com
            inertia_origin = parent_origin_inertia
            child_keys = sorted(
                key for key in material_leaves if key.startswith(parent + "/")
            )
            for key in child_keys:
                child_mass, child_com, child_inertia = json_props(
                    key, node_locations[key]
                )
                mass -= child_mass
                first_moment -= child_mass * child_com
                inertia_origin -= child_inertia + child_mass * (
                    np.dot(child_com, child_com) * np.eye(3)
                    - np.outer(child_com, child_com)
                )
            if mass <= 0.0:
                raise RuntimeError(
                    f"non-positive residual mass for {part['path']}: {mass}"
                )
            com = first_moment / mass
            inertia = inertia_origin - mass * (
                np.dot(com, com) * np.eye(3) - np.outer(com, com)
            )
            source_label = parent + "#residual"
            inertia = project_inertia(source_label, inertia)
            part["resolved_props"] = (mass, com, inertia)
            part["material_path"] = source_label
            residual_sources[part["path"]] = {
                "parent": parent,
                "subtracted_children": child_keys,
                "mass_kg": mass,
            }

    missing_materials = sorted(material_leaves - used_materials)
    if missing_materials:
        raise RuntimeError(f"JSON leaf paths not found in STEP: {missing_materials}")
    expected_links = set(ORIGINS_CAD)
    if set(groups) != expected_links:
        raise RuntimeError(
            f"link grouping mismatch: missing={sorted(expected_links - set(groups))}, extra={sorted(set(groups) - expected_links)}"
        )

    links = (
        json.load(open(os.path.join(HERE, "links.json"), encoding="utf-8"))["links"]
        if args.links
        else {}
    )
    for link in sorted(args.links if args.links else groups):
        origin_robot = M_ROB @ (np.asarray(ORIGINS_CAD[link]) - O_CAD) / 1000.0
        vertices, faces = [], []
        offset = 0
        mesh_buckets = {}
        parts = []
        sources = []
        for part in groups[link]:
            if part["shape"] is not None:
                part_vertices, part_faces = shape_mesh_robot(
                    part["shape"], origin_robot, coarse=link == "waist_roll_link"
                )
                if len(part_vertices):
                    vertices.append(part_vertices)
                    faces.append(part_faces + offset)
                    offset += len(part_vertices)
                    bucket = (
                        waist_mesh_bucket(part["path"])
                        if link == "waist_roll_link"
                        else "core"
                    )
                    bucket_vertices, bucket_faces = mesh_buckets.setdefault(
                        bucket, ([], [])
                    )
                    bucket_offset = sum(len(item) for item in bucket_vertices)
                    bucket_vertices.append(part_vertices)
                    bucket_faces.append(part_faces + bucket_offset)
            key = part["material_path"]
            if key is None:
                continue
            if part["resolved_props"] is None:
                result = (
                    step_props(key, part["shape"])
                    if part["shape"] is not None
                    else None
                )
                mass, com_robot, inertia_robot = result or json_props(
                    key, part["material_location"]
                )
            else:
                mass, com_robot, inertia_robot = part["resolved_props"]
            parts.append((mass, com_robot, inertia_robot))
            sources.append(key)
        if not vertices or not faces:
            raise RuntimeError(f"empty mesh group: {link}")
        mesh = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)
        step_min = np.full(3, np.inf)
        step_max = np.full(3, -np.inf)
        for part in groups[link]:
            if part["shape"] is None:
                continue
            box = Bnd_Box()
            BRepBndLib.AddOptimal_s(part["shape"], box, False, False)
            if box.IsVoid():
                continue
            lower, upper = box.CornerMin(), box.CornerMax()
            xmin, ymin, zmin = lower.X(), lower.Y(), lower.Z()
            xmax, ymax, zmax = upper.X(), upper.Y(), upper.Z()
            corners = np.array(
                [
                    [x, y, z]
                    for x in (xmin, xmax)
                    for y in (ymin, ymax)
                    for z in (zmin, zmax)
                ]
            )
            robot_corners = (M_ROB @ (corners - O_CAD).T).T / 1000.0
            step_min = np.minimum(step_min, robot_corners.min(axis=0))
            step_max = np.maximum(step_max, robot_corners.max(axis=0))
        mesh_min = mesh.bounds[0] + origin_robot
        mesh_max = mesh.bounds[1] + origin_robot
        bbox_error = float(
            max(
                np.max(np.abs(step_min - mesh_min)), np.max(np.abs(step_max - mesh_max))
            )
        )
        if bbox_error > 2e-3:
            raise RuntimeError(f"STEP/mesh bbox mismatch for {link}: {bbox_error} m")
        mesh_files = []
        bucket_order = [
            "core",
            "torso",
            "right_shoulder_mount",
            "left_shoulder_mount",
        ]
        for bucket in bucket_order:
            if bucket not in mesh_buckets:
                continue
            bucket_vertices, bucket_faces = mesh_buckets[bucket]
            bucket_mesh = trimesh.Trimesh(
                np.vstack(bucket_vertices), np.vstack(bucket_faces), process=False
            )
            stem = link if bucket == "core" else f"{link}_{bucket}"
            for index, start in enumerate(range(0, len(bucket_mesh.faces), 180000)):
                stop = min(start + 180000, len(bucket_mesh.faces))
                chunk = bucket_mesh.submesh(
                    [np.arange(start, stop)], append=True, repair=False
                )
                filename = f"{stem}.STL" if index == 0 else f"{stem}_{index}.STL"
                encoded = bytearray(chunk.export(file_type="stl"))
                encoded[:80] = bytes(80)
                triangle_count = struct.unpack_from("<I", encoded, 80)[0]
                attributes = np.ndarray(
                    (triangle_count,),
                    dtype="<u2",
                    buffer=encoded,
                    offset=132,
                    strides=(50,),
                )
                attributes[:] = 0
                with open(os.path.join(MESH_DIR, filename), "wb") as output:
                    output.write(encoded)
                mesh_files.append(filename)
        mass = sum(part[0] for part in parts)
        com_robot = sum(part[0] * part[1] for part in parts) / mass
        inertia = np.zeros((3, 3))
        for part_mass, part_com, part_inertia in parts:
            displacement = part_com - com_robot
            inertia += part_inertia + part_mass * (
                np.dot(displacement, displacement) * np.eye(3)
                - np.outer(displacement, displacement)
            )
        inertia = (inertia + inertia.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(inertia)
        if eigenvalues[0] <= 0.0:
            raise RuntimeError(f"non-positive inertia for {link}: {eigenvalues}")
        links[link] = {
            "origin": origin_robot.tolist(),
            "origin_cad_mm": ORIGINS_CAD[link],
            "mass": mass,
            "com_link": (com_robot - origin_robot).tolist(),
            "inertia_com": inertia.tolist(),
            "inertia_eigenvalues": eigenvalues.tolist(),
            "nparts": len(groups[link]),
            "mesh_files": mesh_files,
            "bbox_step_robot": [step_min.tolist(), step_max.tolist()],
            "bbox_mesh_robot": [mesh_min.tolist(), mesh_max.tolist()],
            "bbox_max_error_m": bbox_error,
            "fusion_hits": len(sources),
            "fusion_total": len(sources),
            "source_paths": sources,
        }

    source_mass = sum(
        float(materials[key]["mass_kg"]) for key in material_leaves
    ) + sum(source["mass_kg"] for source in residual_sources.values())
    generated_mass = sum(link["mass"] for link in links.values())
    if abs(source_mass - generated_mass) > 1e-10:
        raise RuntimeError(
            f"mass mismatch: JSON={source_mass}, generated={generated_mass}"
        )
    payload = {
        "metadata": {
            "step": os.path.basename(STEP),
            "wrist_topology": "serial_roll_yaw_pitch",
            "wrist_inertia_source": "连杆、圆盘及支架归属 roll；电机、抱箍及末端电阻归属 yaw；按源零件融合完整惯量",
            "dynamics": os.path.basename(MATERIALS),
            "coordinate_map": M_ROB.tolist(),
            "cad_origin_mm": O_CAD.tolist(),
            "top_assembly_scale": 1.0,
            "fusion_hits": len(leaves) - len(geometry_only),
            "fusion_total": len(leaves),
            "geometry_only_paths": geometry_only,
            "empty_step_paths": empty_paths,
            "residual_sources": residual_sources,
            "source_mass_kg": source_mass,
            "generated_mass_kg": generated_mass,
            "part_inertia_reference": "part_local_origin",
            "part_inertia_parallel_axis_shifted": len(material_leaves)
            + len(residual_sources),
            "projected_part_inertias": projected_inertias,
        },
        "links": links,
    }
    with open(os.path.join(HERE, "links.json"), "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    for link, data in sorted(links.items()):
        print(
            f"{link:42s} mass={data['mass']:9.6f} kg parts={data['nparts']:3d} fusion={data['fusion_hits']:3d}"
        )
    print(f"fusion paths: {len(leaves) - len(geometry_only)}/{len(leaves)}")
    print(f"geometry-only STEP leaves: {len(geometry_only)}")
    print(f"mass: JSON={source_mass:.12f} kg generated={generated_mass:.12f} kg")


if __name__ == "__main__":
    main()
