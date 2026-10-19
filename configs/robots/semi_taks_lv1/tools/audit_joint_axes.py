import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Circle
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from step_export_links import M_ROB, ORIGINS_CAD, classify

STEP = os.path.join(os.path.dirname(__file__), "..", "Semi_Taks_LV1.step")
SOURCES = json.load(
    open(os.path.join(os.path.dirname(__file__), "..", "Semi_Taks_LV1.json"))
)


def name(label):
    value = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), value):
        return value.Get().ToExtString()
    return "?"


def main():
    document = TDocStd_Document(TCollection_ExtendedString("axis_audit"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(STEP) != IFSelect_RetDone:
        raise RuntimeError(STEP)
    reader.Transfer(document)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    groups = defaultdict(list)
    errors = []
    bearing_axes = defaultdict(set)

    def walk(label, location, path):
        accumulated = location.Multiplied(tool.GetLocation_s(label))
        target = label
        if tool.IsReference_s(label):
            referred = TDF_Label()
            tool.GetReferredShape_s(label, referred)
            target = referred
        path += "/" + (name(label) if name(label) != "?" else name(target))
        if tool.IsAssembly_s(target):
            children = TDF_LabelSequence()
            tool.GetComponents_s(target, children)
            for index in range(1, children.Length() + 1):
                walk(children.Value(index), accumulated, path)
            return
        shape = tool.GetShape_s(target).Located(
            accumulated.Multiplied(tool.GetShape_s(target).Location())
        )
        source_path = path[path.find("/", 1) :]
        if source_path in SOURCES:
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape, props)
            volume = props.Mass()
            if volume > 1e-6:
                center = props.CentreOfMass()
                cad_com = np.array([center.X(), center.Y(), center.Z()])
                source = SOURCES[source_path]
                rotation = accumulated.Multiplied(
                    tool.GetShape_s(target).Location()
                ).Transformation()
                point = source["com_mm"]
                expected = np.array(
                    [
                        sum(rotation.Value(i, j) * point[j - 1] for j in range(1, 4))
                        + rotation.TranslationPart().Coord(i)
                        for i in range(1, 4)
                    ]
                )
                errors.append(
                    (
                        np.linalg.norm(cad_com - expected),
                        source_path,
                        cad_com,
                        expected,
                        volume,
                    )
                )
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(explorer.Current()))
            if curve.GetType() == GeomAbs_Circle:
                circle = curve.Circle()
                center = circle.Location()
                direction = circle.Axis().Direction()
                point = np.array([center.X(), center.Y(), center.Z()])
                axis = np.array([direction.X(), direction.Y(), direction.Z()])
                radius = circle.Radius()
                if 1 <= radius <= 80:
                    link = classify(source_path)
                    if link:
                        groups[link].append((radius, point, axis))
                    if "鱼眼轴承" in source_path and radius >= 2:
                        bearing_axes[source_path].add(
                            (
                                round(radius, 2),
                                *np.round(point, 2),
                                *np.round(np.abs(axis), 2),
                            )
                        )
            explorer.Next()

    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    for index in range(1, roots.Length() + 1):
        walk(roots.Value(index), TopLoc_Location(), "")
    for error, path, actual, expected, volume in sorted(errors, reverse=True):
        if error <= 0.1:
            break
        print(f"COM error={error:.3f} mm volume={volume:.3f} {path}")
        print(f"  STEP={np.round(actual, 3)} JSON={np.round(expected, 3)}")
    for path, circles in bearing_axes.items():
        print("BEARING", path, sorted(circles, reverse=True)[:12])
    root = ET.parse(
        os.path.join(os.path.dirname(__file__), "..", "Semi_Taks_LV1.urdf")
    ).getroot()
    for joint in root.findall("joint"):
        child = joint.find("child").attrib["link"]
        parent = joint.find("parent").attrib["link"]
        if child not in ORIGINS_CAD or parent not in ORIGINS_CAD:
            continue
        axis_node = joint.find("axis")
        if axis_node is None:
            continue
        axis = M_ROB.T @ np.fromstring(axis_node.attrib["xyz"], sep=" ")
        origin = np.asarray(ORIGINS_CAD[child])
        candidates = []
        for radius, point, direction in groups[child] + groups[parent]:
            if abs(np.dot(axis, direction)) < 0.99:
                continue
            distance = np.linalg.norm(np.cross(origin - point, direction))
            candidates.append((distance, radius, point))
        if candidates:
            distance, radius, point = min(candidates, key=lambda item: item[0])
            print(
                f"AXIS {child} gap={distance:.3f} mm r={radius:.2f} point={np.round(point, 2)}"
            )
            if distance > 0.1:
                raise RuntimeError(
                    f"joint axis deviates from STEP: {child}: {distance} mm"
                )
        else:
            raise RuntimeError(f"no STEP circular reference for {child}")


if __name__ == "__main__":
    main()
