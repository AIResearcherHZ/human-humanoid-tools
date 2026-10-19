"""URDF loader + URDF→MJCF transpile for hhtools robot presets.

Pipeline (single function :func:`load_robot`):

1. Resolve the ``RobotPreset`` (from the registry or a direct path).
2. Parse the URDF with :mod:`yourdfpy` — we rely on its trimesh-based scene
   output for UI rendering and its link/joint graph for topology.
3. Compile the URDF into a MuJoCo :class:`mujoco.MjModel` — this both serves
   as a parse validator (mujoco is stricter than yourdfpy and catches things
   like non-unit inertia matrices early) and produces the derived ``.xml``
   artefact Newton's MJCF backend can consume.
4. Pack everything into a concrete :class:`URDFRobotModel`.

Design notes:

* URDF→MJCF is done via ``mujoco.MjModel.from_xml_path(...).save_xml(...)``;
  we don't shell out or depend on ``urdf2mjcf``.  This keeps the conversion
  in the same process and reports errors through the standard mujoco C++
  exception pipeline.
* yourdfpy can produce an empty trimesh scene when the URDF references mesh
  files we can't locate.  :func:`_resolve_mesh_search_paths` tries the preset's
  declared search paths first, then the URDF file's directory, then any
  ``meshes/`` sibling — enough for the RP1 layout (``urdf/rp1.urdf`` with
  ``../meshes/``) and the "flat" layout (``robot.urdf`` with sibling ``meshes/``).
* We never mutate the URDF file; the MJCF output is written to a temp file
  if callers need the path (e.g. for ``newton.ModelBuilder().add_mjcf(...)``),
  otherwise the XML string is kept in-memory on :class:`URDFRobotModel`.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import yourdfpy

from hhtools.robot.base import JointInfo, LinkInfo, RobotModel, RobotPreset

if TYPE_CHECKING:
    import trimesh

_log = logging.getLogger(__name__)


@dataclass
class URDFRobotModel(RobotModel):
    """Concrete :class:`RobotModel` backed by yourdfpy + mujoco.

    Fields are populated by :func:`load_robot`; the dataclass itself is a
    dumb container so we can pickle it, cache it, etc.
    """

    _preset: RobotPreset
    _links: tuple[LinkInfo, ...]
    _joints: tuple[JointInfo, ...]
    _base_link: str
    _actuated: tuple[JointInfo, ...]

    urdf: yourdfpy.URDF = field(repr=False)

    mjcf_xml: str = field(default="", repr=False)
    mujoco_model: mujoco.MjModel | None = field(default=None, repr=False)

    @property
    def preset(self) -> RobotPreset:
        return self._preset

    @property
    def links(self) -> tuple[LinkInfo, ...]:
        return self._links

    @property
    def joints(self) -> tuple[JointInfo, ...]:
        return self._joints

    @property
    def base_link(self) -> str:
        return self._base_link

    @property
    def actuated_joints(self) -> tuple[JointInfo, ...]:
        return self._actuated

    def trimesh_scene(self, *, collision: bool = False) -> "trimesh.Scene":
        """Return the URDF's trimesh scene at the current joint configuration.
        ``collision=False`` (default) uses visual meshes; ``True`` uses the
        URDF ``<collision>`` geometries — usually a cheap decomposition meant
        for the physics solver, handy if the viewer wants to show bounds
        without hitting a multi-million-tri visual mesh.
        """
        return self.urdf.scene if not collision else self.urdf.collision_scene

    def zero_configuration(self) -> dict[str, float]:
        """All actuated joints at 0 — the canonical T-pose for most humanoids.
        Returned as ``{joint_name: q}`` so callers can pass it verbatim to
        ``urdf.update_cfg(...)`` without worrying about DOF ordering.
        """
        return {j.name: 0.0 for j in self._actuated}

    def apply_configuration(self, q: dict[str, float] | np.ndarray) -> None:
        """Set actuated joint angles.
        ``q`` can be:
        * a ``dict`` keyed by joint name (missing joints default to 0), or
        * a 1-D array / sequence in the order returned by :meth:`dof_names`.
        Mutates ``self.urdf`` in place — the trimesh scene returned from
        :meth:`trimesh_scene` reflects the change immediately (yourdfpy
        invalidates transforms on update).
        """
        if isinstance(q, dict):
            cfg = {
                j.name: float(q.get(j.name, 0.0))
                for j in self._joints if j.is_actuated
            }
            self.urdf.update_cfg(cfg)
            return
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.size != len(self._actuated):
            raise ValueError(
                f"configuration array has {arr.size} elements but robot "
                f"has {len(self._actuated)} actuated joints"
            )
        cfg = {j.name: 0.0 for j in self._joints if j.is_actuated}
        cfg.update({j.name: float(v) for j, v in zip(self._actuated, arr, strict=True)})
        self.urdf.update_cfg(cfg)

    def write_mjcf(self, path: str | Path | None = None) -> Path:
        """Persist the MJCF XML to ``path`` (or a temp file) and return the path.
        Newton's MJCF backend (``ModelBuilder.add_mjcf``) requires a file
        path, not a string — this method is the bridge.  Temp-file outputs
        are tagged with the robot name so they're obvious in ``/tmp``.
        """
        if not self.mjcf_xml:
            raise RuntimeError(
                "mjcf_xml is empty; this URDFRobotModel was loaded without "
                "compile_mjcf=True and no MJCF is available"
            )
        if path is None:
            fd, tmp = tempfile.mkstemp(prefix=f"hhtools_{self._preset.name}_", suffix=".mjcf.xml")
            import os

            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fp.write(self.mjcf_xml)
            return Path(tmp)
        out = Path(path)
        out.write_text(self.mjcf_xml, encoding="utf-8")
        return out


def load_robot(
    preset: RobotPreset,
    *,
    compile_mjcf: bool = True,
    build_collision_scene: bool = False,
) -> URDFRobotModel:
    """Materialise a :class:`RobotPreset` into a live :class:`URDFRobotModel`.

    Args:
        preset: A :class:`RobotPreset` — typically obtained from
            :func:`hhtools.robot.registry.get`.
        compile_mjcf: Whether to also compile the URDF into an MJCF string.
            ``True`` is the default because (a) MuJoCo's parser is a better
            validator than yourdfpy's (catches zero-mass links, malformed
            inertias, non-unit axes), and (b) Newton's IK / simulation path
            needs MJCF anyway.  Set ``False`` for UI-only code paths on
            underpowered machines.
        build_collision_scene: Pass-through to :class:`yourdfpy.URDF.load` —
            disabled by default because collision scenes are ~5× slower to
            load and we don't need them for the viewer.

    Raises:
        FileNotFoundError: when ``preset.urdf_path`` is missing or None.
            Registry-level presets with ``has_urdf=False`` short-circuit here
            so the viewer can render the preset as "awaiting URDF" without
            needing to guard every call site.
        RuntimeError: when yourdfpy fails to parse the URDF or mujoco fails
            to compile the MJCF.  The original exception is chained for
            diagnosis.
    """
    if preset.urdf_path is None or not preset.urdf_path.is_file():
        raise FileNotFoundError(
            f"robot preset {preset.name!r} has no URDF on disk "
            f"(expected at {preset.urdf_path}).  Populate the directory "
            f"or remove the preset.  See configs/robots/_template/README.md."
        )

    if preset.ik_map:
        try:
            from hhtools.robot.kinematics import validate_ik_map

            for issue in validate_ik_map(preset.urdf_path, preset.ik_map):
                _log.warning("robot %r ik_map: %s", preset.name, issue.format())
        except Exception:
            pass

    search_paths = _resolve_mesh_search_paths(preset)

    try:
        urdf = yourdfpy.URDF.load(
            str(preset.urdf_path),
            mesh_dir=search_paths[0] if search_paths else None,


            filename_handler=_build_filename_handler(preset.urdf_path, search_paths),
            build_collision_scene_graph=build_collision_scene,
            load_meshes=True,
            build_scene_graph=True,
        )
    except Exception as err:
        raise RuntimeError(
            f"yourdfpy failed to parse {preset.urdf_path}: {err}"
        ) from err

    links = _collect_links(urdf)
    joints = _collect_joints(urdf)
    base = _find_base_link(urdf, links)
    actuated = _order_actuated(joints, preset.dof_order)

    mjcf_xml = ""
    mujoco_model: mujoco.MjModel | None = None
    if compile_mjcf:
        try:
            mujoco_model, mjcf_xml = _compile_mjcf(preset.urdf_path, search_paths)
        except Exception as err:


            _log.warning(
                "URDF→MJCF compilation failed for %s: %s.  UI rendering will "
                "still work but Newton simulation/IK will be unavailable.",
                preset.name, err,
            )
            mujoco_model = None
            mjcf_xml = ""

    return URDFRobotModel(
        _preset=preset,
        _links=links,
        _joints=joints,
        _base_link=base,
        _actuated=actuated,
        urdf=urdf,
        mjcf_xml=mjcf_xml,
        mujoco_model=mujoco_model,
    )


def _resolve_mesh_search_paths(preset: RobotPreset) -> list[Path]:
    """Best-effort list of directories to look for URDF mesh files in.

    Order of precedence:
    1. ``robot.yaml:mesh_search_paths`` (user intent wins).
    2. The URDF's own parent directory (plain flat layout).
    3. ``<urdf_parent>/../meshes`` (RP1-style ``urdf/rp1.urdf`` + ``../meshes``).
    4. ``<robot_root>/meshes`` (hhtools preset convention).

    Duplicates and non-existent entries are filtered out so the caller can
    pass the result to yourdfpy without worrying about ``FileNotFoundError``.
    """
    if preset.urdf_path is None:
        return []
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        try:
            resolved = p.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        if resolved.is_dir():
            roots.append(resolved)
            seen.add(resolved)

    for p in preset.mesh_search_paths:
        _add(p)
    urdf_parent = preset.urdf_path.parent
    _add(urdf_parent)
    _add(urdf_parent.parent)
    _add(urdf_parent.parent / "meshes")
    _add(urdf_parent.parent / "mesh")
    _add(urdf_parent.parent / "convex")
    _add(urdf_parent / "assets")
    _add(urdf_parent.parent / "assets")
    _add(preset.root_dir / "meshes")
    return roots


def _build_filename_handler(urdf_path: Path, search_paths: list[Path]):
    """Yourdfpy filename handler that walks our full search-path list.

    Yourdfpy's default handler only looks at the single ``mesh_dir`` it's
    passed — that's not enough for URDFs that mix ``package://`` and bare
    relative paths (rp1 does exactly that).  Our handler:

    1. strips any ``package://<pkg>/`` prefix (URDF convention — we treat
       the package name as a no-op since our meshes are shipped alongside);
    2. tries each search path in order;
    3. falls back to the raw path (so absolute paths still work).
    """

    def _handler(fname: str) -> str:
        if fname.startswith("package://"):

            rest = fname[len("package://"):]
            _pkg, _, tail = rest.partition("/")
            fname = tail or rest
        candidate = Path(fname)
        if candidate.is_absolute() and candidate.is_file():
            return str(candidate)
        for root in [*search_paths, urdf_path.parent]:
            trial = root / fname
            if trial.is_file():
                return str(trial)


            trial_meshes = root / "meshes" / Path(fname).name
            if trial_meshes.is_file():
                return str(trial_meshes)
        return fname

    return _handler


def _collect_links(urdf: yourdfpy.URDF) -> tuple[LinkInfo, ...]:
    """Extract ordered :class:`LinkInfo` records from a yourdfpy URDF.

    Order matches URDF parse order (``urdf.robot.links``), which is what
    mujoco / Newton also use — keeping them aligned means "link index 3"
    means the same thing across tools.
    """
    out: list[LinkInfo] = []

    parent_of: dict[str, str | None] = {}
    children: dict[str, list[str]] = {}
    for joint in urdf.robot.joints:
        parent_of[joint.child] = joint.parent
        children.setdefault(joint.parent, []).append(joint.name)
    for link in urdf.robot.links:
        out.append(
            LinkInfo(
                name=link.name,
                parent=parent_of.get(link.name),
                child_joint_names=tuple(children.get(link.name, ())),
            )
        )
    return tuple(out)


def _collect_joints(urdf: yourdfpy.URDF) -> tuple[JointInfo, ...]:
    """Extract ordered :class:`JointInfo` records from a yourdfpy URDF."""
    out: list[JointInfo] = []
    for joint in urdf.robot.joints:
        axis = tuple(float(x) for x in (joint.axis if joint.axis is not None else (0.0, 0.0, 1.0)))
        lim = joint.limit
        out.append(
            JointInfo(
                name=joint.name,
                joint_type=joint.type,
                parent_link=joint.parent,
                child_link=joint.child,
                axis=axis,
                limit_lower=getattr(lim, "lower", None) if lim is not None else None,
                limit_upper=getattr(lim, "upper", None) if lim is not None else None,
                velocity_limit=getattr(lim, "velocity", None) if lim is not None else None,
                effort_limit=getattr(lim, "effort", None) if lim is not None else None,
            )
        )
    return tuple(out)


def _find_base_link(
    urdf: yourdfpy.URDF, links: tuple[LinkInfo, ...],
) -> str:
    """Root of the URDF kinematic tree — the one link with no parent.

    URDF requires exactly one; yourdfpy already rejects URDFs with multiple
    roots at parse time, so this is just bookkeeping.
    """
    try:
        return urdf.base_link
    except AttributeError:
        pass
    roots = [link for link in links if link.parent is None]
    if len(roots) != 1:
        raise RuntimeError(
            f"URDF has {len(roots)} root link(s); expected exactly 1.  "
            f"Roots: {[r.name for r in roots]}"
        )
    return roots[0].name


def _order_actuated(
    joints: tuple[JointInfo, ...], dof_order: tuple[str, ...],
) -> tuple[JointInfo, ...]:
    """Return actuated joints in the order declared by ``robot.yaml.dof_order``.

    Behaviour:
    * ``dof_order`` empty → fall back to URDF parse order (filter out fixed).
      Good enough for quick experiments, but flagged in the CSV header comment
      so users know their export order isn't pinned.
    * ``dof_order`` non-empty → each name must match an actuated joint; any
      unknown name raises, any missing actuated joint raises.  This is
      intentional: a partial ``dof_order`` silently skipping DOFs would
      produce CSVs that look fine but fail later in IK, and those bugs are
      horrible to debug.
    """
    actuated_all = [j for j in joints if j.is_actuated]
    if not dof_order:
        return tuple(actuated_all)
    by_name = {j.name: j for j in actuated_all}
    all_by_name = {j.name: j for j in joints}
    ordered: list[JointInfo] = []
    for name in dof_order:
        if name in by_name:
            ordered.append(by_name[name])
            continue
        joint = all_by_name.get(name)
        if joint is not None and not joint.is_actuated:
            _log.warning(
                "robot.yaml dof_order skips non-scalar joint %r (type=%s); "
                "floating/planar roots belong in qpos[0:7], not dof_order",
                name,
                joint.joint_type,
            )
            continue
        raise ValueError(
            f"robot.yaml dof_order references joint {name!r} which is "
            f"not an actuated joint in the URDF.  Available: "
            f"{sorted(by_name)}"
        )

    missing = set(by_name) - set(dof_order)
    if missing:
        _log.info(
            "robot %s: %d actuated joint(s) present in URDF but not in "
            "dof_order (will be omitted from CSV export): %s",


            "<unknown>", len(missing), sorted(missing),
        )
    return tuple(ordered)


def _mesh_filename_prefix(filename: str) -> str | None:
    fname = filename.replace("\\", "/").strip()
    if fname.startswith("./"):
        fname = fname[2:]
    if "/" not in fname:
        return None
    return fname.rsplit("/", 1)[0]


def _detect_mesh_subdir(tree_root, urdf_dir: Path) -> str | None:
    """Return the common subdirectory prefix shared by all mesh filenames.

    If every ``<mesh filename="..."/>`` in the URDF uses the same directory
    prefix (e.g. ``meshes/foo.STL`` or ``./meshes/foo.STL``) and that
    directory exists under *urdf_dir*, return the normalised prefix string
    (``"meshes"``).  Otherwise ``None``.
    """
    prefixes: set[str] = set()
    for mesh in tree_root.iter("mesh"):
        fname = mesh.get("filename", "")
        prefix = _mesh_filename_prefix(fname)
        if prefix is None:
            continue
        candidate = urdf_dir / prefix
        if candidate.is_dir():
            prefixes.add(prefix)
    if len(prefixes) == 1:
        return prefixes.pop()
    return None


def _compile_mjcf(
    urdf_path: Path, search_paths: list[Path],
) -> tuple[mujoco.MjModel, str]:
    """Compile a URDF to MJCF with portable mesh resolution.

    Mesh paths are repaired via :func:`hhtools.robot.urdf_normalize.ensure_urdf_meshes_resolvable`
    (bare ``link.STL`` → ``meshes/link.STL``, ``meshdir`` doubling, etc.) before
    MuJoCo loads the file.  When the preset directory is writable the fixed URDF
    is persisted so the next load does not repeat the repair.
    """
    import os
    import tempfile

    from hhtools.robot.urdf_normalize import (
        default_mesh_search_dirs,
        ensure_urdf_meshes_resolvable,
        urdf_needs_persisted_fixes,
    )

    dirs = list(search_paths) if search_paths else default_mesh_search_dirs(urdf_path)
    urdf_dir = urdf_path.parent
    needs_persist = urdf_needs_persisted_fixes(urdf_path)

    if not needs_persist:
        ensure_urdf_meshes_resolvable(urdf_path, search_dirs=dirs)
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        return model, _dump_mjcf(model)

    persist_target: Path | None = urdf_path
    try:
        probe = urdf_dir / ".hhtools_write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        persist_target = None

    fd, tmp = tempfile.mkstemp(
        prefix=f"hhtools_urdf_{urdf_path.stem}_",
        suffix=".urdf",
        dir=str(urdf_dir),
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        ensure_urdf_meshes_resolvable(
            urdf_path,
            search_dirs=dirs,
            output_path=tmp_path,
        )
        if persist_target is not None and persist_target != tmp_path:
            try:
                tmp_path.replace(persist_target)
            except OSError:
                compile_path = tmp_path
            else:
                compile_path = persist_target
        else:
            compile_path = tmp_path
        model = mujoco.MjModel.from_xml_path(str(compile_path))
        return model, _dump_mjcf(model)
    finally:
        if tmp_path.is_file() and tmp_path != urdf_path:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _dump_mjcf(model: mujoco.MjModel) -> str:
    """Serialise a compiled :class:`mujoco.MjModel` to its MJCF XML string.

    mujoco exposes this via ``mj_saveLastXML`` (which requires a temp file)
    — we capture to a temp file and return the string so callers don't need
    to manage the intermediate.  The XML round-trips: the same ``from_xml``
    call on the output produces an equivalent model.
    """
    import os

    fd, tmp = tempfile.mkstemp(suffix=".mjcf.xml", prefix="hhtools_mjcf_")
    os.close(fd)
    try:
        mujoco.mj_saveLastXML(tmp, model)
        return Path(tmp).read_text(encoding="utf-8")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
