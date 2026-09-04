"""闭链关节的显示态换算。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping

_WAIST_SOLVERS: dict[Path, Any] = {}


def _waist_solver(root_dir: Path) -> Any | None:
    path = (root_dir / "tools" / "waist_closed_chain_ik.py").resolve()
    if not path.is_file():
        return None
    solver = _WAIST_SOLVERS.get(path)
    if solver is not None:
        return solver
    spec = importlib.util.spec_from_file_location("hhtools_semi_taks_waist_ik", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver = module.WaistClosedChainIK(root_dir / "Semi_Taks_LV1.xml")
    _WAIST_SOLVERS[path] = solver
    return solver


def apply_closed_chain_configuration(
    preset: Any,
    configuration: Mapping[str, float],
) -> dict[str, float]:
    block = preset.meta.get("closed_chain")
    if not isinstance(block, Mapping):
        return dict(configuration)
    out = {str(k): float(v) for k, v in configuration.items()}
    affine = block.get("affine")
    if isinstance(affine, Mapping):
        for target, raw in affine.items():
            if not isinstance(raw, Mapping):
                continue
            source = str(raw.get("source") or "")
            if not source:
                continue
            out[str(target)] = (
                float(raw.get("scale", 1.0)) * float(out.get(source, 0.0))
                + float(raw.get("offset", 0.0))
            )
    waist = block.get("waist_parallel")
    if isinstance(waist, Mapping):
        roll = str(waist.get("roll") or "waist_roll_joint")
        pitch = str(waist.get("pitch") or "waist_pitch_joint")
        motors = waist.get("motors")
        if isinstance(motors, (list, tuple)) and len(motors) >= 2:
            solver = _waist_solver(Path(preset.root_dir))
            if solver is not None:
                try:
                    right, left = solver.inverse(
                        float(out.get(roll, 0.0)), float(out.get(pitch, 0.0)),
                    )
                    out[str(motors[0])] = right
                    out[str(motors[1])] = left
                except (ValueError, RuntimeError):
                    pass
    return out


__all__ = ["apply_closed_chain_configuration"]
