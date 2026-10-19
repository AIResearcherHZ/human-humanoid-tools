import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "tools" / "build_urdf_mjcf.py"))
