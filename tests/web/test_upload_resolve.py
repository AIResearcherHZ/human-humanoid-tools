from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hhtools.web import upload_resolve


def test_amass_upload_preserves_missing_weight_error(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "run.npz"
    np.savez(path, poses=np.zeros((2, 72), dtype=np.float32), trans=np.zeros((2, 3)))

    def fail_adapter(path: Path, dataset: str, *, progress=None):
        if dataset == "amass":
            raise FileNotFoundError("No SMPLH (male) weight file found")
        raise ValueError("not this NPZ schema")

    monkeypatch.setattr(upload_resolve, "_load_via_dataset_adapter", fail_adapter)

    with pytest.raises(ValueError, match=r"No SMPLH \(male\) weight file found"):
        upload_resolve._load_mimic(
            path,
            load_motion_file=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("nope")),
            load_via_adapter=lambda _path: (None, None),
        )
