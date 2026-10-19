from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hhtools.bodymodels.engine import _amass_smplh_data_struct


class _Struct(SimpleNamespace):
    pass


_FAKE_SMPLX = SimpleNamespace(utils=SimpleNamespace(Struct=_Struct))


def test_amass_smplh_npz_gets_non_pca_hand_placeholders(tmp_path: Path) -> None:
    model_path = tmp_path / "SMPLH_MALE.npz"
    np.savez(model_path, v_template=np.zeros((2, 3), dtype=np.float32))

    data = _amass_smplh_data_struct(model_path, _FAKE_SMPLX, use_pca=False)

    assert data is not None
    assert data.hands_componentsl.shape == (6, 45)
    assert data.hands_componentsr.shape == (6, 45)
    assert data.hands_meanl.shape == (45,)
    assert data.hands_meanr.shape == (45,)


def test_complete_smplh_npz_needs_no_adapter(tmp_path: Path) -> None:
    model_path = tmp_path / "SMPLH_FEMALE.npz"
    np.savez(
        model_path,
        hands_componentsl=np.zeros((6, 45), dtype=np.float32),
        hands_componentsr=np.zeros((6, 45), dtype=np.float32),
        hands_meanl=np.zeros(45, dtype=np.float32),
        hands_meanr=np.zeros(45, dtype=np.float32),
    )

    assert _amass_smplh_data_struct(model_path, _FAKE_SMPLX, use_pca=False) is None


def test_amass_smplh_npz_rejects_pca_mode(tmp_path: Path) -> None:
    model_path = tmp_path / "SMPLH_NEUTRAL.npz"
    np.savez(model_path, v_template=np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="without MANO hand PCA components"):
        _amass_smplh_data_struct(model_path, _FAKE_SMPLX, use_pca=True)
