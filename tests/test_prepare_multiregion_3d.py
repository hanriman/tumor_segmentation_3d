import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from prepare_data_3d import BRATS_GLI_LABELS, ET_LABEL, process_patient


def _make_patient(root, pid="TEST-001", labels=(0, 1, 2, 3, 4), shape=(24, 24, 24)):
    d = root / pid
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for mod in ("t1n", "t1c", "t2w", "t2f"):
        img = np.abs(rng.standard_normal(shape)).astype(np.float32) + 0.5
        nib.save(nib.Nifti1Image(img, np.eye(4)), d / f"{pid}-{mod}.nii.gz")
    seg = np.zeros(shape, dtype=np.float32)
    # paint each label into a distinct octant so resampling preserves them
    q = shape[0] // 2
    for i, lab in enumerate(labels[1:], start=0):
        sl = tuple(slice(0, q) if (i >> b) & 1 == 0 else slice(q, None) for b in range(3))
        seg[sl] = lab
    nib.save(nib.Nifti1Image(seg, np.eye(4)), d / f"{pid}-seg.nii.gz")
    return d


def test_multiregion_preserves_labels(tmp_path):
    pdir = _make_patient(tmp_path / "raw")
    out = tmp_path / "out"
    out.mkdir()
    rec = process_patient(pdir, out, target_size=16, dtype="float32", label_protocol="multiregion")
    assert rec is not None
    z = np.load(out / f"{pdir.name}_volume.npz")
    assert z["image"].shape == (4, 16, 16, 16)
    vals = set(np.unique(z["mask"]).tolist())
    assert vals <= set(BRATS_GLI_LABELS), vals
    assert vals >= {1, 2, 3, 4}, f"octant labels lost in resampling: {vals}"
    assert rec["has_et"] is True and rec["num_voxels_et"] > 0
    assert ET_LABEL == 4
    # WT count identical to binary mode on the same patient
    out2 = tmp_path / "out2"
    out2.mkdir()
    rec2 = process_patient(pdir, out2, target_size=16, dtype="float32", label_protocol="wt")
    assert rec2["num_voxels_tumor"] == rec["num_voxels_tumor"]


def test_multiregion_rejects_stray_labels(tmp_path):
    import pytest

    pdir = _make_patient(tmp_path / "raw")
    # corrupt one voxel with an illegal label
    seg_path = pdir / f"{pdir.name}-seg.nii.gz"
    img = nib.load(str(seg_path))
    data = img.get_fdata().copy()
    data[0, 0, 0] = 7.0
    nib.save(nib.Nifti1Image(data, img.affine), str(seg_path))
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="unexpected segmentation labels"):
        process_patient(pdir, out, target_size=16, label_protocol="multiregion")


def test_multiregion_et_free_volume(tmp_path):
    pdir = _make_patient(tmp_path / "raw", pid="TEST-NOET", labels=(0, 1, 2))
    out = tmp_path / "out"
    out.mkdir()
    rec = process_patient(pdir, out, target_size=16, dtype="float32", label_protocol="multiregion")
    assert rec["has_et"] is False and rec["num_voxels_et"] == 0
    assert rec["has_tumor"] is True  # WT still present


def test_resample_preserve_labels_unit():
    from prepare_data_3d import resample_3d_volume

    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[2:5, 2:5, 2:5] = 4.0
    vol[6:9, 6:9, 6:9] = 1.0
    out = resample_3d_volume(vol, (8, 8, 8), is_mask=True, preserve_labels=True)
    assert set(np.unique(out).tolist()) <= {0, 1, 4}
    assert (out == 4).sum() > 0 and (out == 1).sum() > 0
    # legacy path still binarizes
    out_bin = resample_3d_volume(vol, (8, 8, 8), is_mask=True)
    assert set(np.unique(out_bin).tolist()) <= {0, 1}
    _ = torch.zeros(1)  # keep torch import used (device parity check downstream)
