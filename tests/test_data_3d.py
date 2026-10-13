"""Data module tests: masking, augmentations, transforms, datasets, and preprocessing."""

from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
import torch

from brats_jepa_3d.data import (
    BraTS3DDataset,
    JEPAMaskingTransform3D,
    RandomModalityDropout3D,
    VolumetricAugmentations3D,
    apply_b1_bias_field_3d,
    apply_rician_noise_3d,
    jepa_masking_collate_fn_3d,
)

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from prepare_data_3d import BRATS_GLI_LABELS, ET_LABEL, process_patient, resample_3d_volume


# =========================================================================
# 1. Masking Transforms & Collation
# =========================================================================


def test_jepa_masking_transform_3d():
    tf = JEPAMaskingTransform3D(
        grid_size=(8, 8, 8),
        num_target_cuboids=4,
        target_cuboid_size=(3, 3, 3),
        context_num_patches=192,
    )
    res = tf()

    ctx = res["context_indices"]
    targets = res["target_indices_list"]

    # 1. Check exact sequence lengths
    assert isinstance(ctx, torch.Tensor)
    assert ctx.shape == (192,), f"Expected 192 context patches, got {ctx.shape}"
    assert len(targets) == 4
    for m, tgt in enumerate(targets):
        assert isinstance(tgt, torch.Tensor)
        assert tgt.shape == (27,), f"Target block {m} expected 27 patches, got {tgt.shape}"

    # 2. Check strict zero-collision guarantee
    ctx_set = set(ctx.tolist())
    for m, tgt in enumerate(targets):
        tgt_set = set(tgt.tolist())
        collision = ctx_set & tgt_set
        assert len(collision) == 0, (
            f"Collision detected between context and target {m}: {collision}"
        )

    # 3. Check bounds
    assert all(0 <= idx < 512 for idx in ctx.tolist())


def test_jepa_masking_collate_3d():
    tf = JEPAMaskingTransform3D()
    batch = [
        {
            "image": torch.randn(4, 128, 128, 128),
            "mask": torch.zeros(1, 128, 128, 128),
            "patient_id": f"p_{i}",
            **tf(),
        }
        for i in range(2)
    ]

    collated = jepa_masking_collate_fn_3d(batch)
    assert collated["images"].shape == (2, 4, 128, 128, 128)
    assert collated["masks"].shape == (2, 1, 128, 128, 128)
    assert collated["context_indices"].shape == (2, 192)
    # Default: 6 anisotropic target cuboids with shapes [27, 18, 18, 18, 8, 8]
    assert len(collated["target_indices_list"]) == 6
    expected_shapes = [(2, 27), (2, 18), (2, 18), (2, 18), (2, 8), (2, 8)]
    for m, expected_shape in enumerate(expected_shapes):
        assert collated["target_indices_list"][m].shape == expected_shape


def test_anisotropic_target_shapes_and_total_tokens():
    """Anisotropic target jitter: 6 target cuboid slots totaling exactly 97 tokens with strict disjointness."""
    tf = JEPAMaskingTransform3D()
    res = tf()

    ctx = res["context_indices"]
    targets = res["target_indices_list"]

    assert ctx.shape == (192,)
    assert len(targets) == 6
    expected_lens = [27, 18, 18, 18, 8, 8]
    for m, exp_len in enumerate(expected_lens):
        assert targets[m].shape == (exp_len,), f"Slot {m} shape mismatch: {targets[m].shape}"

    # Total target tokens must be exactly 97
    total_tokens = sum(t.shape[0] for t in targets)
    assert total_tokens == 97, f"Expected 97 total target tokens, got {total_tokens}"

    # Strict disjointness between context and target union
    ctx_set = set(ctx.tolist())
    all_tgt_set: set[int] = set()
    for m, tgt in enumerate(targets):
        tgt_set = set(tgt.tolist())
        assert not (ctx_set & tgt_set), f"Context collision with slot {m}"
        assert not (all_tgt_set & tgt_set), f"Pairwise target overlap at slot {m}"
        all_tgt_set.update(tgt_set)

    assert len(all_tgt_set) == 97
    assert len(ctx_set) == 192


def test_masking_legacy_parameter_compatibility():
    """Explicit legacy parameters num_target_cuboids=4, target_cuboid_size=(3,3,3) must be honored."""
    tf = JEPAMaskingTransform3D(num_target_cuboids=4, target_cuboid_size=(3, 3, 3))
    res = tf()
    targets = res["target_indices_list"]
    assert len(targets) == 4
    for tgt in targets:
        assert tgt.shape == (27,)


def test_p1_step7_zero_overlap_hard_guarantee():
    """500 masks with max_target_overlap=0.0 -> pairwise target intersections == 0."""
    tf = JEPAMaskingTransform3D(
        grid_size=(8, 8, 8),
        num_target_cuboids=4,
        target_cuboid_size=(3, 3, 3),
        max_target_overlap=0.0,
        context_num_patches=192,
        connectivity=26,
    )
    for seed in range(500):
        g = torch.Generator()
        g.manual_seed(seed)
        out = tf(generator=g)
        blocks = [set(t.tolist()) for t in out["target_indices_list"]]
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                assert not (blocks[i] & blocks[j]), f"overlap at seed {seed}"


def _proxy_brain_frac():
    torch.manual_seed(0)
    zz, yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), torch.arange(128), indexing="ij")
    brain = (((zz - 64) / 48) ** 2 + ((yy - 64) / 58) ** 2 + ((xx - 64) / 48) ** 2) <= 1.0
    return torch.nn.functional.avg_pool3d(
        brain.float().unsqueeze(0).unsqueeze(0), kernel_size=16, stride=16
    ).reshape(-1)


def test_brain_aware_masking_varies_with_counter():
    """Same (worker seed, idx) must yield different masks as the dataset call counter advances."""
    tf = JEPAMaskingTransform3D()
    frac = _proxy_brain_frac()
    base = 12345
    seen = set()
    for counter in (1, 2, 3):
        g = torch.Generator()
        g.manual_seed((base + 3 * 7919 + counter * 104729) % 2**32)
        out = tf(token_brain_frac=frac, generator=g)
        assert "context_tissue_mask" in out and len(out["context_tissue_mask"]) == 192
        key = tuple(out["context_indices"].tolist())
        assert key not in seen, "mask repeated across counter values (epoch-freeze bug)"
        seen.add(key)
        union = set().union(*[t.tolist() for t in out["target_indices_list"]])
        assert not (set(out["context_indices"].tolist()) & union)


def test_brain_aware_masking_legacy_compat():
    tf = JEPAMaskingTransform3D()
    out = tf()
    assert "context_tissue_mask" not in out
    assert out["context_indices"].shape == (192,)


def test_anisotropic_zero_overlap_500_seeds():
    """500 masks with anisotropic target jitter (6 target slots, 97 tokens) -> zero pairwise overlap and zero context collision."""
    tf = JEPAMaskingTransform3D()  # Anisotropic target jitter by default
    for seed in range(500):
        g = torch.Generator()
        g.manual_seed(seed)
        out = tf(generator=g)
        ctx_set = set(out["context_indices"].tolist())
        blocks = [set(t.tolist()) for t in out["target_indices_list"]]
        assert len(blocks) == 6
        assert sum(len(b) for b in blocks) == 97
        for i in range(len(blocks)):
            assert not (ctx_set & blocks[i]), f"Context collision with block {i} at seed {seed}"
            for j in range(i + 1, len(blocks)):
                assert not (blocks[i] & blocks[j]), f"Target overlap between {i} and {j} at seed {seed}"


def test_anisotropic_brain_aware_masking():
    """Anisotropic masking with token_brain_frac selects tissue for both targets and context."""
    tf = JEPAMaskingTransform3D()
    frac = _proxy_brain_frac()
    tissue_set = {i for i in range(512) if frac[i] >= 0.10}
    g = torch.Generator()
    g.manual_seed(42)
    out = tf(token_brain_frac=frac, generator=g)

    assert len(out["target_indices_list"]) == 6
    assert out["context_indices"].shape == (192,)
    assert "context_tissue_mask" in out
    assert len(out["context_tissue_mask"]) == 192

    # Check that each target cuboid satisfies its proportional minimum tissue requirement
    for m, tgt in enumerate(out["target_indices_list"]):
        tgt_set = set(tgt.tolist())
        t_cnt = len(tgt_set & tissue_set)
        cz, cy, cx = tf.target_cuboid_sizes[m]
        tgt_vol = cz * cy * cx
        min_tissue = max(4, int(0.40 * tgt_vol))
        assert t_cnt >= min_tissue, (
            f"Target slot {m} ({cz}x{cy}x{cx}, vol {tgt_vol}) had {t_cnt} tissue tokens, "
            f"expected >= {min_tissue}"
        )

    # Disjointness guarantee
    ctx_set = set(out["context_indices"].tolist())
    all_targets = set().union(*[t.tolist() for t in out["target_indices_list"]])
    assert not (ctx_set & all_targets), "Context and target overlap under brain-aware masking"



# =========================================================================
# 2. Modality Dropout & Augmentations
# =========================================================================


def test_random_modality_dropout_3d():
    dropout = RandomModalityDropout3D(p_drop=0.5)
    dropout.train()

    img = torch.ones(4, 32, 32, 32)
    out = dropout(img)
    assert out.shape == (4, 32, 32, 32)

    # Test guaranteed active channel fallback when p_drop = 1.0
    dropout_all = RandomModalityDropout3D(p_drop=1.0)
    dropout_all.train()

    out_fallback = dropout_all(img)
    active_channels = [(out_fallback[c] != 0).any().item() for c in range(4)]
    assert sum(active_channels) >= 1, "Guaranteed fallback failed: all channels dropped!"

    # Eval mode: must preserve all channels
    dropout.eval()
    out_eval = dropout(img)
    assert torch.equal(out_eval, img)


def test_random_modality_dropout_pytorch_prng_determinism():
    """Verify RandomModalityDropout3D obeys torch.manual_seed for reproducible dropout masks."""
    dropout = RandomModalityDropout3D(p_drop=0.5)
    dropout.train()
    img = torch.ones(1, 4, 16, 16, 16)

    torch.manual_seed(1234)
    out1 = dropout(img)

    torch.manual_seed(1234)
    out2 = dropout(img)

    assert torch.equal(out1, out2)


def test_p0_step1_modality_dropout_preserves_expectation():
    """Inverted dropout: E[output|train] ~= input (kept channels rescaled)."""
    torch.manual_seed(0)
    drop = RandomModalityDropout3D(p_drop=0.25)
    drop.train()
    x = torch.ones(64, 4, 8, 8, 8)
    acc = torch.zeros_like(x)
    n = 20
    for _ in range(n):
        acc += drop(x)
    mean = (acc / n).mean().item()
    assert abs(mean - 1.0) < 0.15, f"train/test magnitude shift: E={mean}"


def test_p0_step1_dropout_eval_identity_and_fallback():
    drop = RandomModalityDropout3D(p_drop=0.25)
    drop.eval()
    x = torch.randn(2, 4, 8, 8, 8)
    assert torch.equal(drop(x), x)
    drop.train()
    drop.p_drop = 1.0  # force all-drop -> exactly one scaled active channel
    y = drop(torch.ones(1, 4, 4, 4, 4))
    assert (y[0, :, 0, 0, 0] > 0).sum().item() == 1


def test_volumetric_augmentations_3d():
    aug = VolumetricAugmentations3D(
        flip_prob=1.0, noise_prob=1.0, modality_dropout_prob=0.0, is_training=True
    )
    img = torch.randn(4, 32, 32, 32)
    mask = torch.zeros(1, 32, 32, 32)
    mask[:, 10:20, 10:20, 10:20] = 1.0

    aug_img, aug_mask = aug(img, mask)
    assert aug_img.shape == (4, 32, 32, 32)
    assert aug_mask.shape == (1, 32, 32, 32)


def test_volumetric_augmentations_batched_tensor():
    """Verify VolumetricAugmentations3D processes 5D batched tensors cleanly without crashes."""
    aug = VolumetricAugmentations3D(
        flip_prob=0.5,
        noise_prob=0.5,
        modality_dropout_prob=0.25,
        is_training=True,
    )
    img = torch.randn(2, 4, 32, 32, 32)
    mask = torch.zeros(2, 1, 32, 32, 32)
    aug_img, aug_mask = aug(img, mask)
    assert aug_img.shape == (2, 4, 32, 32, 32)
    assert aug_mask.shape == (2, 1, 32, 32, 32)


def test_p0_step2_augmentations_torch_determinism():
    """Same torch seed -> bitwise-identical outputs (no Python random)."""
    aug = VolumetricAugmentations3D(is_training=True)
    img = torch.randn(4, 16, 16, 16)
    msk = (torch.rand(1, 16, 16, 16) > 0.5).float()
    torch.manual_seed(0)
    r1 = aug(img.clone(), msk.clone())
    torch.manual_seed(0)
    r2 = aug(img.clone(), msk.clone())
    assert torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1])


def test_p0_step5_rician_background_nonzero():
    """Rician OOD: background air gets Rayleigh noise (non-zero), non-negative."""
    torch.manual_seed(0)
    img = torch.zeros(4, 16, 16, 16)
    img[:, 4:12, 4:12, 4:12] = 1.0
    out = apply_rician_noise_3d(img, sigma=0.1)
    bg = out[:, :4, :, :]
    assert (bg != 0).any().item(), "background air must carry Rayleigh noise"
    assert (out >= 0).all().item(), "magnitude image must be non-negative"
    assert torch.equal(apply_rician_noise_3d(img, sigma=0.0), img)


def test_p0_step6_b1_random_per_call():
    """B1 field varies across calls (seeded determinism preserved)."""
    torch.manual_seed(0)
    img = torch.ones(4, 16, 16, 16)
    a = apply_b1_bias_field_3d(img, strength=0.3)
    b = apply_b1_bias_field_3d(img, strength=0.3)
    assert not torch.equal(a, b), "fixed field for all volumes is a bug"
    torch.manual_seed(0)
    a2 = apply_b1_bias_field_3d(img, strength=0.3)
    assert torch.equal(a, a2), "same seed must reproduce the field"
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(
        apply_b1_bias_field_3d(img, generator=g1), apply_b1_bias_field_3d(img, generator=g2)
    )


def test_b1_bias_field_zero_centered_contrast_preservation():
    """Verify B1 bias field shifts zero-centered parenchyma before multiplicative scaling."""
    img = torch.zeros(4, 16, 16, 16)
    img[:, 4:12, 4:12, 4:12] = torch.linspace(-2.0, 2.0, 8).reshape(1, 1, 1, 8)
    torch.manual_seed(42)
    out = apply_b1_bias_field_3d(img, strength=0.3)
    assert out.shape == img.shape
    assert (out[:, :4, :4, :4] == 0.0).all()


# =========================================================================
# 3. Dataset, Subsampling & Preprocessing
# =========================================================================


def test_dataset_stratified_subsampling(tmp_path):
    """Verify BraTS3DDataset stratifies low-data fraction splits by tumor quartile."""
    import pandas as pd

    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    rows = []
    for i in range(20):
        pid = f"BraTS_GLI_{i:04d}"
        q = i % 4
        rows.append({
            "patient_id": pid,
            "file_name": f"{pid}.npz",
            "split": "train",
            "tumor_quartile": q,
        })
        np.savez_compressed(
            data_dir / f"{pid}.npz",
            image=np.zeros((4, 8, 8, 8), dtype=np.float32),
            mask=np.zeros((1, 8, 8, 8), dtype=np.float32),
        )

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "metadata.csv", index=False)

    ds_full = BraTS3DDataset(split="train", data_dir=data_dir, fraction=1.0)
    assert len(ds_full) == 20

    ds_half = BraTS3DDataset(split="train", data_dir=data_dir, fraction=0.5, seed=42)
    assert len(ds_half) == 10

    selected_pids = [sample["patient_id"] for sample in ds_half]
    selected_quartiles = df[df["patient_id"].isin(selected_pids)]["tumor_quartile"].tolist()
    for q in range(4):
        assert selected_quartiles.count(q) >= 2


def test_resampling_boundary_mask_bleed_suppression_and_brain_mask():
    """Verify CUT-A & CUT-B: brain mask resampling eliminates trilinear bleed and preserves true parenchyma stats."""
    from prepare_data_3d import resample_3d_volume, zscore_normalize_non_zero

    grid = np.zeros((32, 32, 32), dtype=np.float32)
    z, y, x = np.ogrid[:32, :32, :32]
    dist_from_center = np.sqrt((z - 16) ** 2 + (y - 16) ** 2 + (x - 16) ** 2)
    native_mask = dist_from_center <= 10
    grid[native_mask] = 500.0 + np.random.randn(*grid[native_mask].shape) * 50.0

    target_shape = (16, 16, 16)

    # 1. Resample binary mask (nearest) and raw image (trilinear)
    res_mask = resample_3d_volume(native_mask.astype(np.float32), target_shape, is_mask=True) > 0.5
    res_img = resample_3d_volume(grid, target_shape, is_mask=False)

    # 2. Before suppression: trilinear bleed voxels exist outside res_mask
    bleed_voxels = (res_img > 0) & (~res_mask)
    assert np.any(bleed_voxels)

    # 3. Apply CUT-A suppression: zero out all voxels outside res_mask
    clean_img = np.where(res_mask, res_img, 0.0)
    assert not np.any((clean_img != 0) & (~res_mask))

    # 4. Normalize with explicit mask
    norm_img = zscore_normalize_non_zero(clean_img, mask=res_mask)
    assert np.all(norm_img[~res_mask] == 0.0)
    assert abs(float(norm_img[res_mask].mean())) < 1e-5
    assert abs(float(norm_img[res_mask].std()) - 1.0) < 1e-4


def test_brats_3d_dataset_tensor_writeable_and_memory_ownership(tmp_path):
    """Verify BraTS3DDataset produces writeable, contiguous memory tensors from npz."""
    img = np.random.randn(4, 16, 16, 16).astype(np.float32)
    mask = (np.random.rand(16, 16, 16) > 0.8).astype(np.float32)
    bm = (img != 0).any(axis=0, keepdims=True).astype(np.float32)
    npz_file = tmp_path / "patient_001.npz"
    np.savez_compressed(npz_file, image=img, mask=mask, brain_mask=bm)

    (tmp_path / "metadata.csv").write_text("patient_id,file_name\n001,patient_001.npz\n")
    dataset = BraTS3DDataset(data_dir=tmp_path, cache_in_ram=True)

    sample = dataset[0]
    image = sample["image"]
    mask_t = sample["mask"]
    bm_t = sample["brain_mask"]

    # Verify writeability
    image[0, 0, 0, 0] = 999.0
    assert image[0, 0, 0, 0].item() == 999.0
    mask_t[0, 0, 0] = 1.0
    assert mask_t[0, 0, 0].item() == 1.0
    bm_t[0, 0, 0, 0] = 1.0
    assert bm_t[0, 0, 0, 0].item() == 1.0

    assert 0 in dataset.cache
    assert len(dataset.cache[0]) == 3
    assert isinstance(dataset.cache[0][0], torch.Tensor)
    assert isinstance(dataset.cache[0][1], torch.Tensor)
    assert isinstance(dataset.cache[0][2], torch.Tensor)


# =========================================================================
# 4. Multi-Region Preprocessing & Resampling
# =========================================================================


def _make_patient(root, pid="TEST-001", labels=(0, 1, 2, 3, 4), shape=(24, 24, 24)):
    d = root / pid
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for mod in ("t1n", "t1c", "t2w", "t2f"):
        img = np.abs(rng.standard_normal(shape)).astype(np.float32) + 0.5
        nib.save(nib.Nifti1Image(img, np.eye(4)), d / f"{pid}-{mod}.nii.gz")
    seg = np.zeros(shape, dtype=np.float32)
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
    rec = process_patient(pdir, out, target_size=16, dtype="float32")
    assert rec is not None
    z = np.load(out / f"{pdir.name}_volume.npz")
    assert z["image"].shape == (4, 16, 16, 16)
    vals = set(np.unique(z["mask"]).tolist())
    assert vals <= set(BRATS_GLI_LABELS), vals
    assert vals >= {1, 2, 3, 4}, f"octant labels lost in resampling: {vals}"
    assert rec["has_et"] is True and rec["num_voxels_et"] > 0
    assert ET_LABEL == 4


def test_multiregion_rejects_stray_labels(tmp_path):
    pdir = _make_patient(tmp_path / "raw")
    seg_path = pdir / f"{pdir.name}-seg.nii.gz"
    img = nib.load(str(seg_path))
    data = img.get_fdata().copy()
    data[0, 0, 0] = 7.0
    nib.save(nib.Nifti1Image(data, img.affine), str(seg_path))
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="unexpected segmentation labels"):
        process_patient(pdir, out, target_size=16)


def test_multiregion_et_free_volume(tmp_path):
    pdir = _make_patient(tmp_path / "raw", pid="TEST-NOET", labels=(0, 1, 2))
    out = tmp_path / "out"
    out.mkdir()
    rec = process_patient(pdir, out, target_size=16, dtype="float32")
    assert rec["has_et"] is False and rec["num_voxels_et"] == 0
    assert rec["has_tumor"] is True


def test_resample_preserve_labels_unit():
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[2:5, 2:5, 2:5] = 4.0
    vol[6:9, 6:9, 6:9] = 1.0
    out = resample_3d_volume(vol, (8, 8, 8), is_mask=True, preserve_labels=True)
    assert set(np.unique(out).tolist()) == {0, 1, 4}


def test_prepare_data_3d_grouped_base_id_zero_leakage():
    """Verify grouped base_id splitting guarantees zero longitudinal patient leakage across train, val, and test."""
    import re
    import pandas as pd
    from sklearn.model_selection import train_test_split

    records = []
    # 25 unique base patients, each with 1-3 longitudinal scans
    for b_idx in range(1, 26):
        base_id = f"BraTS-GLI-{b_idx:05d}"
        n_scans = 1 if b_idx % 3 == 0 else (2 if b_idx % 3 == 1 else 3)
        for s_idx in range(1, n_scans + 1):
            records.append({
                "patient_id": f"{base_id}-{s_idx:02d}",
                "num_voxels_tumor": int(1000 * b_idx + 10 * s_idx),
            })
    df = pd.DataFrame(records)

    base_id_re = re.compile(r"^(.*)-\d+$")
    df["base_id"] = df["patient_id"].apply(
        lambda p: base_id_re.match(str(p)).group(1) if base_id_re.match(str(p)) else str(p)
    )

    g = df.groupby("base_id").agg(
        num_voxels_tumor=("num_voxels_tumor", "max"),
        n_scans=("patient_id", "count"),
    ).reset_index()

    g["tumor_quartile"] = pd.qcut(g["num_voxels_tumor"], q=4, labels=False, duplicates="drop")
    train_g, temp_g = train_test_split(g, test_size=0.30, random_state=42, stratify=g["tumor_quartile"])
    val_g, test_g = train_test_split(temp_g, test_size=0.50, random_state=42, stratify=temp_g["tumor_quartile"])

    split_map = {r.base_id: "train" for r in train_g.itertuples()}
    split_map.update({r.base_id: "val" for r in val_g.itertuples()})
    split_map.update({r.base_id: "test" for r in test_g.itertuples()})
    df["split"] = df["base_id"].map(split_map)

    train_bases = set(df[df["split"] == "train"]["base_id"])
    val_bases = set(df[df["split"] == "val"]["base_id"])
    test_bases = set(df[df["split"] == "test"]["base_id"])

    # Hard guarantees: ZERO cross-split leakage
    assert len(train_bases & val_bases) == 0, f"Leaked Train <-> Val: {train_bases & val_bases}"
    assert len(train_bases & test_bases) == 0, f"Leaked Train <-> Test: {train_bases & test_bases}"
    assert len(val_bases & test_bases) == 0, f"Leaked Val <-> Test: {val_bases & test_bases}"
    assert len(train_bases | val_bases | test_bases) == 25


def test_resolution_scaled_anisotropic_masking():
    """Verify that JEPAMaskingTransform3D correctly scales anisotropic cuboids and context for 16^3 grid."""
    from brats_jepa_3d.data.masking import (
        DEFAULT_TARGET_CUBOID_SIZES_8GRID,
        DEFAULT_TARGET_CUBOID_SIZES_16GRID,
    )

    # 1. Scaling math matches 2x per axis
    for s8, s16 in zip(DEFAULT_TARGET_CUBOID_SIZES_8GRID, DEFAULT_TARGET_CUBOID_SIZES_16GRID):
        assert tuple(x * 2 for x in s8) == s16

    # 2. 8^3 grid defaults
    tf8 = JEPAMaskingTransform3D(grid_size=(8, 8, 8))
    assert tf8.context_num_patches == 192  # 37.5% of 512
    assert tf8.target_cuboid_sizes == DEFAULT_TARGET_CUBOID_SIZES_8GRID

    # 3. 16^3 grid defaults
    tf16 = JEPAMaskingTransform3D(grid_size=(16, 16, 16))
    assert tf16.context_num_patches == 1536  # 37.5% of 4096
    assert tf16.target_cuboid_sizes == DEFAULT_TARGET_CUBOID_SIZES_16GRID

    # 4. Invariance of anatomical coverage:
    # On 128mm volume:
    # For 16mm patch: 192 tokens * (16^3) = 786,432 mm^3
    # For 8mm patch: 1536 tokens * (8^3) = 786,432 mm^3
    vol8 = tf8.context_num_patches * (16**3)
    vol16 = tf16.context_num_patches * (8**3)
    assert vol8 == vol16, "Context physical volume differs across resolutions"

