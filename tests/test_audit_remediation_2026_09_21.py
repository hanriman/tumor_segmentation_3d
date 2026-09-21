"""Failing-first regression harness for docs/remediation_plan_audit_2026-09-21.md.

Each test asserts the DESIRED end-state. Tests covering unfixed steps FAIL until
their remediation step lands (TDD). Run:
    pytest tests/test_audit_remediation_2026_09_21.py -q
"""
import torch

from brats_jepa_3d.data import JEPAMaskingTransform3D
from brats_jepa_3d.data.transforms import (
    RandomModalityDropout3D,
    VolumetricAugmentations3D,
    apply_b1_bias_field_3d,
    apply_rician_noise_3d,
)
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d


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


def test_p0_step2_augmentations_torch_determinism():
    """Same torch seed -> bitwise-identical outputs (no Python `random`)."""
    aug = VolumetricAugmentations3D(is_training=True)
    img = torch.randn(4, 16, 16, 16)
    msk = (torch.rand(1, 16, 16, 16) > 0.5).float()
    torch.manual_seed(0)
    r1 = aug(img.clone(), msk.clone())
    torch.manual_seed(0)
    r2 = aug(img.clone(), msk.clone())
    assert torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1])


def test_p0_step3_sigreg_tissue_parity():
    """SigRegJEPA3D filters projected tokens by tissue mask like VisReg."""
    import inspect

    from brats_jepa_3d.models import SigRegJEPA3D, VisRegJEPA3D
    assert "context_tissue_mask" in inspect.signature(SigRegJEPA3D.forward).parameters
    assert "context_tissue_mask" in inspect.signature(VisRegJEPA3D.forward).parameters


def test_p0_step4_per_sample_tissue_fallback():
    """B=2 with tissue counts [100, 5] keeps 100+192 rows (not batch fallback)."""
    import sys

    sys.path.insert(0, "src")
    from brats_jepa_3d.models._tissue_filter import filter_tissue_tokens

    proj = torch.randn(2, 192, 128)
    mask = torch.zeros(2, 192, dtype=torch.bool)
    mask[0, :100] = True
    mask[1, :5] = True
    out = filter_tissue_tokens(proj, mask, min_tokens=32)
    assert out.shape == (100 + 192, 128), out.shape


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


def test_p1_step9_dice_background_contract():
    """Binary task invariant to include_background; 4-class differs; CE covers all voxels."""
    from brats_jepa_3d.losses import CombinedDiceBCELoss3D, VolumetricDiceLoss

    torch.manual_seed(0)
    logits_b = torch.randn(2, 1, 8, 8, 8)
    target_b = (torch.rand(2, 1, 8, 8, 8) > 0.9).float()
    l_true = VolumetricDiceLoss(include_background=True)(logits_b, target_b).item()
    l_false = VolumetricDiceLoss(include_background=False)(logits_b, target_b).item()
    assert l_true == l_false, "binary Dice must ignore include_background"

    logits_m = torch.randn(2, 4, 8, 8, 8)
    target_m = torch.randint(0, 4, (2, 1, 8, 8, 8)).float()
    m_true = VolumetricDiceLoss(include_background=True)(logits_m, target_m).item()
    m_false = VolumetricDiceLoss(include_background=False)(logits_m, target_m).item()
    assert m_true != m_false, "multi-class flag must take effect"
    # CE asymmetry: false positives on background penalized even when Dice ignores it.
    res = CombinedDiceBCELoss3D(include_background=False)(logits_m, target_m)
    assert torch.isfinite(res["loss"]).item() and torch.isfinite(res["bce_loss"]).item()


def test_p1_step7_zero_overlap_hard_guarantee():
    """500 masks with max_target_overlap=0.0 -> pairwise target intersections == 0."""
    tf = JEPAMaskingTransform3D(
        grid_size=(8, 8, 8), num_target_cuboids=4,
        target_cuboid_size=(3, 3, 3), max_target_overlap=0.0,
        context_num_patches=192, connectivity=26,
    )
    for seed in range(500):
        g = torch.Generator()
        g.manual_seed(seed)
        out = tf(generator=g)
        blocks = [set(t.tolist()) for t in out["target_indices_list"]]
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                assert not (blocks[i] & blocks[j]), f"overlap at seed {seed}"


def test_p1_step8_hd95_tumor_only_nan_semantics():
    """Tumor-only aggregates are nan (not 1.0/0.0) when HD95 is off or cohort tumor-free."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, 16)
    masks = torch.zeros(2, 1, 16, 16, 16)
    res = compute_volumetric_metrics_3d(logits, masks, compute_hd95=False)
    assert res["hd95_tumor_only"] != res["hd95_tumor_only"], "must be nan when HD95 off"
    assert res["dice_tumor_only"] != res["dice_tumor_only"], "must be nan when cohort tumor-free"
    assert res["iou_tumor_only"] != res["iou_tumor_only"], "must be nan when cohort tumor-free"
    res2 = compute_volumetric_metrics_3d(logits, masks, compute_hd95=True)
    assert res2["hd95_tumor_only"] != res2["hd95_tumor_only"], "must be nan on tumor-free cohort"
    assert res2["dice_tumor_only"] != res2["dice_tumor_only"], "must be nan on tumor-free cohort"
    assert res2["iou_tumor_only"] != res2["iou_tumor_only"], "must be nan on tumor-free cohort"


def test_p2_step11_no_mode_side_effect():
    """forward() preserves caller mode; grads flow only through the context path."""
    from brats_jepa_3d.models import VisRegJEPA3D

    torch.manual_seed(0)
    model = VisRegJEPA3D()
    model.train()
    B, N_ctx, N_tgt = 1, 8, 4
    images = torch.randn(B, 4, 128, 128, 128)
    ctx = torch.randint(0, 512, (B, N_ctx))
    tgt = [torch.randint(0, 512, (B, N_tgt))]
    out = model(images, ctx, tgt)
    assert model.training and model.context_encoder.training
    out["predictions"][0].mean().backward()
    model.eval()
    with torch.no_grad():
        model(images, ctx, tgt)
    assert not model.training and not model.context_encoder.training


def test_p2_step12_pos_embed_frozen():
    """Sinusoidal topology fixed: requires_grad False, keys unchanged for ckpt compat."""
    from brats_jepa_3d.models import VisRegJEPA3D

    model = VisRegJEPA3D()
    assert model.context_encoder.pos_embed.requires_grad is False
    assert model.predictor.pos_embed.requires_grad is False
    assert "context_encoder.pos_embed" in model.state_dict()
    assert "predictor.pos_embed" in model.state_dict()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not any("pos_embed" in n for n in trainable)


def test_p2_step14_unet_hooks_bounded():
    """One hook set for life: consecutive forwards identical, eval single tensor."""
    from brats_jepa_3d.models import BraTS3DUNet

    torch.manual_seed(0)
    m = BraTS3DUNet(channels=(8, 16, 32, 64, 128), num_res_units=1, dropout=0.0)
    n_handles_0 = len(m._ds_hook_handles)
    assert n_handles_0 > 0
    x = torch.randn(1, 4, 32, 32, 32)
    m.train()
    with torch.no_grad():
        r1 = [o.clone() for o in m(x)]
        r2 = [o.clone() for o in m(x)]
    assert len(m._ds_hook_handles) == n_handles_0, "hook leak across forwards"
    for a, b in zip(r1, r2):
        assert torch.equal(a, b)
    m.eval()
    with torch.no_grad():
        y = m(x)
    assert isinstance(y, torch.Tensor)
    m.close_hooks()
    assert len(m._ds_hook_handles) == 0


def test_residual_tissue_filter_rejects_non3d():
    """filter_tissue_tokens fails loud on pre-flattened 2D input (mask would misalign)."""
    import pytest

    from brats_jepa_3d.models._tissue_filter import filter_tissue_tokens

    with pytest.raises(ValueError):
        filter_tissue_tokens(torch.randn(100, 128), torch.ones(1, 8, dtype=torch.bool))


def test_f1_dropout_disabled_restores_probs():
    """dropout_disabled preserves mode and restores Dropout/Attention probs."""
    from brats_jepa_3d.models import VisRegJEPA3D
    from brats_jepa_3d.models.vision_transformer_3d import dropout_disabled

    torch.manual_seed(0)
    model = VisRegJEPA3D()
    for block in model.context_encoder.blocks:
        for m in block.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.5
            if isinstance(m, torch.nn.MultiheadAttention):
                m.dropout = 0.2
    model.train()
    B, N_ctx, N_tgt = 1, 8, 4
    images = torch.randn(B, 4, 128, 128, 128)
    ctx = torch.randint(0, 512, (B, N_ctx))
    tgt = [torch.randint(0, 512, (B, N_tgt))]
    with dropout_disabled(model.context_encoder):
        for m in model.context_encoder.modules():
            if isinstance(m, torch.nn.Dropout):
                assert m.p == 0.0
    out = model(images, ctx, tgt)
    assert model.training and model.context_encoder.training
    for block in model.context_encoder.blocks:
        for m in block.modules():
            if isinstance(m, torch.nn.Dropout):
                assert m.p == 0.5
            if isinstance(m, torch.nn.MultiheadAttention):
                assert float(m.dropout) == 0.2
    out["predictions"][0].mean().backward()


def test_f3_ood_preserves_negatives():
    """Sign-preserving OOD: tissue negatives survive Rician/B1; air models differ."""
    neg = -1.5 * torch.ones(2, 4, 8, 8, 8)
    neg[:, :, 0:2] = 0.0
    bm = torch.ones_like(neg)
    bm[:, :, 0:2] = 0.0
    torch.manual_seed(0)
    r = apply_rician_noise_3d(neg, sigma=0.05, brain_mask=bm)
    assert (r[bm > 0] < 0).any().item(), "Rician rectified all negatives"
    assert (r[bm == 0] != 0).any().item(), "air must carry Rayleigh noise"
    b = apply_b1_bias_field_3d(neg, strength=0.3, brain_mask=bm)
    assert (b[bm > 0] < 0).any().item(), "B1 must preserve negative contrast"
    assert (b[bm == 0] == 0).all().item(), "B1 air stays zero"


def test_f10_tversky_include_background_flag():
    """Raw Tversky honors include_background for multi-class; binary invariant."""
    from brats_jepa_3d.losses import VolumetricTverskyLoss

    torch.manual_seed(0)
    logits_m = torch.randn(2, 4, 8, 8, 8)
    target_m = torch.randint(0, 4, (2, 1, 8, 8, 8)).float()
    t_true = VolumetricTverskyLoss(include_background=True)(logits_m, target_m).item()
    t_false = VolumetricTverskyLoss(include_background=False)(logits_m, target_m).item()
    assert t_true != t_false
    logits_b = torch.randn(2, 1, 8, 8, 8)
    target_b = (torch.rand(2, 1, 8, 8, 8) > 0.9).float()
    b_true = VolumetricTverskyLoss(include_background=True)(logits_b, target_b).item()
    b_false = VolumetricTverskyLoss(include_background=False)(logits_b, target_b).item()
    assert b_true == b_false


def test_f9_aggregation_reports_dice_std(tmp_path):
    """Master aggregation keeps best row but reports n_runs + dice_std."""
    import pandas as pd

    from brats_jepa_3d.utils.aggregation import aggregate_master_benchmarks

    for i, dice in enumerate(["70.0 ± 1.0", "80.0 ± 1.0"]):
        d = tmp_path / f"exp{i}"
        (d / "metrics").mkdir(parents=True)
        pd.DataFrame([{"Model Architecture": "3D VisReg JEPA (FPN)", "Dice (%)": dice}]).to_csv(
            d / "metrics" / "master_3d_benchmark.csv", index=False
        )
    df = aggregate_master_benchmarks([tmp_path / "exp0", tmp_path / "exp1"])
    assert len(df) == 1
    assert df.iloc[0]["n_runs"] == 2
    assert df.iloc[0]["dice_std"] >= 0.0
    assert "80.0" in str(df.iloc[0]["Dice (%)"])


def test_f5_hd95_per_volume_determinism():
    """Same pair reproducible; different volumes valid; cap keeps it fast."""
    import numpy as np
    import time

    from brats_jepa_3d.metrics import compute_hd95_3d

    rng = np.random.default_rng(0)
    a = (rng.random((32, 32, 32)) > 0.7).astype(np.float32)
    b = (rng.random((32, 32, 32)) > 0.7).astype(np.float32)
    h1 = compute_hd95_3d(a, b)
    h2 = compute_hd95_3d(a, b)
    assert h1 == h2
    c = (rng.random((48, 48, 48)) > 0.7).astype(np.float32)
    d = (rng.random((48, 48, 48)) > 0.7).astype(np.float32)
    t0 = time.perf_counter()
    h3 = compute_hd95_3d(c, d)
    assert time.perf_counter() - t0 < 10.0
    assert np.isfinite(h3)
