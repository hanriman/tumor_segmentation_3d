"""Loss module tests: JEPA, VisReg, SigReg, Dice, BCE, Tversky, and Deep Supervision."""

import pytest
import torch
import torch.nn.functional as F

from brats_jepa_3d.losses import (
    CombinedDiceBCELoss3D,
    CombinedTverskyBCEWithLogitsLoss3D,
    DeepSupervisionLoss3D,
    JEPAPredictionLoss,
    SigRegLoss,
    VisRegLoss,
    VolumetricDiceLoss,
    VolumetricTverskyLoss,
    build_segmentation_criterion,
    resolve_seg_loss_type,
)


# =========================================================================
# 1. Self-Supervised JEPA Losses (Prediction Loss, VisReg, SigReg)
# =========================================================================


def test_jepa_prediction_loss():
    criterion = JEPAPredictionLoss(loss_type="smooth_l1")
    preds = [torch.randn(2, 27, 384, requires_grad=True)]
    targets = [torch.randn(2, 27, 384)]

    loss = criterion(preds, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    assert preds[0].grad is not None


def test_jepa_prediction_loss_empty_predictions_raises():
    """Verify JEPAPredictionLoss raises ValueError when predictions or targets lists are empty."""
    criterion = JEPAPredictionLoss()
    with pytest.raises(ValueError, match="must be non-empty"):
        criterion([], [])


def test_sigreg_loss():
    criterion = SigRegLoss(sigreg_weight=1.0, num_projections=64)
    preds = [torch.randn(2, 27, 384, requires_grad=True)]
    targets = [torch.randn(2, 27, 384)]
    proj_tokens = torch.randn(2, 192, 128, requires_grad=True)

    res = criterion(preds, targets, projected_tokens=proj_tokens)
    assert "loss" in res
    assert "sigreg_loss" in res
    assert res["loss"].item() > 0.0

    res["loss"].backward()
    assert proj_tokens.grad is not None


def test_sigreg_loss_custom_parameters():
    """Verify SigRegLoss initializes with custom parameters from YAML configs."""
    criterion = SigRegLoss(
        loss_type="smooth_l1",
        sigreg_weight=0.7,
        num_projections=128,
        t_max=2.5,
        n_knots=15,
        normalize_measure=False,
    )
    assert criterion.sigreg_weight == 0.7
    assert criterion.num_projections == 128
    assert criterion.ep_test.t_max == 2.5
    assert criterion.ep_test.n_knots == 15
    assert criterion.ep_test.normalize_measure is False

    preds = [torch.randn(2, 16, 64)]
    targets = [torch.randn(2, 16, 64)]
    reg_tokens = torch.randn(2, 16, 64)
    out = criterion(preds, targets, tokens=reg_tokens)
    assert "loss" in out
    assert out["loss"].item() > 0


def test_sigreg_loss_gaussian_input():
    """SigReg EP test should produce near-zero loss for standard Gaussian input."""
    torch.manual_seed(123)
    criterion = SigRegLoss(sigreg_weight=1.0, num_projections=128)
    preds = [torch.randn(2, 27, 384)]
    targets = [torch.randn(2, 27, 384)]
    proj_tokens = torch.randn(2, 192, 128)

    res = criterion(preds, targets, projected_tokens=proj_tokens)
    assert res["sigreg_loss"].item() < res["jepa_loss"].item() * 5.0


def test_sigreg_anti_collapse_penalty():
    """SigReg loss should strongly penalize collapsed representation compared to isotropic Gaussian."""
    torch.manual_seed(42)
    criterion = SigRegLoss(sigreg_weight=1.0, num_projections=64)
    preds = [torch.randn(2, 27, 384)]
    targets = [torch.randn(2, 27, 384)]

    proj_normal = torch.randn(2, 192, 128)
    res_normal = criterion(preds, targets, projected_tokens=proj_normal)

    proj_collapsed = torch.randn(2, 192, 128) * 0.01
    res_collapsed = criterion(preds, targets, projected_tokens=proj_collapsed)

    assert res_collapsed["sigreg_loss"].item() > 10.0 * res_normal["sigreg_loss"].item()


def test_visreg_loss():
    criterion = VisRegLoss(center_weight=1.0, scale_weight=1.0, swd_weight=1.0, num_projections=64)
    preds = [torch.randn(2, 27, 384, requires_grad=True)]
    targets = [torch.randn(2, 27, 384)]
    proj_tokens = torch.randn(2, 192, 128, requires_grad=True)

    res = criterion(preds, targets, projected_tokens=proj_tokens)
    assert "loss" in res
    assert "center_loss" in res
    assert "scale_loss" in res
    assert "shape_loss" in res

    res["loss"].backward()
    assert proj_tokens.grad is not None


def test_visreg_loss_custom_parameters():
    """Verify VisRegLoss initializes with custom parameters from YAML configs."""
    criterion = VisRegLoss(
        loss_type="smooth_l1",
        center_weight=2.0,
        scale_weight=0.5,
        swd_weight=1.5,
        num_projections=128,
        target_std=1.2,
        swd_metric="l1",
        scale_loss_type="hinge",
    )
    assert criterion.center_weight == 2.0
    assert criterion.scale_weight == 0.5
    assert criterion.swd_weight == 1.5
    assert criterion.num_projections == 128
    assert criterion.target_std == 1.2
    assert criterion.swd_metric == "l1"
    assert criterion.scale_loss_type == "hinge"

    preds = [torch.randn(2, 16, 64)]
    targets = [torch.randn(2, 16, 64)]
    reg_tokens = torch.randn(2, 16, 64)
    out = criterion(preds, targets, tokens=reg_tokens)
    assert "loss" in out
    assert out["loss"].item() > 0


def test_visreg_swd_standard_normal():
    """Verify SWD between standard normal samples and target quantiles is small (<0.10)."""
    criterion = VisRegLoss(num_projections=128)
    torch.manual_seed(42)
    z = torch.randn(2048, 128)
    swd = criterion._sliced_wasserstein_distance(z)
    assert swd.item() < 0.10, f"SWD of standard normal is too large: {swd.item()}"


def test_visreg_loss_reg_scale_annealing():
    """Verify VisRegLoss scales regularizers via reg_scale."""
    criterion = VisRegLoss(num_projections=64)
    criterion.train()

    preds = [torch.randn(2, 27, 64, requires_grad=True)]
    targets = [torch.randn(2, 27, 64)]
    proj1 = torch.randn(2, 192, 32, requires_grad=True)

    # Step 1: Full regularizer scale (reg_scale=1.0)
    out1 = criterion(preds, targets, projected_tokens=proj1, reg_scale=1.0)
    assert "reg_scale" in out1
    assert out1["reg_scale"].item() == 1.0
    out1["loss"].backward()
    assert proj1.grad is not None

    # Step 2: Annealed regularizer scale (reg_scale=0.10)
    preds2 = [torch.randn(2, 27, 64, requires_grad=True)]
    proj2 = torch.randn(2, 192, 32, requires_grad=True)
    out2 = criterion(preds2, targets, projected_tokens=proj2, reg_scale=0.10)
    assert out2["reg_scale"].item() == pytest.approx(0.10, abs=1e-5)
    expected_loss = out2["jepa_loss"] + 0.10 * (
        criterion.center_weight * out2["center_loss"]
        + criterion.scale_weight * out2["scale_loss"]
        + criterion.swd_weight * out2["swd_loss"]
    )
    assert torch.allclose(out2["loss"], expected_loss, atol=1e-5)


def test_sigreg_loss_reg_scale_annealing():
    """Verify SigRegLoss scales regularizer via reg_scale."""
    criterion = SigRegLoss(num_projections=64)
    criterion.train()

    preds = [torch.randn(2, 27, 64, requires_grad=True)]
    targets = [torch.randn(2, 27, 64)]
    proj1 = torch.randn(2, 192, 32, requires_grad=True)

    out1 = criterion(preds, targets, projected_tokens=proj1, reg_scale=1.0)
    assert "reg_scale" in out1
    assert out1["reg_scale"].item() == 1.0
    out1["loss"].backward()
    assert proj1.grad is not None

    # Annealed scale
    preds2 = [torch.randn(2, 27, 64, requires_grad=True)]
    proj2 = torch.randn(2, 192, 32, requires_grad=True)
    out2 = criterion(preds2, targets, projected_tokens=proj2, reg_scale=0.10)
    assert out2["reg_scale"].item() == pytest.approx(0.10, abs=1e-5)
    expected_loss = out2["jepa_loss"] + 0.10 * criterion.sigreg_weight * out2["sigreg_loss"]
    assert torch.allclose(out2["loss"], expected_loss, atol=1e-5)


# =========================================================================
# 2. Supervised Segmentation Losses (Dice, BCE)
# =========================================================================


def test_dice_bce_loss_3d():
    criterion = CombinedDiceBCELoss3D()
    logits = torch.randn(2, 1, 32, 32, 32, requires_grad=True)
    targets = (torch.rand(2, 1, 32, 32, 32) > 0.8).float()

    res = criterion(logits, targets)
    assert "loss" in res
    assert "dice_loss" in res
    assert "bce_loss" in res

    res["loss"].backward()
    assert logits.grad is not None


def test_volumetric_dice_include_background():
    """Verify include_background=False excludes channel 0 in multi-class Dice loss."""
    dice_inc = VolumetricDiceLoss(num_classes=4, include_background=True)
    dice_exc = VolumetricDiceLoss(num_classes=4, include_background=False)

    logits = torch.zeros(1, 4, 8, 8, 8)
    logits[:, 0] = 10.0
    logits[:, 1:] = -10.0

    targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    targets[:, 2:6, 2:6, 2:6] = 1

    loss_inc = dice_inc(logits, targets).item()
    loss_exc = dice_exc(logits, targets).item()
    assert loss_exc > loss_inc


def test_combined_dice_bce_loss_include_background():
    """Verify CombinedDiceBCELoss3D forwards include_background correctly."""
    loss_fn = CombinedDiceBCELoss3D(num_classes=4, include_background=False)
    assert loss_fn.dice_loss.include_background is False

    logits = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    res = loss_fn(logits, targets)

    assert "loss" in res
    assert res["loss"].item() > 0
    res["loss"].backward()
    assert logits.grad is not None


def test_combined_dice_bce_loss_multiclass_ignore_index():
    """Verify that CrossEntropyLoss supervises background (class 0) even when include_background=False."""
    loss_exc = CombinedDiceBCELoss3D(num_classes=4, include_background=False)
    loss_inc = CombinedDiceBCELoss3D(num_classes=4, include_background=True)

    logits = torch.zeros(1, 4, 4, 4, 4)
    logits[:, 0] = -10.0
    logits[:, 1] = 10.0
    targets = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    targets[:, 0, 0, 0] = 1

    res_inc = loss_inc(logits, targets)
    res_exc = loss_exc(logits, targets)
    assert res_inc["bce_loss"].item() > 5.0
    assert res_exc["bce_loss"].item() > 5.0


def test_p1_step9_dice_background_contract():
    """Binary task invariant to include_background; 4-class differs; CE covers all voxels."""
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
    res = CombinedDiceBCELoss3D(include_background=False)(logits_m, target_m)
    assert torch.isfinite(res["loss"]).item() and torch.isfinite(res["bce_loss"]).item()


def test_dice_loss_multiclass():
    """Dice loss should handle multi-class segmentation with per-class computation."""
    criterion = VolumetricDiceLoss(num_classes=4)
    logits = torch.randn(2, 4, 16, 16, 16, requires_grad=True)
    targets = torch.randint(0, 4, (2, 16, 16, 16))

    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert 0.0 <= loss.item() <= 1.0

    loss.backward()
    assert logits.grad is not None


def test_dice_loss_perfect_multiclass():
    """Dice loss should be ~0 when predictions perfectly match targets."""
    criterion = VolumetricDiceLoss(num_classes=4)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    logits = torch.zeros(2, 4, 8, 8, 8)
    for c in range(4):
        logits[:, c][targets == c] = 100.0

    loss = criterion(logits, targets)
    assert loss.item() < 0.01


def test_losses_multiclass_active():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8, 8, 8)
    target = torch.randint(0, 5, (2, 1, 8, 8, 8)).float()
    for crit in (
        VolumetricDiceLoss(),
        VolumetricDiceLoss(include_background=False),
        VolumetricTverskyLoss(),
        CombinedDiceBCELoss3D(),
        CombinedTverskyBCEWithLogitsLoss3D(),
    ):
        out = crit(logits, target)
        val = out["loss"] if isinstance(out, dict) else out
        assert torch.isfinite(val).item(), type(crit).__name__
    a = VolumetricTverskyLoss(alpha=0.5, beta=0.5)(logits, target).item()
    b = VolumetricDiceLoss(squared_pred=False)(logits, target).item()
    assert abs(a - b) < 1e-4


def test_tversky_bce_binary_unchanged():
    torch.manual_seed(1)
    logits = torch.randn(2, 1, 8, 8, 8)
    target = (torch.rand(2, 1, 8, 8, 8) > 0.9).float()
    res = CombinedTverskyBCEWithLogitsLoss3D()(logits, target)
    assert set(res) == {"loss", "dice_loss", "bce_loss", "tversky_loss"}


# =========================================================================
# 3. Tversky Losses
# =========================================================================


def _toy(pred_val=0.0, tumor=True):
    torch.manual_seed(0)
    logits = torch.full((1, 1, 16, 16, 16), pred_val)
    target = torch.zeros(1, 1, 16, 16, 16)
    if tumor:
        target[:, :, 4:12, 4:12, 4:12] = 1.0
    return logits, target


def test_tversky_reduces_to_dice_at_half_weights():
    logits, target = _toy()
    t_half = VolumetricTverskyLoss(alpha=0.5, beta=0.5)(logits, target).item()
    d = VolumetricDiceLoss(squared_pred=False)(logits, target).item()
    assert abs(t_half - d) < 1e-4, f"{t_half} vs {d}"


def test_tversky_penalizes_fn_more_than_fp():
    torch.manual_seed(0)
    tumor = torch.zeros(1, 1, 16, 16, 16)
    tumor[:, :, 4:12, 4:12, 4:12] = 1.0
    loss = VolumetricTverskyLoss(alpha=0.3, beta=0.7)
    l_fn = loss(torch.full((1, 1, 16, 16, 16), -10.0), tumor).item()
    l_fp = loss(torch.full((1, 1, 16, 16, 16), 10.0), torch.zeros_like(tumor)).item()
    n_tumor = tumor.sum().item()
    n_bg = tumor.numel() - n_tumor
    assert l_fn / n_tumor > l_fp / n_bg, f"per-voxel fn={l_fn/n_tumor:.5f} fp={l_fp/n_bg:.5f}"


def test_tversky_asymmetry_ratio():
    loss = VolumetricTverskyLoss(alpha=0.3, beta=0.7)
    assert loss.beta / loss.alpha == pytest.approx(0.7 / 0.3)


def test_combined_tversky_contract_and_grad():
    logits, target = _toy()
    crit = CombinedTverskyBCEWithLogitsLoss3D()
    logits.requires_grad_(True)
    res = crit(logits, target)
    assert set(res) == {"loss", "dice_loss", "bce_loss", "tversky_loss"}
    assert torch.isfinite(res["loss"]).item()
    res["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_factory_defaults_match_legacy():
    c1 = build_segmentation_criterion("dice_bce", deep_supervision=True)
    assert isinstance(c1, DeepSupervisionLoss3D)
    assert isinstance(c1.base_loss, CombinedDiceBCELoss3D)
    c2 = build_segmentation_criterion("dice_bce", deep_supervision=False)
    assert isinstance(c2, CombinedDiceBCELoss3D)
    c3 = build_segmentation_criterion("tversky", deep_supervision=True)
    assert isinstance(c3, DeepSupervisionLoss3D)
    assert isinstance(c3.base_loss, CombinedTverskyBCEWithLogitsLoss3D)
    with pytest.raises(ValueError):
        build_segmentation_criterion("focal")


def _ns(**kw):
    class _A:
        pass
    a = _A()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_resolve_seg_loss_type_collision():
    assert resolve_seg_loss_type(_ns(loss_type="smooth_l1"), cli_args=[]) == "dice_bce"
    assert resolve_seg_loss_type(_ns(loss_type="tversky"), cli_args=["--loss_type=tversky"]) == "tversky"
    assert resolve_seg_loss_type(_ns(), cli_args=[]) == "dice_bce"


def test_tversky_ds_multiscale_backward():
    torch.manual_seed(0)
    crit = build_segmentation_criterion("tversky", deep_supervision=True)
    heads = [torch.randn(1, 1, s, s, s, requires_grad=True) for s in (32, 16, 8, 4)]
    target = (torch.rand(1, 1, 32, 32, 32) > 0.9).float()
    loss = crit(heads, target)["loss"]
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(torch.isfinite(h.grad).all() for h in heads)


def test_f10_tversky_include_background_flag():
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


# =========================================================================
# 4. Deep Supervision Losses
# =========================================================================


def test_deep_supervision_loss_3d():
    criterion = DeepSupervisionLoss3D()
    logits_list = [
        torch.randn(2, 1, 32, 32, 32, requires_grad=True),
        torch.randn(2, 1, 16, 16, 16, requires_grad=True),
        torch.randn(2, 1, 8, 8, 8, requires_grad=True),
    ]
    targets = (torch.rand(2, 1, 32, 32, 32) > 0.8).float()

    res = criterion(logits_list, targets)
    assert "loss" in res
    res["loss"].backward()
    assert logits_list[0].grad is not None


def test_deep_supervision_4d_targets():
    """DeepSupervisionLoss3D should handle 4D targets [B, D, H, W] without channel dimension."""
    criterion = DeepSupervisionLoss3D()
    logits_list = [
        torch.randn(2, 1, 32, 32, 32, requires_grad=True),
        torch.randn(2, 1, 16, 16, 16, requires_grad=True),
    ]
    targets_4d = (torch.rand(2, 32, 32, 32) > 0.8).float()
    res = criterion(logits_list, targets_4d)
    assert "loss" in res
    assert torch.isfinite(res["loss"])
    res["loss"].backward()
    assert logits_list[0].grad is not None


def test_deep_supervision_loss_integer_target_dtype():
    """Verify DeepSupervisionLoss3D handles int64/uint8/bool targets without dtype crashes."""
    loss_fn = DeepSupervisionLoss3D()
    logits = [
        torch.randn(2, 1, 16, 16, 16, requires_grad=True),
        torch.randn(2, 1, 8, 8, 8, requires_grad=True),
    ]
    for dtype in [torch.int64, torch.uint8, torch.bool]:
        targets = (torch.rand(2, 1, 16, 16, 16) > 0.5).to(dtype=dtype)
        res = loss_fn(logits, targets)
        assert "loss" in res
        assert not torch.isnan(res["loss"])
        res["loss"].backward()
        assert logits[0].grad is not None
        logits[0].grad.zero_()


def test_deep_supervision_loss_variable_heads():
    """Verify DeepSupervisionLoss3D dynamically weights any number of heads (>4) and flows gradients to all."""
    criterion = DeepSupervisionLoss3D()
    num_heads = 5
    weights_5 = criterion.get_weights(num_heads)
    assert len(weights_5) == 5
    assert abs(sum(weights_5) - 1.0) < 1e-6
