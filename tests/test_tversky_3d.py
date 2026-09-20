import pytest
import torch

from brats_jepa_3d.losses import (
    CombinedDiceBCELoss3D,
    CombinedTverskyBCEWithLogitsLoss3D,
    DeepSupervisionLoss3D,
    VolumetricTverskyLoss,
    build_segmentation_criterion,
    resolve_seg_loss_type,
)


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
    from brats_jepa_3d.losses import VolumetricDiceLoss
    # alpha=beta=0.5 Tversky == linear-cardinality Dice (squared_pred=False).
    d = VolumetricDiceLoss(squared_pred=False)(logits, target).item()
    assert abs(t_half - d) < 1e-4, f"{t_half} vs {d}"


def test_tversky_penalizes_fn_more_than_fp():
    # Same count of errors: all-FN (predict empty on tumor) vs all-FP (predict full on empty).
    torch.manual_seed(0)
    tumor = torch.zeros(1, 1, 16, 16, 16)
    tumor[:, :, 4:12, 4:12, 4:12] = 1.0
    loss = VolumetricTverskyLoss(alpha=0.3, beta=0.7)
    l_fn = loss(torch.full((1, 1, 16, 16, 16), -10.0), tumor).item()
    l_fp = loss(torch.full((1, 1, 16, 16, 16), 10.0), torch.zeros_like(tumor)).item()
    n_tumor = tumor.sum().item()
    n_bg = tumor.numel() - n_tumor
    # Normalize by error count: per-voxel FN penalty must exceed per-voxel FP penalty.
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
    # JEPA model YAMLs share the loss_type key (smooth_l1): ignored unless explicit CLI.
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
