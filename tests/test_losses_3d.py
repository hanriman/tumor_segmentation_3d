import torch

from brats_jepa_3d.losses import (
    CombinedDiceBCELoss3D,
    DeepSupervisionLoss3D,
    IJEPALoss,
    SigRegLoss,
    VisRegLoss,
)


def test_ijepa_loss():
    criterion = IJEPALoss(loss_type="smooth_l1")
    preds = [torch.randn(2, 27, 384, requires_grad=True)]
    targets = [torch.randn(2, 27, 384)]

    loss = criterion(preds, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    assert preds[0].grad is not None


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


def test_sigreg_loss_gaussian_input():
    """SigReg EP test should produce near-zero loss for standard Gaussian input."""
    torch.manual_seed(123)
    criterion = SigRegLoss(sigreg_weight=1.0, num_projections=128)
    preds = [torch.randn(2, 27, 384)]
    targets = [torch.randn(2, 27, 384)]
    # Standard Gaussian projected tokens: EP statistic should be small
    proj_tokens = torch.randn(2, 192, 128)

    res = criterion(preds, targets, projected_tokens=proj_tokens)
    # The SigReg (EP) loss component should be small for standardized Gaussian input
    assert res["sigreg_loss"].item() < res["jepa_loss"].item() * 5.0, (
        f"SigReg loss {res['sigreg_loss'].item():.4f} unexpectedly large for Gaussian input "
        f"(JEPA loss: {res['jepa_loss'].item():.4f})"
    )


def test_dice_loss_multiclass():
    """Dice loss should handle multi-class segmentation with per-class computation."""
    from brats_jepa_3d.losses.dice_bce_loss_3d import VolumetricDiceLoss

    criterion = VolumetricDiceLoss(num_classes=4)
    # 4-class logits
    logits = torch.randn(2, 4, 16, 16, 16, requires_grad=True)
    # Integer class targets
    targets = torch.randint(0, 4, (2, 16, 16, 16))

    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert 0.0 <= loss.item() <= 1.0, f"Dice loss should be in [0,1], got {loss.item()}"

    loss.backward()
    assert logits.grad is not None


def test_dice_loss_perfect_multiclass():
    """Dice loss should be ~0 when predictions perfectly match targets."""
    from brats_jepa_3d.losses.dice_bce_loss_3d import VolumetricDiceLoss

    criterion = VolumetricDiceLoss(num_classes=4)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    # Create perfect predictions: one-hot with very high logits
    logits = torch.zeros(2, 4, 8, 8, 8)
    for c in range(4):
        logits[:, c][targets == c] = 100.0  # Very confident correct prediction

    loss = criterion(logits, targets)
    assert loss.item() < 0.01, f"Perfect predictions should give near-zero Dice loss, got {loss.item()}"


def test_sigreg_anti_collapse_penalty():
    """SigReg loss should strongly penalize collapsed representation compared to isotropic Gaussian."""
    torch.manual_seed(42)
    criterion = SigRegLoss(sigreg_weight=1.0, num_projections=64)
    preds = [torch.randn(2, 27, 384)]
    targets = [torch.randn(2, 27, 384)]

    # 1. Standard isotropic Gaussian representations
    proj_normal = torch.randn(2, 192, 128)
    res_normal = criterion(preds, targets, projected_tokens=proj_normal)

    # 2. Collapsed representations (scaled close to 0)
    proj_collapsed = torch.randn(2, 192, 128) * 0.01
    res_collapsed = criterion(preds, targets, projected_tokens=proj_collapsed)

    # SigReg loss should be significantly higher for collapsed tokens than standard normal tokens
    assert res_collapsed["sigreg_loss"].item() > 10.0 * res_normal["sigreg_loss"].item(), (
        f"Collapsed SigReg loss {res_collapsed['sigreg_loss'].item():.4f} should be > 10x normal loss {res_normal['sigreg_loss'].item():.4f}"
    )


def test_deep_supervision_4d_targets():
    """DeepSupervisionLoss3D should handle 4D targets [B, D, H, W] without channel dimension."""
    criterion = DeepSupervisionLoss3D()
    logits_list = [
        torch.randn(2, 1, 32, 32, 32, requires_grad=True),
        torch.randn(2, 1, 16, 16, 16, requires_grad=True),
    ]
    # 4D targets without channel dimension
    targets_4d = (torch.rand(2, 32, 32, 32) > 0.8).float()

    res = criterion(logits_list, targets_4d)
    assert "loss" in res
    assert res["loss"].dim() == 0
    res["loss"].backward()
    assert logits_list[0].grad is not None
    assert logits_list[1].grad is not None

