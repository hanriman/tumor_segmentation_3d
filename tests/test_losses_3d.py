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
