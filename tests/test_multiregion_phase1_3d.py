import pytest
import torch

from brats_jepa_3d.losses import (
    CombinedDiceBCELoss3D,
    CombinedTverskyBCEWithLogitsLoss3D,
    VolumetricDiceLoss,
    VolumetricTverskyLoss,
)
from brats_jepa_3d.metrics import (
    BRATS_REGIONS,
    brats_region_masks_from_logits,
    brats_region_masks_from_target,
    compute_brats_regions_3d,
)
from brats_jepa_3d.utils import predict_with_tta_3d, predict_with_tta_multiclass_3d


def _hand_built():
    """logits [1,5,4,4,4] whose argmax paints known voxels per class."""
    logits = torch.full((1, 5, 4, 4, 4), -10.0)
    logits[0, 1, 0, 0, 0] = 10.0  # NCR
    logits[0, 2, 0, 0, 1] = 10.0  # ED
    logits[0, 3, 0, 1, 0] = 10.0  # NET
    logits[0, 4, 0, 1, 1] = 10.0  # ET
    target = torch.zeros(1, 1, 4, 4, 4)
    target[0, 0, 0, 0, 0] = 1.0
    target[0, 0, 0, 0, 1] = 2.0
    target[0, 0, 0, 1, 0] = 3.0
    target[0, 0, 0, 1, 1] = 4.0
    return logits, target


def test_region_definitions():
    assert BRATS_REGIONS == {"WT": (1, 2, 3, 4), "TC": (1, 3, 4), "ET": (4,)}


def test_region_derivation_hand_built():
    logits, target = _hand_built()
    pm = brats_region_masks_from_logits(logits)
    tm = brats_region_masks_from_target(target)
    assert pm["WT"].sum().item() == 4
    assert pm["TC"].sum().item() == 3
    assert pm["ET"].sum().item() == 1
    for k in ("WT", "TC", "ET"):
        assert torch.equal(pm[k], tm[k])


def test_region_metrics_perfect_and_et_free_nan():
    logits, target = _hand_built()
    res = compute_brats_regions_3d(logits, target, compute_hd95=False)
    assert res["WT"]["dice"] == 1.0 and res["TC"]["dice"] == 1.0 and res["ET"]["dice"] == 1.0
    # ET-free cohort: ET dice falls back gracefully, ET HD95 is NaN (never 0.0).
    tgt_noet = target.clone()
    tgt_noet[tgt_noet == 4] = 2.0
    res2 = compute_brats_regions_3d(logits, tgt_noet, compute_hd95=True)
    assert res2["ET"]["hd95_tumor_only"] != res2["ET"]["hd95_tumor_only"]
    assert res2["WT"]["dice"] > 0.0


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
    # Tversky(0.5, 0.5) == linear Dice on the same multiclass input.
    a = VolumetricTverskyLoss(alpha=0.5, beta=0.5)(logits, target).item()
    b = VolumetricDiceLoss(squared_pred=False)(logits, target).item()
    assert abs(a - b) < 1e-4


def test_tversky_bce_binary_unchanged():
    torch.manual_seed(1)
    logits = torch.randn(2, 1, 8, 8, 8)
    target = (torch.rand(2, 1, 8, 8, 8) > 0.9).float()
    res = CombinedTverskyBCEWithLogitsLoss3D()(logits, target)
    assert set(res) == {"loss", "dice_loss", "bce_loss", "tversky_loss"}


class _Fixed(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        torch.manual_seed(3)
        self.base = torch.randn(1, c, 8, 8, 8)

    def forward(self, x):
        return self.base.expand(x.shape[0], -1, -1, -1, -1)


def test_tta_multiclass_contract_and_binary_guard():
    m5 = _Fixed(5).eval()
    vol = torch.randn(1, 4, 8, 8, 8)
    with torch.no_grad():
        out = predict_with_tta_multiclass_3d(m5, vol)
    assert tuple(out.shape) == (1, 5, 8, 8, 8)
    # probabilities still sum to one after the flip-average roundtrip
    assert torch.allclose(out.softmax(dim=1).sum(dim=1), torch.ones(1, 8, 8, 8), atol=1e-4)
    m1 = _Fixed(1).eval()
    with torch.no_grad():
        predict_with_tta_3d(m1, vol)  # binary path unaffected
        with pytest.raises(ValueError, match="binary-only"):
            predict_with_tta_3d(_Fixed(4).eval(), vol)
        with pytest.raises(ValueError, match="C>=2"):
            predict_with_tta_multiclass_3d(m1, vol)


def test_factory_num_classes_threading():
    from brats_jepa_3d.losses import build_segmentation_criterion
    legacy = build_segmentation_criterion("dice_bce", True)
    assert legacy.base_loss.num_classes == 4
    v2 = build_segmentation_criterion("dice_bce", True, num_classes=5)
    assert v2.base_loss.num_classes == 5
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 5, (1, 1, 8, 8, 8)).float()
    loss = v2([logits], target)["loss"]
    loss.backward()
    assert torch.isfinite(loss).item()


def test_models_out_channels_5():
    from brats_jepa_3d.models import BraTS3DUNet, JEPASegmentationModel3D
    torch.manual_seed(0)
    unet = BraTS3DUNet(channels=(8, 16, 32, 64, 128), num_res_units=1, out_channels=5)
    unet.train()
    with torch.no_grad():
        outs = unet(torch.randn(1, 4, 32, 32, 32))
    assert [tuple(o.shape) for o in outs] == [
        (1, 5, 32, 32, 32), (1, 5, 16, 16, 16), (1, 5, 8, 8, 8), (1, 5, 4, 4, 4)]
    seg = JEPASegmentationModel3D(
        img_size=(32, 32, 32), patch_size=(16, 16, 16), encoder_depth=1,
        decoder_type="multiscale", out_channels=5)
    seg.eval()
    with torch.no_grad():
        y = seg(torch.randn(1, 4, 32, 32, 32))
    assert tuple(y.shape) == (1, 5, 32, 32, 32)


def test_benchmark_model_region_rows():
    import sys
    sys.path.insert(0, "scripts")
    from torch.utils.data import DataLoader

    import evaluate_3d

    torch.manual_seed(0)
    base = torch.randn(1, 5, 16, 16, 16)

    class _M(torch.nn.Module):
        def forward(self, x):
            return base.expand(x.shape[0], -1, -1, -1, -1)

    data = [
        {"image": torch.randn(4, 16, 16, 16),
         "mask": torch.randint(0, 5, (1, 16, 16, 16)).float()}
        for _ in range(2)
    ]
    loader = DataLoader(data, batch_size=1)
    stats = evaluate_3d.benchmark_model(
        _M().eval(), loader, torch.device("cpu"), amp=False, smoke_test=True)
    assert set(stats["regions"]) == {"WT", "TC", "ET"}
    assert stats["regions"]["WT"]["dice_mean"] >= 0.0
    assert stats["latency_ms"] >= 0.0


def test_channel_mismatch_raises_actionable_error():
    import pytest

    from brats_jepa_3d.models import JEPASegmentationModel3D, load_downstream_state_dict
    torch.manual_seed(0)
    c1 = JEPASegmentationModel3D(
        img_size=(32, 32, 32), patch_size=(16, 16, 16), encoder_depth=1,
        decoder_type="multiscale", out_channels=1)
    c5 = JEPASegmentationModel3D(
        img_size=(32, 32, 32), patch_size=(16, 16, 16), encoder_depth=1,
        decoder_type="multiscale", out_channels=5)
    sd1 = c1.state_dict()
    with pytest.raises(RuntimeError, match="out_channels mismatch"):
        load_downstream_state_dict(c5, sd1, "test-model")
    # matching channels load cleanly
    missing, _ = load_downstream_state_dict(c5, c5.state_dict(), "test-model")
    assert missing == []
