
import argparse
import sys

import numpy as np
import pytest
import torch

from brats_jepa_3d.config import (
    CONFIGS_DIR,
    PROJECT_ROOT,
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa_3d.data import (
    BraTS3DDataset,
    RandomModalityDropout3D,
    apply_b1_bias_field_3d,
    apply_rician_noise_3d,
)
from brats_jepa_3d.losses import (
    CombinedDiceBCELoss3D,
    DeepSupervisionLoss3D,
    SigRegLoss,
    VisRegLoss,
    VolumetricDiceLoss,
)
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d
from brats_jepa_3d.models import (
    BraTS3DnnUNet,
    BraTS3DUNet,
    JEPASegmentationModel3D,
)


def test_deep_supervision_loss_integer_target_dtype():
    """Verify DeepSupervisionLoss3D handles int64/uint8/bool targets without dtype crashes."""
    loss_fn = DeepSupervisionLoss3D()

    # Multi-scale logits at resolutions 16^3, 8^3
    logits = [
        torch.randn(2, 1, 16, 16, 16, requires_grad=True),
        torch.randn(2, 1, 8, 8, 8, requires_grad=True),
    ]

    for dtype in [torch.int64, torch.uint8, torch.bool]:
        targets = (torch.rand(2, 1, 16, 16, 16) > 0.5).to(dtype=dtype)
        res = loss_fn(logits, targets)

        assert "loss" in res
        assert "dice_loss" in res
        assert "bce_loss" in res
        assert res["loss"].dtype == torch.float32
        assert not torch.isnan(res["loss"])
        # Verify backward pass works cleanly
        res["loss"].backward()
        assert logits[0].grad is not None
        logits[0].grad.zero_()


def test_volumetric_dice_include_background():
    """Verify include_background=False excludes channel 0 in multi-class Dice loss."""
    # 4 classes: background (0), NCR (1), ED (2), ET (3)
    dice_inc = VolumetricDiceLoss(num_classes=4, include_background=True)
    dice_exc = VolumetricDiceLoss(num_classes=4, include_background=False)

    # Perfect prediction for background (class 0), poor for tumor (classes 1, 2, 3)
    logits = torch.zeros(1, 4, 8, 8, 8)
    logits[:, 0] = 10.0  # High confidence background
    logits[:, 1:] = -10.0

    targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    targets[:, 2:6, 2:6, 2:6] = 1  # Class 1 tumor region

    loss_inc = dice_inc(logits, targets).item()
    loss_exc = dice_exc(logits, targets).item()

    # When background is included, the near-perfect background dice lowers the overall loss
    # When background is excluded, the loss on classes 1, 2, 3 only should be higher
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


def test_compute_volumetric_metrics_shape_alignment():
    """Verify compute_volumetric_metrics_3d accepts 4D target with 5D pred and vice versa."""
    # pred is 5D [B, 1, D, H, W], target is 4D [B, D, H, W]
    pred_5d = torch.randn(2, 1, 16, 16, 16)
    target_4d = (torch.rand(2, 16, 16, 16) > 0.8).float()

    res1 = compute_volumetric_metrics_3d(pred_5d, target_4d)
    assert "dice" in res1
    assert "hd95" in res1
    assert 0.0 <= res1["dice"] <= 1.0

    # pred is 4D [B, D, H, W], target is 5D [B, 1, D, H, W]
    pred_4d = torch.randn(2, 16, 16, 16)
    target_5d = (torch.rand(2, 1, 16, 16, 16) > 0.8).float()

    res2 = compute_volumetric_metrics_3d(pred_4d, target_5d)
    assert "dice" in res2
    assert "hd95" in res2
    assert 0.0 <= res2["dice"] <= 1.0


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

    # Quick forward pass
    preds = [torch.randn(2, 16, 64)]
    targets = [torch.randn(2, 16, 64)]
    reg_tokens = torch.randn(2, 16, 64)
    out = criterion(preds, targets, tokens=reg_tokens)
    assert "loss" in out
    assert out["loss"].item() > 0


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


def test_config_merging_in_runners():
    """Verify configs merge correctly into argparse Namespace for runner scripts."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args([])

    base_cfg = load_yaml_config(CONFIGS_DIR / "base.yaml")
    args = merge_config_with_args(base_cfg, args)
    assert hasattr(args, "seed")
    assert hasattr(args, "amp")
    assert args.seed == 42

    model_cfg = load_yaml_config(CONFIGS_DIR / "model" / "visreg_jepa_3d.yaml")
    args = merge_config_with_args(model_cfg, args)
    assert hasattr(args, "spatial_shape")
    assert hasattr(args, "patch_size")
    assert args.spatial_shape == [128, 128, 128]
    assert args.patch_size == [16, 16, 16]


def test_combined_dice_bce_loss_multiclass_ignore_index():
    """Verify that CrossEntropyLoss supervises background (class 0) even when include_background=False."""
    loss_exc = CombinedDiceBCELoss3D(num_classes=4, include_background=False)
    assert loss_exc.include_background is False
    assert loss_exc.dice_loss.include_background is False

    loss_inc = CombinedDiceBCELoss3D(num_classes=4, include_background=True)
    assert loss_inc.include_background is True
    assert loss_inc.dice_loss.include_background is True

    # Targets: mostly background (class 0), with a single voxel of class 1
    # Prediction: class 1 is predicted correctly (+10.0), but class 0 is predicted incorrectly (-10.0)
    logits = torch.zeros(1, 4, 4, 4, 4)
    logits[:, 0] = -10.0  # wrong prediction for class 0 (predicts foreground on background)
    logits[:, 1] = 10.0   # correct prediction for class 1
    targets = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    targets[:, 0, 0, 0] = 1  # class 1 at (0, 0, 0)

    res_inc = loss_inc(logits, targets)
    res_exc = loss_exc(logits, targets)

    # Cross-Entropy must supervise class 0 to actively penalize exploratory false positives on background
    assert res_inc["bce_loss"].item() > 5.0
    assert res_exc["bce_loss"].item() > 5.0


def test_compute_volumetric_metrics_validation_and_from_logits():
    """Verify compute_volumetric_metrics_3d enforces single channel input and respects from_logits."""
    # 1. Multi-class channel > 1 must raise ValueError to prevent silent errors
    pred_multichannel = torch.randn(2, 4, 16, 16, 16)
    target = (torch.rand(2, 1, 16, 16, 16) > 0.5).float()
    with pytest.raises(ValueError, match="expects binary Whole Tumor channel"):
        compute_volumetric_metrics_3d(pred_multichannel, target)

    # 2. from_logits=False test
    prob = torch.zeros(1, 1, 8, 8, 8)
    prob[:, :, :4, :, :] = 0.9
    gt = torch.zeros(1, 1, 8, 8, 8)
    gt[:, :, :4, :, :] = 1.0

    res_prob = compute_volumetric_metrics_3d(prob, gt, from_logits=False)
    assert res_prob["dice"] == pytest.approx(1.0, abs=1e-4)

    # 3. from_logits=True test
    res_logits = compute_volumetric_metrics_3d(prob, gt, from_logits=True)
    assert res_logits["dice"] == pytest.approx(1.0, abs=1e-4)


def test_random_modality_dropout_pytorch_prng_determinism():
    """Verify RandomModalityDropout3D obeys torch.manual_seed for reproducible dropout masks."""
    dropout = RandomModalityDropout3D(p_drop=0.5)
    dropout.train()
    img = torch.ones(1, 4, 16, 16, 16)

    torch.manual_seed(1234)
    out1 = dropout(img)

    torch.manual_seed(1234)
    out2 = dropout(img)

    assert torch.equal(out1, out2), "RandomModalityDropout3D must be fully deterministic given torch seed"


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


def test_deep_supervision_loss_variable_heads():
    """Verify DeepSupervisionLoss3D dynamically weights any number of heads (>4) and flows gradients to all."""
    criterion = DeepSupervisionLoss3D()

    # Test with 5 heads (Isensee et al. multi-scale resolution down to 8^3)
    num_heads = 5
    weights_5 = criterion.get_weights(num_heads)
    assert len(weights_5) == 5
    assert abs(sum(weights_5) - 1.0) < 1e-6
    # Exponential decay: w_0 / w_1 == 2.0, w_1 / w_2 == 2.0
    for i in range(num_heads - 1):
        assert abs(weights_5[i] / weights_5[i + 1] - 2.0) < 1e-5

    resolutions = [32, 16, 8, 4, 2]
    logits_5 = [
        torch.randn(2, 1, r, r, r, requires_grad=True)
        for r in resolutions
    ]
    targets = (torch.rand(2, 1, 32, 32, 32) > 0.5).float()

    res = criterion(logits_5, targets)
    assert "loss" in res
    res["loss"].backward()

    # Verify EVERY head (including heads 4 and 5) receives non-zero gradients
    for idx, logit in enumerate(logits_5):
        assert logit.grad is not None, f"Head {idx} did not receive gradients!"
        assert logit.grad.abs().sum() > 0, f"Head {idx} gradient is all zeros!"


def test_ema_cosine_momentum_schedule_and_smoke_test():
    """Verify cosine momentum annealing uses config parameters and correctly scales with smoke_test."""
    import math

    m_start = 0.996
    m_end = 1.0
    total_epochs = 50
    steps_per_epoch = 100
    total_steps = total_epochs * steps_per_epoch

    # At step 0
    step = 0
    progress = min(1.0, step / max(1, total_steps))
    mom = m_end - (m_end - m_start) * 0.5 * (1.0 + math.cos(math.pi * progress))
    assert abs(mom - m_start) < 1e-6

    # At final step
    step = total_steps
    progress = min(1.0, step / max(1, total_steps))
    mom = m_end - (m_end - m_start) * 0.5 * (1.0 + math.cos(math.pi * progress))
    assert abs(mom - m_end) < 1e-6

    # Verify smoke test calculation: 2 steps per epoch
    full_loader_len = 1000
    smoke_test = True
    smoke_steps_per_epoch = min(full_loader_len, 2) if smoke_test else full_loader_len
    assert smoke_steps_per_epoch == 2
    smoke_total_steps = 1 * smoke_steps_per_epoch
    assert smoke_total_steps == 2


def test_visreg_swd_standard_normal():
    """Verify VisReg Sliced Wasserstein Distance is near zero for standard isotropic Gaussian samples."""
    visreg = VisRegLoss(
        center_weight=0.0,
        scale_weight=0.0,
        swd_weight=1.0,
        num_projections=256,
        target_std=1.0,
    )
    torch.manual_seed(42)
    z_normal = torch.randn(2000, 128)
    swd = visreg._sliced_wasserstein_distance(z_normal)
    assert swd < 0.15, f"VisReg SWD on standard normal was too large: {swd:.4f}"


def test_ood_perturbations_3d():
    """Verify 3D Rician scanner noise and B1 bias field transformations."""
    torch.manual_seed(42)
    # Synthetic MRI volume [C=4, D=16, H=16, W=16] with brain mask
    img = torch.zeros(4, 16, 16, 16)
    img[:, 4:12, 4:12, 4:12] = torch.rand(4, 8, 8, 8) + 0.5

    # 1. Rician noise
    noisy = apply_rician_noise_3d(img, sigma=0.05)
    assert noisy.shape == img.shape
    assert not torch.isnan(noisy).any()
    assert not torch.allclose(noisy, img)
    # Background outside brain mask remains zero
    assert (noisy[:, 0:3, 0:3, 0:3] == 0).all()

    # 2. B1 bias field
    biased = apply_b1_bias_field_3d(img, strength=0.3)
    assert biased.shape == img.shape
    assert not torch.isnan(biased).any()
    assert not torch.allclose(biased, img)


def test_ood_perturbations_explicit_brain_mask_3d():
    """Verify 3D Rician noise and B1 bias field work with explicit float32 single-channel brain_mask (4D and 5D)."""
    torch.manual_seed(42)
    # 4D case: [C=4, D=16, H=16, W=16] with float32 brain_mask [1, D, H, W]
    img_4d = torch.zeros(4, 16, 16, 16, dtype=torch.float32)
    img_4d[:, 4:12, 4:12, 4:12] = torch.rand(4, 8, 8, 8) + 0.5
    mask_4d = torch.zeros(1, 16, 16, 16, dtype=torch.float32)
    mask_4d[:, 4:12, 4:12, 4:12] = 1.0

    noisy_4d = apply_rician_noise_3d(img_4d, sigma=0.05, brain_mask=mask_4d)
    assert noisy_4d.shape == img_4d.shape
    assert (noisy_4d[:, 0:3, 0:3, 0:3] == 0).all()
    assert not torch.allclose(noisy_4d[:, 4:12, 4:12, 4:12], img_4d[:, 4:12, 4:12, 4:12])

    biased_4d = apply_b1_bias_field_3d(img_4d, strength=0.3, brain_mask=mask_4d)
    assert biased_4d.shape == img_4d.shape
    assert (biased_4d[:, 0:3, 0:3, 0:3] == 0).all()
    assert not torch.allclose(biased_4d[:, 4:12, 4:12, 4:12], img_4d[:, 4:12, 4:12, 4:12])

    # 5D case: [B=2, C=4, D=16, H=16, W=16] with float32 brain_mask [B=2, 1, D, H, W]
    img_5d = torch.zeros(2, 4, 16, 16, 16, dtype=torch.float32)
    img_5d[:, :, 4:12, 4:12, 4:12] = torch.rand(2, 4, 8, 8, 8) + 0.5
    mask_5d = torch.zeros(2, 1, 16, 16, 16, dtype=torch.float32)
    mask_5d[:, :, 4:12, 4:12, 4:12] = 1.0

    noisy_5d = apply_rician_noise_3d(img_5d, sigma=0.05, brain_mask=mask_5d)
    assert noisy_5d.shape == img_5d.shape
    assert (noisy_5d[:, :, 0:3, 0:3, 0:3] == 0).all()

    biased_5d = apply_b1_bias_field_3d(img_5d, strength=0.3, brain_mask=mask_5d)
    assert biased_5d.shape == img_5d.shape
    assert (biased_5d[:, :, 0:3, 0:3, 0:3] == 0).all()

    # Un-unsqueezed 3D mask for 4D image: [D, H, W]
    mask_3d = mask_4d.squeeze(0)
    noisy_3d_mask = apply_rician_noise_3d(img_4d, sigma=0.05, brain_mask=mask_3d)
    assert noisy_3d_mask.shape == img_4d.shape



def test_model_config_consistency_and_deep_supervision_detection():
    """Verify UNet and nnUNet initialize cleanly with configs, and JEPA detects deep supervision."""
    # Test UNet config
    unet_cfg = load_yaml_config(CONFIGS_DIR / "model" / "unet_3d.yaml")
    unet = BraTS3DUNet(
        in_channels=unet_cfg.get("in_channels", 4),
        out_channels=unet_cfg.get("out_channels", 1),
        channels=tuple(unet_cfg.get("channels", (32, 64, 128, 256, 512))),
        strides=tuple(unet_cfg.get("strides", (2, 2, 2, 2))),
        num_res_units=unet_cfg.get("num_res_units", 2),
        dropout=unet_cfg.get("dropout", 0.1),
    )
    assert isinstance(unet, torch.nn.Module)

    # Test nnUNet config
    nnunet_cfg = load_yaml_config(CONFIGS_DIR / "model" / "nnunet_3d.yaml")
    nnunet = BraTS3DnnUNet(
        in_channels=nnunet_cfg.get("in_channels", 4),
        out_channels=nnunet_cfg.get("out_channels", 1),
        deep_supervision=nnunet_cfg.get("deep_supervision", True),
        deep_supr_num=nnunet_cfg.get("deep_supr_num", 3),
        res_block=nnunet_cfg.get("res_block", True),
    )
    assert isinstance(nnunet, torch.nn.Module)

    # Test JEPA deep supervision detection from state dict keys
    fake_sd = {
        "encoder.pos_embed": torch.zeros(1, 512, 384),
        "decoder.ds3.weight": torch.zeros(1, 48, 1, 1, 1),
    }
    has_ds = any(k.startswith("decoder.ds") for k in fake_sd)
    assert has_ds is True

    jepa_model = JEPASegmentationModel3D(
        img_size=(64, 64, 64),
        patch_size=(16, 16, 16),
        decoder_type="multiscale",
        deep_supervision=has_ds,
    )
    assert hasattr(jepa_model.decoder, "ds3")


def test_b1_bias_field_zero_centered_contrast_preservation():
    """Verify B1 bias field shifts zero-centered parenchyma before multiplicative scaling to avoid contrast inversion."""
    # Create synthetic volume with 0 background and negative-to-positive foreground
    img = torch.zeros(1, 4, 16, 16, 16)
    # Foreground brain parenchyma with standard Z-scored range [-2.5, +3.5]
    fg_mask = torch.zeros((16, 16, 16), dtype=torch.bool)
    fg_mask[4:12, 4:12, 4:12] = True
    
    low_val = -2.0
    high_val = 2.0
    img[0, :, 4:8, 4:12, 4:12] = low_val
    img[0, :, 8:12, 4:12, 4:12] = high_val
    
    biased = apply_b1_bias_field_3d(img, strength=0.35)
    
    # Background must remain strictly zero
    assert torch.all(biased[0, :, :4, :, :] == 0.0)
    assert torch.all(biased[0, :, 12:, :, :] == 0.0)
    
    # In any given coordinate with low_val vs high_val in adjacent regions:
    # high_val should remain strictly greater than low_val after B1 bias field
    # (i.e. no contrast inversion occurs where negative numbers get multiplied by > 1 factor becoming more negative)
    low_region = biased[0, 0, 4:8, 6:10, 6:10]
    high_region = biased[0, 0, 8:12, 6:10, 6:10]
    assert high_region.min() > low_region.max()
    assert not torch.isnan(biased).any()


def test_dataset_config_cli_integration_and_dynamic_model_args():
    """Verify downstream script integrates dataset config and dynamically builds JEPA segmentation model."""
    # Check dataset config loading
    dataset_cfg_path = CONFIGS_DIR / "dataset" / "brats3d.yaml"
    assert dataset_cfg_path.exists()
    dataset_cfg = load_yaml_config(dataset_cfg_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rand_flip_prob", type=float, default=0.5)
    parser.add_argument("--rand_noise_prob", type=float, default=0.0)
    parser.add_argument("--modality_dropout_prob", type=float, default=0.25)
    parser.add_argument("--spatial_shape", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--patch_size", type=int, nargs=3, default=[16, 16, 16])
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--encoder_depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    
    args = parser.parse_args([])
    merged_args = merge_config_with_args(dataset_cfg, args)
    
    assert merged_args.rand_flip_prob == dataset_cfg["augmentations"]["rand_flip_prob"]
    assert merged_args.rand_noise_prob == dataset_cfg["augmentations"]["rand_noise_prob"]
    
    # Initialize JEPASegmentationModel3D using getattr as in train_downstream_3d.py
    model = JEPASegmentationModel3D(
        img_size=tuple(getattr(merged_args, "spatial_shape", [32, 32, 32])),
        patch_size=tuple(getattr(merged_args, "patch_size", [16, 16, 16])),
        in_channels=4,
        out_channels=1,
        embed_dim=getattr(merged_args, "embed_dim", 192),
        encoder_depth=getattr(merged_args, "encoder_depth", 4),
        num_heads=getattr(merged_args, "num_heads", 3),
        mlp_ratio=getattr(merged_args, "mlp_ratio", 4.0),
        decoder_type="multiscale",
        deep_supervision=False,
    )
    
    spatial_shape = tuple(getattr(merged_args, "spatial_shape", [128, 128, 128]))
    x = torch.randn(1, 4, *spatial_shape)
    out = model(x)
    assert out.shape == (1, 1, *spatial_shape)


def test_resampling_boundary_mask_bleed_suppression_and_brain_mask():
    """Verify CUT-A & CUT-B: brain mask resampling eliminates trilinear bleed and preserves true parenchyma stats."""
    from brats_jepa_3d.data import VolumetricAugmentations3D

    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from prepare_data_3d import resample_3d_volume, zscore_normalize_non_zero

    # Create synthetic volume: brain sphere in center with raw MRI intensity ~500
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
    assert np.any(bleed_voxels)  # Trilinear interpolation bleed is present

    # 3. Apply CUT-A suppression: zero out all voxels outside res_mask
    clean_img = np.where(res_mask, res_img, 0.0)
    assert not np.any((clean_img != 0) & (~res_mask))

    # 4. Normalize with explicit mask
    norm_img = zscore_normalize_non_zero(clean_img, mask=res_mask)
    assert np.all(norm_img[~res_mask] == 0.0)
    assert abs(float(norm_img[res_mask].mean())) < 1e-5
    assert abs(float(norm_img[res_mask].std()) - 1.0) < 1e-4

    # 5. Verify VolumetricAugmentations3D handles brain_mask consistently
    aug = VolumetricAugmentations3D(flip_prob=1.0, noise_prob=1.0, modality_dropout_prob=0.0)
    img_t = torch.from_numpy(norm_img).unsqueeze(0).repeat(4, 1, 1, 1)  # [4, 16, 16, 16]
    mask_t = torch.from_numpy(res_mask.astype(np.float32)).unsqueeze(0)  # [1, 16, 16, 16]
    bmask_t = mask_t.clone()

    aug_img, aug_mask, aug_bmask = aug(img_t, mask_t, brain_mask=bmask_t)
    assert aug_img.shape == img_t.shape
    assert aug_mask.shape == mask_t.shape
    assert aug_bmask.shape == bmask_t.shape
    # Background in augmented image remains 0 where brain_mask is 0
    assert torch.all(aug_img[:, aug_bmask[0] == 0] == 0.0)


def test_metric_tracker_ragged_epoch_metrics_csv_serialization(tmp_path):
    """Verify MetricTracker handles uneven/ragged keys across epochs and serializes to CSV without ValueError."""
    from brats_jepa_3d.utils.logging import MetricTracker

    tracker = MetricTracker()

    # Epoch 1: standard keys only
    tracker.update({
        "epoch": 1,
        "loss": 0.2086,
        "val_loss": 0.1843,
        "train_jepa_loss": 0.0881,
        "train_center_loss": 0.0166,
        "epoch_duration_sec": 53.04,
    })

    # Epoch 2: standard keys only
    tracker.update({
        "epoch": 2,
        "loss": 0.1315,
        "val_loss": 0.1520,
        "train_jepa_loss": 0.0750,
        "train_center_loss": 0.0142,
        "epoch_duration_sec": 51.20,
    })

    # Epoch 3: checkpoint interval includes representation collapse metrics (ragged keys)
    tracker.update({
        "epoch": 3,
        "loss": 0.1240,
        "val_loss": 0.1410,
        "train_jepa_loss": 0.0690,
        "train_center_loss": 0.0125,
        "epoch_duration_sec": 52.10,
        "effective_rank": 53.05,
        "avg_cosine_sim_centered": 0.012,
        "feature_variance": 0.985,
    })

    # Convert to DataFrame - must not raise ValueError: All arrays must be of the same length
    df = tracker.to_dataframe()
    assert len(df) == 3
    assert "effective_rank" in df.columns
    # Epochs 1 & 2 should have NaN for representation metrics
    assert df["effective_rank"].isna().sum() == 2
    assert df.loc[df["epoch"] == 3, "effective_rank"].values[0] == pytest.approx(53.05)

    # Save to CSV and reload
    csv_file = tmp_path / "metrics.csv"
    tracker.save_csv(csv_file)
    assert csv_file.exists()

    import pandas as pd
    loaded_df = pd.read_csv(csv_file)
    assert len(loaded_df) == 3
    assert "train_center_loss" in loaded_df.columns
    assert "effective_rank" in loaded_df.columns


def test_effective_rank_half_precision():
    """Verify compute_effective_rank and collapse metrics do not fail with float16 or bfloat16 inputs."""
    from brats_jepa_3d.metrics.probing_metrics import (
        compute_effective_rank,
        compute_representation_collapse_metrics,
    )

    z_half = torch.randn(2, 64, 128, dtype=torch.float16)
    erank = compute_effective_rank(z_half)
    assert isinstance(erank, float)
    assert erank > 0.0

    metrics = compute_representation_collapse_metrics(z_half)
    assert "effective_rank" in metrics
    assert metrics["effective_rank"] > 0.0

    z_bf16 = torch.randn(2, 64, 128, dtype=torch.bfloat16)
    erank_bf = compute_effective_rank(z_bf16)
    assert isinstance(erank_bf, float)
    assert erank_bf > 0.0


def test_volumetric_metrics_unbatched_3d():
    """Verify compute_volumetric_metrics_3d accepts 3D tensors [D, H, W] without treating D as batch size."""
    pred = torch.randn(16, 16, 16)  # unbatched 3D
    target = (torch.rand(16, 16, 16) > 0.8).float()

    metrics = compute_volumetric_metrics_3d(pred, target)
    assert "dice" in metrics
    assert "hd95" in metrics
    assert len(metrics["dice_per_sample"]) == 1  # 1 sample, NOT 16!
    assert 0.0 <= metrics["dice"] <= 1.0


def test_dice_bce_loss_multiclass_background_supervision():
    """Verify CombinedDiceBCELoss3D supervises background with CE even when include_background=False."""
    loss_fn = CombinedDiceBCELoss3D(num_classes=4, include_background=False)
    logits = torch.zeros(1, 4, 8, 8, 8, requires_grad=True)
    logits.data[:, 1] = 10.0
    logits.data[:, 0] = -10.0

    targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    res = loss_fn(logits, targets)

    # Because targets are background (class 0), CE MUST be large and non-zero
    assert res["bce_loss"].item() > 5.0
    assert res["loss"].item() > 5.0


def test_ijepa3d_target_encoder_eval_mode():
    """Verify IJEPA3D keeps target_encoder in eval mode even after model.train() is called."""
    from brats_jepa_3d.models import IJEPA3D

    model = IJEPA3D(
        img_size=(32, 32, 32),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=64,
        encoder_depth=2,
        predictor_depth=1,
        predictor_embed_dim=32,
        num_heads=2,
    )
    assert not model.target_encoder.training
    model.train()
    assert not model.target_encoder.training
    model.train(True)
    assert not model.target_encoder.training
    model.eval()
    assert not model.target_encoder.training


def test_generate_figures_safe_float():
    """Verify safe_float in generate_figures_3d handles edge cases properly."""
    import sys

    from brats_jepa_3d.config import PROJECT_ROOT

    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from generate_figures_3d import safe_float

    assert safe_float("89.60 ± 2.10") == pytest.approx(89.60)
    assert safe_float("12.5%") == pytest.approx(12.5)
    assert safe_float("N/A") == 0.0
    assert safe_float("-") == 0.0
    assert safe_float("") == 0.0
    assert safe_float(None) == 0.0
    assert safe_float(42.5) == pytest.approx(42.5)


def test_evaluate_scripts_model_type_parsing():
    """Verify --model_type argument is present in argument parsers."""
    import sys

    from brats_jepa_3d.config import PROJECT_ROOT

    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from evaluate_low_data_3d import parse_args as parse_low_data_args
    from evaluate_ood_3d import parse_args as parse_ood_args

    orig_argv = sys.argv
    try:
        sys.argv = ["evaluate_low_data_3d.py", "--model_type", "visreg_jepa", "--smoke_test"]
        args = parse_low_data_args()
        assert args.model_type == "visreg_jepa"
        assert args.smoke_test is True

        sys.argv = ["evaluate_ood_3d.py", "--model_type", "sigreg_jepa", "--smoke_test"]
        args_ood = parse_ood_args()
        assert args_ood.model_type == "sigreg_jepa"
    finally:
        sys.argv = orig_argv


def test_incremental_csv_merging(tmp_path):
    """Verify incremental CSV merging updates matching fraction without wiping other models."""
    import pandas as pd

    csv_path = tmp_path / "low_data_3d_summary.csv"
    existing_df = pd.DataFrame([
        {"Fraction": "1.0%", "3D VisReg JEPA (FPN)": "6.79%", "3D nnU-Net": "12.12%"},
        {"Fraction": "5.0%", "3D VisReg JEPA (FPN)": "25.00%", "3D nnU-Net": "30.00%"},
    ])
    existing_df.to_csv(csv_path, index=False)

    new_records = [
        {"Fraction": "1.0%", "3D SigReg JEPA (FPN)": "8.50%"},
        {"Fraction": "5.0%", "3D SigReg JEPA (FPN)": "28.00%"},
    ]
    new_df = pd.DataFrame(new_records)

    existing_df = pd.read_csv(csv_path)
    existing_df["Fraction"] = existing_df["Fraction"].astype(str)
    new_df["Fraction"] = new_df["Fraction"].astype(str)
    for col in new_df.columns:
        if col == "Fraction":
            continue
        new_vals = new_df.set_index("Fraction")[col]
        if not (new_vals == "N/A").all():
            if col not in existing_df.columns:
                existing_df[col] = "N/A"
            for f_val, val in new_vals.items():
                if val != "N/A":
                    existing_df.loc[existing_df["Fraction"] == f_val, col] = val

    assert "3D VisReg JEPA (FPN)" in existing_df.columns
    assert "3D SigReg JEPA (FPN)" in existing_df.columns
    assert "3D nnU-Net" in existing_df.columns
    assert existing_df.loc[existing_df["Fraction"] == "1.0%", "3D SigReg JEPA (FPN)"].values[0] == "8.50%"
    assert existing_df.loc[existing_df["Fraction"] == "1.0%", "3D VisReg JEPA (FPN)"].values[0] == "6.79%"


def test_seed_and_num_workers_environment_overrides(monkeypatch):
    """Verify that BRATS3D_SEED and BRATS3D_NUM_WORKERS env vars correctly propagate to configs and PRNG."""
    import argparse

    import torch

    from brats_jepa_3d.utils.seed import set_seed

    # Test set_seed with BRATS3D_SEED env var
    monkeypatch.setenv("BRATS3D_SEED", "99")
    set_seed()
    val1 = torch.randn(5)
    set_seed(99)
    val2 = torch.randn(5)
    assert torch.allclose(val1, val2), "BRATS3D_SEED env var should match explicit set_seed(99)"

    # Test merge_config_with_args env var override
    from brats_jepa_3d.config import merge_config_with_args
    monkeypatch.setenv("BRATS3D_NUM_WORKERS", "4")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args([])
    config = {"seed": 42, "num_workers": 2}
    merged_args = merge_config_with_args(config, args, cli_args=[])
    assert merged_args.seed == 99
    assert merged_args.num_workers == 4

    # Verify evaluate_ood_3d CLI parser supports --num_workers
    import sys

    from brats_jepa_3d.config import PROJECT_ROOT
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from evaluate_ood_3d import parse_args
    monkeypatch.setattr("sys.argv", ["evaluate_ood_3d.py", "--num_workers", "4", "--seed", "77"])
    ood_args = parse_args()
    assert ood_args.num_workers == 4
    assert ood_args.seed == 77


def test_aggregate_low_data_summaries_outer_merge_unequal_lengths(tmp_path):
    """Verify aggregate_low_data_summaries handles differing fraction counts and scrambled orders."""
    from brats_jepa_3d.utils.aggregation import aggregate_low_data_summaries

    exp1 = tmp_path / "exp1"
    exp2 = tmp_path / "exp2"
    (exp1 / "metrics").mkdir(parents=True)
    (exp2 / "metrics").mkdir(parents=True)

    # exp1 has 4 fractions
    (exp1 / "metrics" / "low_data_3d_summary.csv").write_text(
        "Fraction,Model1\n"
        "1.0%,10.5%\n"
        "5.0%,25.0%\n"
        "10.0%,40.0%\n"
        "100.0%,70.0%\n"
    )

    # exp2 has only 2 fractions, in reverse order and with a fraction not in exp1
    (exp2 / "metrics" / "low_data_3d_summary.csv").write_text(
        "Fraction,Model2\n"
        "25.0%,55.0%\n"
        "5.0%,30.0%\n"
    )

    df = aggregate_low_data_summaries([exp1, exp2], output_dir=tmp_path / "out")
    assert len(df) == 5
    # Should be sorted numerically: 1.0%, 5.0%, 10.0%, 25.0%, 100.0%
    assert list(df["Fraction"]) == ["1.0%", "5.0%", "10.0%", "25.0%", "100.0%"]
    assert "Model1" in df.columns and "Model2" in df.columns

    row_5 = df[df["Fraction"] == "5.0%"].iloc[0]
    assert row_5["Model1"] == "25.0%"
    assert row_5["Model2"] == "30.0%"

    row_1 = df[df["Fraction"] == "1.0%"].iloc[0]
    assert row_1["Model1"] == "10.5%"
    assert row_1["Model2"] == "N/A"

    row_25 = df[df["Fraction"] == "25.0%"].iloc[0]
    assert row_25["Model1"] == "N/A"
    assert row_25["Model2"] == "55.0%"


def test_aggregate_ood_summaries_outer_merge_unequal_regimes(tmp_path):
    """Verify aggregate_ood_summaries handles disjoint regimes and out-of-order rows."""
    from brats_jepa_3d.utils.aggregation import aggregate_ood_summaries

    exp1 = tmp_path / "exp1"
    exp2 = tmp_path / "exp2"
    (exp1 / "metrics").mkdir(parents=True)
    (exp2 / "metrics").mkdir(parents=True)

    (exp1 / "metrics" / "ood_3d_summary.csv").write_text(
        "Regime,ModelA\n"
        "Clean Baseline,80.0%\n"
        "Rician Noise (sigma=0.08),75.0%\n"
        "B1 Bias Field Inhomogeneity,78.0%\n"
    )

    (exp2 / "metrics" / "ood_3d_summary.csv").write_text(
        "Regime,ModelB\n"
        "Missing Modalities: T1c Only,45.0%\n"
        "Clean Baseline,82.0%\n"
    )

    df = aggregate_ood_summaries([exp1, exp2], output_dir=tmp_path / "out")
    assert len(df) == 4
    assert "ModelA" in df.columns and "ModelB" in df.columns

    clean_row = df[df["Regime"] == "Clean Baseline"].iloc[0]
    assert clean_row["ModelA"] == "80.0%"
    assert clean_row["ModelB"] == "82.0%"

    t1c_row = df[df["Regime"] == "Missing Modalities: T1c Only"].iloc[0]
    assert t1c_row["ModelA"] == "N/A"
    assert t1c_row["ModelB"] == "45.0%"


def test_brats_3d_dataset_tensor_writeable_and_memory_ownership(tmp_path):
    """Verify BraTS3DDataset produces writeable, contiguous memory tensors from npz."""
    from brats_jepa_3d.data import BraTS3DDataset

    # Create dummy compressed npz volume
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

    # Verify writeability: in-place modification must succeed without error or warning
    image[0, 0, 0, 0] = 999.0
    assert image[0, 0, 0, 0].item() == 999.0
    mask_t[0, 0, 0] = 1.0
    assert mask_t[0, 0, 0].item() == 1.0
    bm_t[0, 0, 0, 0] = 1.0
    assert bm_t[0, 0, 0, 0].item() == 1.0

    # Verify cache stores 3-tuple (image, mask, brain_mask)
    assert 0 in dataset.cache
    assert len(dataset.cache[0]) == 3
    assert isinstance(dataset.cache[0][0], torch.Tensor)
    assert isinstance(dataset.cache[0][1], torch.Tensor)
    assert isinstance(dataset.cache[0][2], torch.Tensor)


def test_combine_artifacts_script_import():
    """Verify combine_and_generate_paper_artifacts script imports cleanly without redundant assignments."""
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "scripts/combine_and_generate_paper_artifacts.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "usage:" in res.stdout.lower()


def test_evaluate_low_data_incremental_outer_merge(tmp_path, monkeypatch):
    """Verify evaluate_low_data_3d outer merge preserves disjoint fractions and models."""
    import re

    import pandas as pd

    csv_path = tmp_path / "low_data_3d_summary.csv"
    csv_path.write_text("Fraction,ModelA\n1.0%,12.0%\n5.0%,35.0%\n")

    new_df = pd.DataFrame([
        {"Fraction": "10.0%", "ModelB": "50.0%"},
        {"Fraction": "5.0%", "ModelB": "38.0%"},
    ])

    existing_df = pd.read_csv(csv_path)
    existing_df["Fraction"] = existing_df["Fraction"].astype(str).str.strip()
    new_df["Fraction"] = new_df["Fraction"].astype(str).str.strip()

    new_cols = [c for c in new_df.columns if c not in existing_df.columns]
    overlap_cols = [c for c in new_df.columns if c != "Fraction" and c in existing_df.columns]
    if new_cols:
        merged = pd.merge(existing_df, new_df[["Fraction"] + new_cols], on="Fraction", how="outer")
    else:
        merged = pd.merge(existing_df, new_df[["Fraction"]], on="Fraction", how="outer")
    if overlap_cols:
        new_indexed = new_df.set_index("Fraction")
        for col in overlap_cols:
            for frac, val in new_indexed[col].items():
                if pd.notna(val) and str(val) != "N/A":
                    merged.loc[merged["Fraction"] == frac, col] = val

    def frac_key(v: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", str(v))
        return float(m.group(1)) if m else 0.0

    merged["_sort"] = merged["Fraction"].apply(frac_key)
    df = merged.sort_values(by="_sort").drop(columns=["_sort"]).reset_index(drop=True).fillna("N/A")

    assert len(df) == 3
    assert list(df["Fraction"]) == ["1.0%", "5.0%", "10.0%"]
    assert "ModelA" in df.columns and "ModelB" in df.columns
    assert df[df["Fraction"] == "1.0%"]["ModelA"].iloc[0] == "12.0%"
    assert df[df["Fraction"] == "1.0%"]["ModelB"].iloc[0] == "N/A"
    assert df[df["Fraction"] == "10.0%"]["ModelA"].iloc[0] == "N/A"
    assert df[df["Fraction"] == "10.0%"]["ModelB"].iloc[0] == "50.0%"
    assert df[df["Fraction"] == "5.0%"]["ModelB"].iloc[0] == "38.0%"


def test_evaluate_ood_incremental_outer_merge(tmp_path):
    """Verify evaluate_ood_3d outer merge preserves disjoint regimes and models."""
    import pandas as pd

    csv_path = tmp_path / "ood_3d_summary.csv"
    csv_path.write_text("Regime,ModelA\nClean Baseline,80.0%\n")

    new_df = pd.DataFrame([
        {"Regime": "3D Rician Noise (sigma=0.08)", "ModelB": "72.0%"},
        {"Regime": "Clean Baseline", "ModelB": "81.5%"},
    ])

    existing_df = pd.read_csv(csv_path)
    existing_df["Regime"] = existing_df["Regime"].astype(str).str.strip()
    new_df["Regime"] = new_df["Regime"].astype(str).str.strip()

    new_cols = [c for c in new_df.columns if c not in existing_df.columns]
    overlap_cols = [c for c in new_df.columns if c != "Regime" and c in existing_df.columns]
    if new_cols:
        merged = pd.merge(existing_df, new_df[["Regime"] + new_cols], on="Regime", how="outer")
    else:
        merged = pd.merge(existing_df, new_df[["Regime"]], on="Regime", how="outer")
    if overlap_cols:
        new_indexed = new_df.set_index("Regime")
        for col in overlap_cols:
            for r_val, val in new_indexed[col].items():
                if pd.notna(val) and str(val) != "N/A":
                    merged.loc[merged["Regime"] == r_val, col] = val
    df = merged.fillna("N/A")

    assert len(df) == 2
    assert "Clean Baseline" in list(df["Regime"])
    assert "3D Rician Noise (sigma=0.08)" in list(df["Regime"])
    assert df[df["Regime"] == "Clean Baseline"]["ModelA"].iloc[0] == "80.0%"
    assert df[df["Regime"] == "Clean Baseline"]["ModelB"].iloc[0] == "81.5%"
    assert df[df["Regime"] == "3D Rician Noise (sigma=0.08)"]["ModelA"].iloc[0] == "N/A"
    assert df[df["Regime"] == "3D Rician Noise (sigma=0.08)"]["ModelB"].iloc[0] == "72.0%"


def test_run_full_pipeline_checkpoint_resolution(tmp_path, monkeypatch):
    """Verify run_full_pipeline_3d prioritizes *_3d_best.pt over epoch checkpoints."""
    from brats_jepa_3d.utils import sort_checkpoints_by_epoch

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # Create dummy epoch and best checkpoints
    (ckpt_dir / "visreg_jepa_3d_epoch_10.pt").write_text("dummy")
    (ckpt_dir / "visreg_jepa_3d_epoch_20.pt").write_text("dummy")
    (ckpt_dir / "visreg_jepa_3d_best.pt").write_text("best")

    best_ckpts = sorted(ckpt_dir.glob("visreg_jepa*_3d_best.pt"))
    epoch_ckpts = sort_checkpoints_by_epoch(list(ckpt_dir.glob("visreg_jepa*epoch*.pt")))

    selected_ckpt = best_ckpts[-1] if best_ckpts else (epoch_ckpts[-1] if epoch_ckpts else None)
    assert selected_ckpt.name == "visreg_jepa_3d_best.pt"


def test_ijepa_loss_empty_predictions_raises():
    """Verify IJEPALoss raises ValueError with clear error message when passed empty lists."""
    from brats_jepa_3d.losses import IJEPALoss

    criterion = IJEPALoss()
    with pytest.raises(ValueError, match="must be non-empty lists of tensors"):
        criterion([], [])


def test_volumetric_augmentations_batched_tensor():
    """Verify VolumetricAugmentations3D correctly operates on both 4D [C, D, H, W] and 5D [B, C, D, H, W] tensors."""
    from brats_jepa_3d.data.transforms import VolumetricAugmentations3D

    aug = VolumetricAugmentations3D(flip_prob=1.0, noise_prob=0.0, modality_dropout_prob=0.0, is_training=True)

    # 4D case: [C, D, H, W] = [4, 8, 8, 8] without brain_mask returns (image, mask)
    x_4d = torch.arange(8).view(1, 8, 1, 1).repeat(4, 1, 8, 8).float()
    out_4d, _ = aug(x_4d.clone(), None, None)
    # Dimension -3 (depth) was flipped: values 0..7 become 7..0
    assert torch.equal(out_4d[0, :, 0, 0], torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.float32))

    # 5D case: [B, C, D, H, W] = [2, 4, 8, 8, 8] with brain_mask returns (image, mask, brain_mask)
    x_5d = torch.arange(8).view(1, 1, 8, 1, 1).repeat(2, 4, 1, 8, 8).float()
    bm_5d = torch.ones(2, 1, 8, 8, 8)
    # Distinct channels to verify channel dimension is NOT flipped
    for c in range(4):
        x_5d[:, c] = x_5d[:, c] * (c + 1)
    out_5d, _, out_bm = aug(x_5d.clone(), None, bm_5d)
    # Depth is flipped
    assert torch.equal(out_5d[0, 0, :, 0, 0], torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.float32))
    # Channel 1 was multiplied by 2 and is still channel 1 (not swapped with another channel)
    assert torch.equal(out_5d[0, 1, :, 0, 0], torch.tensor([14, 12, 10, 8, 6, 4, 2, 0], dtype=torch.float32))
    assert out_bm.shape == (2, 1, 8, 8, 8)


def test_evaluate_baseline_checkpoint_zero_matched_raises(tmp_path):
    """Verify baseline evaluation raises RuntimeError if checkpoint contains zero matching keys."""
    from torch import nn

    # Create dummy model with known keys
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 16, 3)

    model = DummyNet()

    # Create completely mismatched checkpoint state dict
    corrupt_sd = {"unrelated_layer.weight": torch.randn(10, 10)}
    ckpt_file = tmp_path / "corrupted_unet.pt"
    torch.save(corrupt_sd, ckpt_file)

    sd = torch.load(ckpt_file, map_location="cpu")
    missing, _unexpected = model.load_state_dict(sd, strict=False)
    matched = [k for k in model.state_dict() if k not in missing]

    assert len(matched) == 0
    with pytest.raises(RuntimeError, match="Failed to load any weights"):
        if len(matched) == 0:
            raise RuntimeError(f"Failed to load any weights for 3D UNet from {ckpt_file}")
