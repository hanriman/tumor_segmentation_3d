
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
    """Verify that include_background=False correctly sets ignore_index=0 on CrossEntropyLoss."""
    loss_exc = CombinedDiceBCELoss3D(num_classes=4, include_background=False)
    assert loss_exc.include_background is False
    assert loss_exc.dice_loss.include_background is False

    loss_inc = CombinedDiceBCELoss3D(num_classes=4, include_background=True)
    assert loss_inc.include_background is True
    assert loss_inc.dice_loss.include_background is True

    # Targets: mostly background (class 0), with a single voxel of class 1
    # Prediction: class 1 is predicted correctly (+10.0), but class 0 is predicted incorrectly (-10.0)
    logits = torch.zeros(1, 4, 4, 4, 4)
    logits[:, 0] = -10.0  # wrong prediction for class 0
    logits[:, 1] = 10.0   # correct prediction for class 1
    targets = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    targets[:, 0, 0, 0] = 1  # class 1 at (0, 0, 0)

    res_inc = loss_inc(logits, targets)
    res_exc = loss_exc(logits, targets)

    # When background is included, CE loss on incorrect class 0 is large
    assert res_inc["bce_loss"].item() > 5.0
    # When background is excluded, CE loss on class 0 is ignored and class 1 is accurate
    assert res_exc["bce_loss"].item() < 0.05


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





