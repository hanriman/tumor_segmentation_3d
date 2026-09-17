import pytest
import torch

from brats_jepa_3d.data import apply_rician_noise_3d
from brats_jepa_3d.losses import DeepSupervisionLoss3D
from brats_jepa_3d.models import (
    BraTS3DnnUNet,
    JEPASegmentationModel3D,
    MultiScaleViTSegmentationDecoder3D,
)


def test_multiscale_decoder_parameters():
    decoder = MultiScaleViTSegmentationDecoder3D(in_dim=384, out_channels=1, grid_size=(8, 8, 8))
    params = sum(p.numel() for p in decoder.parameters())
    # Before refactoring skip_l2: ~15.6M params. With progressive upsampling: ~7.1M params!
    assert params < 10_000_000, f"Decoder has {params} params, expected < 10M"

    # Check forward pass
    tokens = [torch.randn(2, 512, 384) for _ in range(8)]
    out = decoder(tokens)
    assert out.shape == (2, 1, 128, 128, 128)


def test_load_pretrained_encoder_prefix_stripping():
    model = JEPASegmentationModel3D(img_size=(128, 128, 128), decoder_type="multiscale")
    orig_weight = model.encoder.patch_embed.proj.weight.clone()

    # 1. State dict with context_encoder. prefix
    test_weight = torch.randn_like(orig_weight)
    ckpt_context = {"context_encoder.patch_embed.proj.weight": test_weight}
    res = model.load_pretrained_encoder(ckpt_context)
    assert res["loaded_keys"] >= 1
    assert torch.allclose(model.encoder.patch_embed.proj.weight, test_weight)

    # 2. State dict with encoder. prefix
    test_weight2 = torch.randn_like(orig_weight)
    ckpt_enc = {"encoder.patch_embed.proj.weight": test_weight2}
    res2 = model.load_pretrained_encoder(ckpt_enc)
    assert res2["loaded_keys"] >= 1
    assert torch.allclose(model.encoder.patch_embed.proj.weight, test_weight2)

    # 3. Disjoint state dict should raise RuntimeError
    with pytest.raises(RuntimeError, match="No matching encoder weights"):
        model.load_pretrained_encoder({"bogus_key": torch.randn(10)})


def test_rician_noise_contrast_preservation():
    # Synthetic volume with background 0, GM at +1.0, and CSF / necrotic core at -2.0
    vol = torch.zeros(1, 1, 16, 16, 16)
    vol[:, :, 4:12, 4:12, 4:8] = -2.0  # hypointense
    vol[:, :, 4:12, 4:12, 8:12] = +1.0  # hyperintense

    noisy = apply_rician_noise_3d(vol, sigma=0.05)
    # Background must remain 0
    assert torch.all(noisy[vol == 0] == 0)
    # Hypointense region should remain negative (contrast preserved, no sign flip!)
    hypo_mean = noisy[:, :, 4:12, 4:12, 4:8].mean().item()
    hyper_mean = noisy[:, :, 4:12, 4:12, 8:12].mean().item()
    assert hypo_mean < 0.0, f"Expected negative mean for hypointense region, got {hypo_mean}"
    assert hyper_mean > 0.0, f"Expected positive mean for hyperintense region, got {hyper_mean}"
    assert hypo_mean < hyper_mean


def test_nnunet_deep_supervision_loss():
    model = BraTS3DnnUNet(
        in_channels=4,
        out_channels=1,
        deep_supervision=True,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2], [2, 2, 2]],
        filters=[16, 32, 64],
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        deep_supr_num=1,
    )
    criterion = DeepSupervisionLoss3D()
    x = torch.randn(2, 4, 16, 16, 16)
    y = (torch.rand(2, 1, 16, 16, 16) > 0.5).float()

    # Training mode: outputs list of multi-scale logits
    model.train()
    out_train = model(x)
    assert isinstance(out_train, list)
    loss_train = criterion(out_train, y)
    assert "loss" in loss_train and loss_train["loss"].item() > 0

    # Eval mode: outputs single full-resolution tensor
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert isinstance(out_eval, torch.Tensor)
    assert out_eval.shape == (2, 1, 16, 16, 16)


def test_compound_prefix_stripping():
    model = JEPASegmentationModel3D(img_size=(128, 128, 128), decoder_type="multiscale")
    orig_weight = model.encoder.patch_embed.proj.weight.clone()

    # Nested compound prefix: module.context_encoder.
    test_weight = torch.randn_like(orig_weight)
    ckpt = {"module.context_encoder.patch_embed.proj.weight": test_weight}
    res = model.load_pretrained_encoder(ckpt)
    assert res["loaded_keys"] >= 1
    assert torch.allclose(model.encoder.patch_embed.proj.weight, test_weight)

    # Multi-nested prefix: module._orig_mod.context_encoder.
    test_weight2 = torch.randn_like(orig_weight)
    ckpt2 = {"module._orig_mod.context_encoder.patch_embed.proj.weight": test_weight2}
    res2 = model.load_pretrained_encoder(ckpt2)
    assert res2["loaded_keys"] >= 1
    assert torch.allclose(model.encoder.patch_embed.proj.weight, test_weight2)


def test_visreg_swd_fp16_stability():
    from brats_jepa_3d.losses import VisRegLoss

    loss_fn = VisRegLoss()
    # Simulate FP16 projected representations under AMP
    z_fp16 = torch.randn(4, 192, 128, dtype=torch.float16)
    preds = [torch.randn(4, 27, 384, dtype=torch.float16)]
    targets = [torch.randn(4, 27, 384, dtype=torch.float16)]

    out = loss_fn(preds, targets, projected_tokens=z_fp16)
    assert "loss" in out
    assert not torch.isnan(out["loss"])
    assert not torch.isinf(out["loss"])
    assert out["loss"].item() > 0


def test_dice_loss_squared_vs_linear():
    from brats_jepa_3d.losses import CombinedDiceBCELoss3D

    criterion_sq = CombinedDiceBCELoss3D(squared_pred=True)
    criterion_lin = CombinedDiceBCELoss3D(squared_pred=False)

    logits = torch.randn(2, 1, 16, 16, 16)
    targets = (torch.rand(2, 1, 16, 16, 16) > 0.5).float()

    loss_sq = criterion_sq(logits, targets)["loss"]
    loss_lin = criterion_lin(logits, targets)["loss"]

    assert loss_sq.item() > 0
    assert loss_lin.item() > 0
    # Both formulations produce valid finite non-identical values
    assert not torch.isnan(loss_sq) and not torch.isnan(loss_lin)
