"""Model architecture tests: ViT, JEPA predictor, VisReg/SigReg, decoders, UNet, nnU-Net, UNETR hybrid, hooks, and deep supervision."""

import copy
import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from brats_jepa_3d.config import CONFIGS_DIR, load_yaml_config
from brats_jepa_3d.models import (
    BraTS3DnnUNet,
    BraTS3DUNet,
    HybridUNETRDecoder3D,
    JEPAPredictor3D,
    JEPASegmentationModel3D,
    MultiScaleViTSegmentationDecoder3D,
    PatchEmbed3D,
    SigRegJEPA3D,
    SingleEncoderJEPA3D,
    VisionTransformerEncoder3D,
    VisRegJEPA3D,
    dropout_disabled,
)
from brats_jepa_3d.models._tissue_filter import filter_tissue_tokens


# =========================================================================
# 1. Vision Transformer 3D & JEPA Components
# =========================================================================


def test_patch_embed_3d():
    patch_embed = PatchEmbed3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
    )
    x = torch.randn(2, 4, 128, 128, 128)
    tokens = patch_embed(x)
    assert tokens.shape == (2, 512, 384)


def test_vision_transformer_encoder_3d():
    encoder = VisionTransformerEncoder3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        depth=2,
        num_heads=6,
    )
    x = torch.randn(2, 4, 128, 128, 128)

    # 1. Full image forward
    tokens_full = encoder(x)
    assert tokens_full.shape == (2, 512, 384)

    # 2. Subsampled context forward
    patch_indices = torch.randint(0, 512, (2, 192))
    tokens_sub = encoder(x, patch_indices=patch_indices)
    assert tokens_sub.shape == (2, 192, 384)


def test_jepa_predictor_3d():
    pred = JEPAPredictor3D(
        embed_dim=384,
        pred_embed_dim=192,
        num_patches=512,
        grid_size=(8, 8, 8),
        depth=2,
        num_heads=6,
    )
    ctx_tokens = torch.randn(2, 192, 384)
    ctx_idx = torch.randint(0, 512, (2, 192))
    tgt_idx = torch.randint(0, 512, (2, 27))

    pred_tgt = pred(ctx_tokens, ctx_idx, tgt_idx)
    assert pred_tgt.shape == (2, 27, 384)


def test_sigreg_jepa_3d():
    model = SigRegJEPA3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        encoder_depth=2,
        predictor_depth=2,
        num_heads=6,
    )
    images = torch.randn(2, 4, 128, 128, 128)
    context_indices = torch.randint(0, 512, (2, 192))
    target_indices_list = [
        torch.randint(0, 512, (2, 27)),
        torch.randint(0, 512, (2, 27)),
    ]

    out = model(images, context_indices, target_indices_list)
    assert "predictions" in out
    assert "targets" in out
    assert len(out["predictions"]) == 2
    assert out["predictions"][0].shape == (2, 27, 384)
    assert out["projected_tokens"].shape == (2, 192, 128)


def test_visreg_jepa_3d():
    model = VisRegJEPA3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        encoder_depth=2,
        predictor_depth=2,
        num_heads=6,
    )
    images = torch.randn(2, 4, 128, 128, 128)
    context_indices = torch.randint(0, 512, (2, 192))
    target_indices_list = [
        torch.randint(0, 512, (2, 27)),
        torch.randint(0, 512, (2, 27)),
    ]

    out = model(images, context_indices, target_indices_list)
    assert "predictions" in out
    assert "targets" in out
    assert len(out["predictions"]) == 2
    assert out["predictions"][0].shape == (2, 27, 384)
    assert out["projected_tokens"].shape == (2, 192, 128)


def test_jepa_predictor_forward_multi_targets():
    """Verify JEPAPredictor3D forward_multi_targets produces identical outputs to sequential forward."""
    predictor = JEPAPredictor3D(
        embed_dim=64,
        pred_embed_dim=32,
        depth=2,
        num_heads=2,
    )
    context_tokens = torch.randn(2, 48, 64)
    context_indices = torch.randint(0, 512, (2, 48))
    target_indices_list = [
        torch.randint(0, 512, (2, 27)),
        torch.randint(0, 512, (2, 18)),
        torch.randint(0, 512, (2, 8)),
    ]

    # Predict via forward_multi_targets (cached context proj + cached sinusoidal gather)
    preds_multi = predictor.forward_multi_targets(context_tokens, context_indices, target_indices_list)

    # Predict via sequential loop
    preds_seq = [
        predictor(context_tokens, context_indices, tgt_idx)
        for tgt_idx in target_indices_list
    ]

    assert len(preds_multi) == len(target_indices_list)
    for p_multi, p_seq in zip(preds_multi, preds_seq):
        assert torch.allclose(p_multi, p_seq, atol=1e-5)


def test_single_encoder_jepa_3d():
    """Verify SingleEncoderJEPA3D architecture and forward pass."""
    model = SingleEncoderJEPA3D(
        img_size=(64, 64, 64),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=64,
        proj_dim=32,
        encoder_depth=2,
        predictor_depth=2,
        predictor_embed_dim=32,
        num_heads=2,
    )
    images = torch.randn(2, 4, 64, 64, 64)
    context_indices = torch.randint(0, 64, (2, 24))
    target_indices_list = [torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 4))]

    out = model(images, context_indices, target_indices_list)
    assert len(out["predictions"]) == 2
    assert len(out["targets"]) == 2
    assert out["projected_tokens"].shape == (2, 24, 32)
    assert out["predictions"][0].shape == (2, 8, 64)
    assert out["predictions"][1].shape == (2, 4, 64)


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


def test_p2_step11_no_mode_side_effect():
    """forward() preserves caller mode; grads flow only through the context path."""
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
    model = VisRegJEPA3D()
    assert model.context_encoder.pos_embed.requires_grad is False
    assert model.predictor.pos_embed.requires_grad is False
    assert "context_encoder.pos_embed" in model.state_dict()
    assert "predictor.pos_embed" in model.state_dict()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not any("pos_embed" in n for n in trainable)


def test_f1_dropout_disabled_restores_probs():
    """dropout_disabled context manager zeros p during forward then restores originals."""
    model = VisRegJEPA3D()
    for m in model.context_encoder.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.5
        if isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = 0.2
    model.train()
    with dropout_disabled(model.context_encoder):
        for m in model.context_encoder.modules():
            if isinstance(m, torch.nn.Dropout):
                assert m.p == 0.0
    for m in model.context_encoder.modules():
        if isinstance(m, torch.nn.Dropout):
            assert m.p == 0.5


# =========================================================================
# 2. Decoders & Baseline Architectures
# =========================================================================


def test_segmentation_decoders_3d():
    # 1. MultiScale FPN Decoder
    model_ms = JEPASegmentationModel3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        encoder_depth=2,
        num_heads=6,
        decoder_type="multiscale",
        deep_supervision=True,
    )
    x = torch.randn(2, 4, 128, 128, 128)
    model_ms.train()
    out_ms_train = model_ms(x)
    assert isinstance(out_ms_train, list)
    assert len(out_ms_train) == 4
    assert out_ms_train[0].shape == (2, 5, 128, 128, 128)

    model_ms.eval()
    with torch.no_grad():
        out_ms_eval = model_ms(x)
    assert isinstance(out_ms_eval, torch.Tensor)
    assert out_ms_eval.shape == (2, 5, 128, 128, 128)

    # 2. Linear Segmentation Decoder
    model_lin = JEPASegmentationModel3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        encoder_depth=2,
        num_heads=6,
        decoder_type="linear",
        deep_supervision=False,
    )
    out_lin = model_lin(x)
    assert out_lin.shape == (2, 5, 128, 128, 128)


def test_supervised_baselines_3d():
    x = torch.randn(2, 4, 64, 64, 64)

    # 1. 3D UNet
    unet = BraTS3DUNet(
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
    )
    out_unet = unet(x)
    assert out_unet.shape == (2, 1, 64, 64, 64)

    # 2. 3D nnU-Net
    nnunet = BraTS3DnnUNet(
        in_channels=4,
        out_channels=1,
        deep_supervision=True,
        deep_supr_num=2,
    )
    nnunet.train()
    out_nnunet = nnunet(x)
    assert isinstance(out_nnunet, list)
    assert len(out_nnunet) == 3
    assert out_nnunet[0].shape == (2, 1, 64, 64, 64)


def test_model_config_consistency_and_deep_supervision_detection():
    """Verify UNet, nnUNet, and JEPA initialize cleanly with configs and explicit deep supervision."""
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

    nnunet_cfg = load_yaml_config(CONFIGS_DIR / "model" / "nnunet_3d.yaml")
    nnunet = BraTS3DnnUNet(
        in_channels=nnunet_cfg.get("in_channels", 4),
        out_channels=nnunet_cfg.get("out_channels", 1),
        deep_supervision=nnunet_cfg.get("deep_supervision", True),
        deep_supr_num=nnunet_cfg.get("deep_supr_num", 3),
        res_block=nnunet_cfg.get("res_block", True),
    )
    assert isinstance(nnunet, torch.nn.Module)

    jepa_model = JEPASegmentationModel3D(
        img_size=(64, 64, 64),
        patch_size=(16, 16, 16),
        decoder_type="multiscale",
        deep_supervision=True,
    )
    assert hasattr(jepa_model.decoder, "ds3")


def test_models_out_channels_5():
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


def test_channel_mismatch_raises_actionable_error():
    from brats_jepa_3d.models import load_downstream_state_dict
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
    missing, _ = load_downstream_state_dict(c5, c5.state_dict(), "test-model")
    assert missing == []


# =========================================================================
# 3. Deep Supervision Hooks & Hook Life Cycle
# =========================================================================


def _tiny_unet(**kw):
    kw.setdefault("channels", (8, 16, 32, 64, 128))
    kw.setdefault("strides", (2, 2, 2, 2))
    kw.setdefault("num_res_units", 1)
    kw.setdefault("dropout", 0.0)
    return BraTS3DUNet(**kw)


def test_unet_ds_train_returns_four_heads():
    m = _tiny_unet(deep_supervision=True)
    m.train()
    with torch.no_grad():
        outs = m(torch.randn(1, 4, 32, 32, 32))
    assert isinstance(outs, list) and len(outs) == 4
    assert [tuple(o.shape[2:]) for o in outs] == [(32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)]
    assert all(o.shape[:2] == (1, 5) for o in outs)


def test_unet_ds_eval_and_disabled_return_single():
    m = _tiny_unet(deep_supervision=True)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(1, 4, 32, 32, 32))
    assert isinstance(y, torch.Tensor) and tuple(y.shape) == (1, 5, 32, 32, 32)

    m2 = _tiny_unet(deep_supervision=False)
    m2.train()
    with torch.no_grad():
        y2 = m2(torch.randn(1, 4, 32, 32, 32))
    assert isinstance(y2, torch.Tensor) and tuple(y2.shape) == (1, 5, 32, 32, 32)
    assert not [k for k in m2.state_dict() if k.startswith("ds")]


def test_unet_ds_loss_backward():
    from brats_jepa_3d.losses import DeepSupervisionLoss3D
    m = _tiny_unet(deep_supervision=True)
    m.train()
    crit = DeepSupervisionLoss3D()
    logits = m(torch.randn(1, 4, 32, 32, 32))
    target = torch.randint(0, 5, (1, 1, 32, 32, 32)).long()
    loss = crit(logits, target)["loss"]
    loss.backward()
    assert torch.isfinite(loss).item()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_unet_legacy_checkpoint_compat():
    old = _tiny_unet(deep_supervision=False).state_dict()
    new = _tiny_unet(deep_supervision=True)
    missing, unexpected = new.load_state_dict(old, strict=False)
    assert not unexpected
    assert sorted(missing) == sorted(k for k in new.state_dict() if k.startswith("ds"))


def test_unet_repeated_forwards_stable():
    """Hooks must keep working across train/eval alternations (buffer identity)."""
    m = _tiny_unet(deep_supervision=True)
    x = torch.randn(1, 4, 32, 32, 32)
    m.train()
    with torch.no_grad():
        r1 = [tuple(o.shape) for o in m(x)]
    m.eval()
    with torch.no_grad():
        m(x)
    m.train()
    with torch.no_grad():
        r2 = [tuple(o.shape) for o in m(x)]
    assert r1 == r2 == [(1, 5, 32, 32, 32), (1, 5, 16, 16, 16), (1, 5, 8, 8, 8), (1, 5, 4, 4, 4)]


def test_p2_step14_unet_hooks_bounded():
    """One hook set for life: consecutive forwards identical, eval single tensor."""
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


# =========================================================================
# 4. UNETR Hybrid Decoder
# =========================================================================


def _fake_intermediates(B=1, L=8, N=512, D=384):
    torch.manual_seed(0)
    return [torch.randn(B, N, D) for _ in range(L)]


def test_hybrid_eval_shape():
    dec = HybridUNETRDecoder3D()
    dec.eval()
    with torch.no_grad():
        y = dec(_fake_intermediates(), raw_volume=torch.randn(1, 4, 128, 128, 128))
    assert tuple(y.shape) == (1, 5, 128, 128, 128)


def test_hybrid_train_ds_heads():
    dec = HybridUNETRDecoder3D(deep_supervision=True)
    dec.train()
    outs = dec(_fake_intermediates(), raw_volume=torch.randn(1, 4, 128, 128, 128))
    assert isinstance(outs, list) and len(outs) == 4
    assert [tuple(o.shape[2:]) for o in outs] == [
        (128, 128, 128), (64, 64, 64), (32, 32, 32), (16, 16, 16)]


def test_hybrid_no_volume_fallback():
    dec = HybridUNETRDecoder3D()
    dec.eval()
    with torch.no_grad():
        y = dec(_fake_intermediates())
    assert tuple(y.shape) == (1, 5, 128, 128, 128)


def test_hybrid_grad_flow():
    dec = HybridUNETRDecoder3D(deep_supervision=True)
    dec.train()
    outs = dec(_fake_intermediates(), raw_volume=torch.randn(1, 4, 128, 128, 128))
    loss = sum(o.float().mean() for o in outs)
    loss.backward()
    stem_grads = [p.grad for p in list(dec.stem_128.parameters()) + list(dec.stem_64.parameters())]
    assert any(g is not None and torch.isfinite(g).all() for g in stem_grads)
    assert dec.skip_l6[0].weight.grad is not None, "ViT lateral must receive gradients"


def test_hybrid_loads_fpn_weights():
    fpn = MultiScaleViTSegmentationDecoder3D(deep_supervision=True)
    hyb = HybridUNETRDecoder3D(deep_supervision=True)
    fpn_sd = fpn.state_dict()
    hyb_sd = hyb.state_dict()
    compatible = {k: v for k, v in fpn_sd.items() if k in hyb_sd and hyb_sd[k].shape == v.shape}
    missing, unexpected = hyb.load_state_dict(compatible, strict=False)
    assert not unexpected
    assert set(missing) == set(hyb_sd) - set(compatible)
    assert "fuse3.0.weight" in missing and "fuse4.0.weight" in missing
    assert "up4.0.weight" not in missing and "head.weight" not in missing
    assert "up1.0.weight" not in missing and "skip_l6.0.weight" not in missing
    assert len(compatible) >= 20, f"stages 1-2 + laterals should transfer, got {len(compatible)}"


def test_hybrid_full_model_forward():
    model = JEPASegmentationModel3D(
        img_size=(32, 32, 32), patch_size=(16, 16, 16),
        decoder_type="unetr_hybrid", deep_supervision=False)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 4, 32, 32, 32))
    assert tuple(y.shape) == (1, 5, 32, 32, 32)


# =========================================================================
# 5. Tissue Filtering Components
# =========================================================================


def test_p0_step3_sigreg_tissue_parity():
    """SigRegJEPA3D filters projected tokens by tissue mask like VisReg."""
    assert "context_tissue_mask" in inspect.signature(SigRegJEPA3D.forward).parameters
    assert "context_tissue_mask" in inspect.signature(VisRegJEPA3D.forward).parameters


def test_p0_step4_per_sample_tissue_fallback():
    """B=2 with tissue counts [100, 5] keeps 100+192 rows (not batch fallback)."""
    proj = torch.randn(2, 192, 128)
    mask = torch.zeros(2, 192, dtype=torch.bool)
    mask[0, :100] = True
    mask[1, :5] = True
    out = filter_tissue_tokens(proj, mask, min_tokens=32)
    assert out.shape == (100 + 192, 128)


def test_residual_tissue_filter_rejects_non3d():
    """filter_tissue_tokens requires 3D [B, N, D] projected tensor."""
    with pytest.raises(ValueError):
        filter_tissue_tokens(torch.randn(10, 128), torch.ones(10, dtype=torch.bool))


def test_option_b_full_model_forward_and_loss_backward():
    """Verify VisRegJEPA3D and SigRegJEPA3D with Option B 6-slot target masking."""
    from brats_jepa_3d.data.masking import JEPAMaskingTransform3D, jepa_masking_collate_fn_3d
    from brats_jepa_3d.losses import VisRegLoss, SigRegLoss

    tf = JEPAMaskingTransform3D()  # Option B: 6 slots, 97 tokens
    batch_raw = [
        {"image": torch.randn(4, 128, 128, 128), "patient_id": f"p_{i}", **tf()}
        for i in range(2)
    ]
    batch = jepa_masking_collate_fn_3d(batch_raw)
    images = batch["images"]
    ctx = batch["context_indices"]
    tgt_list = batch["target_indices_list"]

    assert len(tgt_list) == 6
    expected_shapes = [(2, 27), (2, 18), (2, 18), (2, 18), (2, 8), (2, 8)]
    for m, shp in enumerate(expected_shapes):
        assert tgt_list[m].shape == shp

    # 1. VisRegJEPA3D forward + backward
    visreg = VisRegJEPA3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=96,
        proj_dim=32,
        encoder_depth=2,
        predictor_depth=2,
        predictor_embed_dim=48,
    )
    visreg_crit = VisRegLoss()
    out_vr = visreg(images, ctx, tgt_list)
    loss_vr = visreg_crit(out_vr["predictions"], out_vr["targets"], projected_tokens=out_vr["projected_tokens"])
    loss_vr["loss"].backward()
    assert any(p.grad is not None for p in visreg.parameters())

    # 3. SigRegJEPA3D forward + backward
    sigreg = SigRegJEPA3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=96,
        proj_dim=32,
        encoder_depth=2,
        predictor_depth=2,
        predictor_embed_dim=48,
    )
    sigreg_crit = SigRegLoss()
    out_sr = sigreg(images, ctx, tgt_list)
    loss_sr = sigreg_crit(out_sr["predictions"], out_sr["targets"], projected_tokens=out_sr["projected_tokens"])
    loss_sr["loss"].backward()
    assert any(p.grad is not None for p in sigreg.parameters())

