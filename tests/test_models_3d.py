import torch

from brats_jepa_3d.models import (
    IJEPA3D,
    BraTS3DnnUNet,
    BraTS3DUNet,
    JEPAPredictor3D,
    JEPASegmentationModel3D,
    MultiScaleViTSegmentationDecoder3D,
    PatchEmbed3D,
    SigRegJEPA3D,
    VisionTransformerEncoder3D,
    VisRegJEPA3D,
    ViTSegmentationDecoder3D,
)


def test_patch_embed_3d(sample_volume_3d):
    pe = PatchEmbed3D(
        img_size=(128, 128, 128), patch_size=(16, 16, 16), in_channels=4, embed_dim=384
    )
    tokens = pe(sample_volume_3d)
    assert tokens.shape == (2, 512, 384)


def test_vision_transformer_encoder_3d(sample_volume_3d):
    enc = VisionTransformerEncoder3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        depth=4,
        num_heads=6,
    )

    # 1. Full unmasked forward pass
    out_full = enc(sample_volume_3d)
    assert out_full.shape == (2, 512, 384)

    # 2. Context-selective pass (zero-attention leakage)
    ctx_indices = torch.randint(0, 512, (2, 192))
    out_ctx = enc(sample_volume_3d, patch_indices=ctx_indices)
    assert out_ctx.shape == (2, 192, 384)

    # 3. Intermediate tokens extraction
    out, intermediates = enc(sample_volume_3d, return_intermediate=True)
    assert out.shape == (2, 512, 384)
    assert len(intermediates) == 4
    assert intermediates[0].shape == (2, 512, 384)


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


def test_sigreg_jepa_3d(sample_volume_3d):
    model = SigRegJEPA3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        proj_dim=128,
        encoder_depth=2,
        predictor_depth=2,
    )
    ctx_idx = torch.randint(0, 512, (2, 192))
    tgt_idx_list = [torch.randint(0, 512, (2, 27)) for _ in range(4)]

    out = model(sample_volume_3d, ctx_idx, tgt_idx_list)

    assert len(out["predictions"]) == 4
    assert len(out["targets"]) == 4
    assert out["predictions"][0].shape == (2, 27, 384)
    assert out["targets"][0].shape == (2, 27, 384)
    assert out["projected_tokens"].shape == (2, 192, 128)

    # Backward gradient check
    loss = out["predictions"][0].sum() + out["projected_tokens"].sum()
    loss.backward()
    assert model.context_encoder.patch_embed.proj.weight.grad is not None


def test_visreg_jepa_3d(sample_volume_3d):
    model = VisRegJEPA3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        proj_dim=128,
        encoder_depth=2,
        predictor_depth=2,
    )
    ctx_idx = torch.randint(0, 512, (2, 192))
    tgt_idx_list = [torch.randint(0, 512, (2, 27)) for _ in range(2)]

    out = model(sample_volume_3d, ctx_idx, tgt_idx_list)
    assert out["projected_tokens"].shape == (2, 192, 128)


def test_ijepa_3d_and_ema(sample_volume_3d):
    model = IJEPA3D(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_channels=4,
        embed_dim=384,
        encoder_depth=2,
        predictor_depth=2,
        ema_momentum=0.99,
    )
    ctx_idx = torch.randint(0, 512, (2, 192))
    tgt_idx_list = [torch.randint(0, 512, (2, 27))]

    out = model(sample_volume_3d, ctx_idx, tgt_idx_list)
    assert out["predictions"][0].shape == (2, 27, 384)

    # Verify EMA update modifies target encoder weights
    orig_weight = model.target_encoder.patch_embed.proj.weight.clone()
    with torch.no_grad():
        model.context_encoder.patch_embed.proj.weight.add_(1.0)
    model.update_target_encoder()
    new_weight = model.target_encoder.patch_embed.proj.weight
    assert not torch.equal(orig_weight, new_weight)


def test_segmentation_decoders_3d():
    tokens = torch.randn(2, 512, 384)
    bot_dec = ViTSegmentationDecoder3D(in_dim=384, out_channels=1)
    logits_bot = bot_dec(tokens)
    assert logits_bot.shape == (2, 1, 128, 128, 128)

    intermediates = [torch.randn(2, 512, 384) for _ in range(4)]
    fpn_dec = MultiScaleViTSegmentationDecoder3D(in_dim=384, out_channels=1)
    logits_fpn = fpn_dec(intermediates)
    assert logits_fpn.shape == (2, 1, 128, 128, 128)

    seg_model = JEPASegmentationModel3D(encoder_depth=2, decoder_type="multiscale")
    logits_seg = seg_model(torch.randn(2, 4, 128, 128, 128))
    assert logits_seg.shape == (2, 1, 128, 128, 128)


def test_supervised_baselines_3d(sample_volume_3d):
    unet = BraTS3DUNet(channels=(16, 32, 64, 128, 256), num_res_units=1)
    out_unet = unet(sample_volume_3d)
    assert out_unet.shape == (2, 1, 128, 128, 128)

    nnunet = BraTS3DnnUNet(filters=[16, 32, 64, 128, 256], deep_supervision=False)
    out_nnunet = nnunet(sample_volume_3d)
    assert out_nnunet.shape == (2, 1, 128, 128, 128)
