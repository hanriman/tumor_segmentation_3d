import torch

from brats_jepa_3d.data import (
    JEPAMaskingTransform3D,
    RandomModalityDropout3D,
    VolumetricAugmentations3D,
    jepa_masking_collate_fn_3d,
)


def test_jepa_masking_transform_3d():
    tf = JEPAMaskingTransform3D(
        grid_size=(8, 8, 8),
        num_target_cuboids=4,
        target_cuboid_size=(3, 3, 3),
        context_num_patches=192,
    )
    res = tf()

    ctx = res["context_indices"]
    targets = res["target_indices_list"]

    # 1. Check exact sequence lengths
    assert isinstance(ctx, torch.Tensor)
    assert ctx.shape == (192,), f"Expected 192 context patches, got {ctx.shape}"
    assert len(targets) == 4
    for m, tgt in enumerate(targets):
        assert isinstance(tgt, torch.Tensor)
        assert tgt.shape == (27,), f"Target block {m} expected 27 patches, got {tgt.shape}"

    # 2. Check strict zero-collision guarantee
    ctx_set = set(ctx.tolist())
    for m, tgt in enumerate(targets):
        tgt_set = set(tgt.tolist())
        collision = ctx_set & tgt_set
        assert len(collision) == 0, (
            f"Collision detected between context and target {m}: {collision}"
        )

    # 3. Check bounds
    assert all(0 <= idx < 512 for idx in ctx.tolist())


def test_jepa_masking_collate_3d():
    tf = JEPAMaskingTransform3D()
    batch = [
        {
            "image": torch.randn(4, 128, 128, 128),
            "mask": torch.zeros(1, 128, 128, 128),
            "patient_id": f"p_{i}",
            **tf(),
        }
        for i in range(2)
    ]

    collated = jepa_masking_collate_fn_3d(batch)
    assert collated["images"].shape == (2, 4, 128, 128, 128)
    assert collated["masks"].shape == (2, 1, 128, 128, 128)
    assert collated["context_indices"].shape == (2, 192)
    assert len(collated["target_indices_list"]) == 4
    assert collated["target_indices_list"][0].shape == (2, 27)


def test_random_modality_dropout_3d():
    dropout = RandomModalityDropout3D(p_drop=0.5)
    dropout.train()

    img = torch.ones(4, 32, 32, 32)
    out = dropout(img)
    assert out.shape == (4, 32, 32, 32)

    # Test guaranteed active channel fallback when p_drop = 1.0
    dropout_all = RandomModalityDropout3D(p_drop=1.0)
    dropout_all.train()

    out_fallback = dropout_all(img)
    active_channels = [(out_fallback[c] != 0).any().item() for c in range(4)]
    assert sum(active_channels) >= 1, "Guaranteed fallback failed: all channels dropped!"

    # Eval mode: must preserve all channels
    dropout.eval()
    out_eval = dropout(img)
    assert torch.equal(out_eval, img)


def test_volumetric_augmentations_3d():
    aug = VolumetricAugmentations3D(
        flip_prob=1.0, noise_prob=1.0, modality_dropout_prob=0.0, is_training=True
    )
    img = torch.randn(4, 32, 32, 32)
    mask = torch.zeros(1, 32, 32, 32)
    mask[:, 10:20, 10:20, 10:20] = 1.0

    aug_img, aug_mask = aug(img, mask)
    assert aug_img.shape == (4, 32, 32, 32)
    assert aug_mask.shape == (1, 32, 32, 32)
