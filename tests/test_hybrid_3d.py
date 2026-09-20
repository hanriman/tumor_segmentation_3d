import torch

from brats_jepa_3d.models import HybridUNETRDecoder3D, JEPASegmentationModel3D


def _fake_intermediates(B=1, L=8, N=512, D=384):
    torch.manual_seed(0)
    return [torch.randn(B, N, D) for _ in range(L)]


def test_hybrid_eval_shape():
    dec = HybridUNETRDecoder3D()
    dec.eval()
    with torch.no_grad():
        y = dec(_fake_intermediates(), raw_volume=torch.randn(1, 4, 128, 128, 128))
    assert tuple(y.shape) == (1, 1, 128, 128, 128)


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
    assert tuple(y.shape) == (1, 1, 128, 128, 128)


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
    from brats_jepa_3d.models import MultiScaleViTSegmentationDecoder3D
    fpn = MultiScaleViTSegmentationDecoder3D(deep_supervision=True)
    hyb = HybridUNETRDecoder3D(deep_supervision=True)
    fpn_sd = fpn.state_dict()
    hyb_sd = hyb.state_dict()
    # strict=False still rejects size mismatches, so filter to compatible keys
    # (the realistic transfer path); fuse3/up4/fuse4/head stay freshly initialized.
    compatible = {k: v for k, v in fpn_sd.items() if k in hyb_sd and hyb_sd[k].shape == v.shape}
    missing, unexpected = hyb.load_state_dict(compatible, strict=False)
    assert not unexpected
    assert set(missing) == set(hyb_sd) - set(compatible)
    # Only shape-diverged fusion layers + fresh stem stay uninitialized.
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
    assert tuple(y.shape) == (1, 1, 32, 32, 32)
