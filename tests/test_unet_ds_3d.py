import torch

from brats_jepa_3d.losses import DeepSupervisionLoss3D
from brats_jepa_3d.models import BraTS3DUNet


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
    assert all(o.shape[:2] == (1, 1) for o in outs)


def test_unet_ds_eval_and_disabled_return_single():
    m = _tiny_unet(deep_supervision=True)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(1, 4, 32, 32, 32))
    assert isinstance(y, torch.Tensor) and tuple(y.shape) == (1, 1, 32, 32, 32)

    m2 = _tiny_unet(deep_supervision=False)
    m2.train()
    with torch.no_grad():
        y2 = m2(torch.randn(1, 4, 32, 32, 32))
    assert isinstance(y2, torch.Tensor) and tuple(y2.shape) == (1, 1, 32, 32, 32)
    assert not [k for k in m2.state_dict() if k.startswith("ds")]


def test_unet_ds_loss_backward():
    m = _tiny_unet(deep_supervision=True)
    m.train()
    crit = DeepSupervisionLoss3D()
    logits = m(torch.randn(1, 4, 32, 32, 32))
    target = (torch.rand(1, 1, 32, 32, 32) > 0.5).float()
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
    assert r1 == r2 == [(1, 1, 32, 32, 32), (1, 1, 16, 16, 16), (1, 1, 8, 8, 8), (1, 1, 4, 4, 4)]
