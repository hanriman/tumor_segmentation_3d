import torch

from brats_jepa_3d.utils import TTA_FLIP_DIMS, predict_with_tta_3d


class _ConstLogit(torch.nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def forward(self, x):
        return torch.full((x.shape[0], 1) + x.shape[2:], self.c)


class _PatternLogit(torch.nn.Module):
    """Input-independent asymmetric pattern: TTA output must equal manual flip-average."""

    def forward(self, x):
        B = x.shape[0]
        base = torch.linspace(-2.0, 2.0, x.shape[2] * x.shape[3] * x.shape[4]).reshape(
            1, 1, x.shape[2], x.shape[3], x.shape[4]
        )
        return base.expand(B, -1, -1, -1, -1)


def test_tta_views():
    assert len(TTA_FLIP_DIMS) == 4 and TTA_FLIP_DIMS[0] is None


def test_tta_constant_model_identity():
    torch.manual_seed(0)
    model = _ConstLogit(0.7).eval()
    vol = torch.randn(2, 4, 16, 16, 16)
    with torch.no_grad():
        single = model(vol)
        tta = predict_with_tta_3d(model, vol)
    assert tta.shape == single.shape
    assert torch.allclose(tta, single, atol=1e-4)


def test_tta_flip_average_matches_manual():
    torch.manual_seed(1)
    model = _PatternLogit().eval()
    vol = torch.randn(1, 2, 8, 8, 8)
    with torch.no_grad():
        tta = predict_with_tta_3d(model, vol)
        acc = None
        for dims in TTA_FLIP_DIMS:
            v = torch.flip(vol, dims=dims) if dims is not None else vol
            p = torch.sigmoid(model(v).float())
            if dims is not None:
                p = torch.flip(p, dims=dims)
            acc = p if acc is None else acc + p
        expected = torch.logit((acc / 4).clamp(1e-6, 1 - 1e-6)).to(tta.dtype)
    assert torch.allclose(tta, expected, atol=1e-5)


def test_tta_preserves_eval_mode_and_shape():
    model = _ConstLogit(-0.3)
    model.train()
    vol = torch.randn(1, 1, 8, 8, 8)
    out = predict_with_tta_3d(model, vol)
    assert out.shape == (1, 1, 8, 8, 8)
    assert model.training, "TTA must restore training mode"
