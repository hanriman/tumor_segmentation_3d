"""Region-resolved battery tests: low-data tiers and OOD regimes report WT/TC/ET.

Covers the Phase-2 extension of evaluate_low_data_3d.train_and_eval and
evaluate_ood_3d.evaluate_perturbation from scalar means to per-region dicts.
Uses tiny synthetic volumes (16^3) so the suite stays CPU-fast.
"""
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "scripts")
import evaluate_low_data_3d as lowmod
import evaluate_ood_3d as oodmod


class _Fixed5(torch.nn.Module):
    """Fixed C=5 logits: argmax paints WT everywhere, ET/TC on known voxels."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        B = x.shape[0]
        base = torch.full((1, 5, 16, 16, 16), -10.0)
        base[0, 2] = 10.0  # ED everywhere -> WT=1
        base[0, 4, :4, :4, :4] = 20.0  # ET cube (strict max over ED) -> TC/ET=1 there
        return base.expand(B, -1, -1, -1, -1) + self.dummy


class _Fixed1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return torch.full((x.shape[0], 1, 16, 16, 16), 2.0) + self.dummy


def _loader(n=2, et_free=False):
    data = []
    for _ in range(n):
        m = torch.full((1, 16, 16, 16), 2.0)  # ED
        if not et_free:
            m[0, :4, :4, :4] = 4.0
        data.append({"image": torch.randn(4, 16, 16, 16), "mask": m,
                     "brain_mask": torch.ones(1, 16, 16, 16)})
    return DataLoader(data, batch_size=1)


def test_low_data_returns_regions_multiclass():
    torch.manual_seed(0)
    loader = _loader()
    out = lowmod.train_and_eval(
        _Fixed5().eval(), loader, loader, torch.device("cpu"),
        epochs=1, amp=False, smoke_test=True, num_classes=5)
    assert set(out) == {"WT", "TC", "ET", "mean"}, out.keys()
    assert abs(out["mean"] - (out["WT"] + out["TC"] + out["ET"]) / 3.0) < 1e-6
    assert out["WT"] == 1.0 and out["TC"] == 1.0 and out["ET"] == 1.0


def test_low_data_binary_legacy_shape():
    torch.manual_seed(0)
    data = [{"image": torch.randn(4, 16, 16, 16),
             "mask": (torch.rand(1, 16, 16, 16) > 0.9).float()} for _ in range(2)]
    loader = DataLoader(data, batch_size=1)
    out = lowmod.train_and_eval(
        _Fixed1().eval(), loader, loader, torch.device("cpu"),
        epochs=1, amp=False, smoke_test=True, num_classes=1)
    assert set(out) == {"mean"}, out.keys()
    assert 0.0 <= out["mean"] <= 1.0


def test_ood_returns_regions_and_et_free_graceful():
    torch.manual_seed(0)
    out = oodmod.evaluate_perturbation(
        _Fixed5().eval(), _loader(et_free=False), torch.device("cpu"),
        None, amp=False, smoke_test=True)
    assert set(out) == {"WT", "TC", "ET", "mean"}
    assert out["ET"] == 1.0
    # ET-free cohort: no crash, ET mean falls back to empty-empty perfect 1.0s
    out2 = oodmod.evaluate_perturbation(
        _Fixed5().eval(), _loader(et_free=True), torch.device("cpu"),
        None, amp=False, smoke_test=True)
    assert out2["ET"] >= 0.0
    # binary legacy
    data = [{"image": torch.randn(4, 16, 16, 16),
             "mask": (torch.rand(1, 16, 16, 16) > 0.9).float(),
             "brain_mask": torch.ones(1, 16, 16, 16)} for _ in range(2)]
    out3 = oodmod.evaluate_perturbation(
        _Fixed1().eval(), DataLoader(data, batch_size=1), torch.device("cpu"),
        None, amp=False, smoke_test=True)
    assert set(out3) == {"mean"}
