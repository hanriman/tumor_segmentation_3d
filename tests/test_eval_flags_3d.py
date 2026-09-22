import importlib.util
import sys

import torch
from torch.utils.data import DataLoader

from brats_jepa_3d.models import BraTS3DUNet


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [name]
    spec.loader.exec_module(mod)
    return mod


def test_low_data_flags():
    m = _load_script("evaluate_low_data_3d")
    sys.argv = ["x"]
    a = m.parse_args()
    assert a.decoder_type == "multiscale" and a.loss_type == "dice_bce" and a.tta is False
    sys.argv = ["x", "--decoder_type", "unetr_hybrid", "--loss_type", "tversky", "--tta"]
    a = m.parse_args()
    assert a.decoder_type == "unetr_hybrid" and a.loss_type == "tversky" and a.tta is True


def test_ood_flags():
    m = _load_script("evaluate_ood_3d")
    sys.argv = ["x"]
    a = m.parse_args()
    assert a.decoder_type == "multiscale" and a.tta is False
    sys.argv = ["x", "--decoder_type", "unetr_hybrid", "--tta"]
    a = m.parse_args()
    assert a.decoder_type == "unetr_hybrid" and a.tta is True


def _tiny_loader(n=2):
    data = [
        {"image": torch.randn(4, 32, 32, 32), "mask": (torch.rand(1, 32, 32, 32) > 0.9).float()}
        for _ in range(n)
    ]
    return DataLoader(data, batch_size=1)


def test_low_data_train_and_eval_tversky_tta():
    m = _load_script("evaluate_low_data_3d")
    device = torch.device("cpu")
    model = BraTS3DUNet(channels=(8, 16, 32, 64, 128), num_res_units=1, dropout=0.0)
    scores = m.train_and_eval(
        model, _tiny_loader(), _tiny_loader(), device, epochs=1, amp=False,
        smoke_test=True, loss_type="tversky", tta=True,
    )
    assert set(scores) == {"mean"}  # binary C=1 keeps legacy scalar shape
    assert 0.0 <= scores["mean"] <= 1.0


def test_low_data_train_and_eval_dice_no_tta():
    m = _load_script("evaluate_low_data_3d")
    device = torch.device("cpu")
    model = BraTS3DUNet(channels=(8, 16, 32, 64, 128), num_res_units=1, dropout=0.0)
    scores = m.train_and_eval(
        model, _tiny_loader(), _tiny_loader(), device, epochs=1, amp=False, smoke_test=True,
    )
    assert set(scores) == {"mean"}
    assert 0.0 <= scores["mean"] <= 1.0


def test_ood_perturbation_tta():
    m = _load_script("evaluate_ood_3d")
    device = torch.device("cpu")
    model = BraTS3DUNet(channels=(8, 16, 32, 64, 128), num_res_units=1, dropout=0.0)
    scores = m.evaluate_perturbation(
        model, _tiny_loader(), device, None, amp=False, smoke_test=True, tta=True,
    )
    assert set(scores) == {"mean"}
    assert 0.0 <= scores["mean"] <= 1.0
