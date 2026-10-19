import torch

# Orthogonal reflection axes over [B, C, D, H, W]: sagittal (W), coronal (H), axial (D).
TTA_FLIP_DIMS: tuple[tuple[int, ...] | None, ...] = (None, (-1,), (-2,), (-3,))

_LOGIT_EPS = 1e-6


def _model_logits(model: torch.nn.Module, volume: torch.Tensor) -> torch.Tensor:
    out = model(volume)
    return out[0] if isinstance(out, (list, tuple)) else out


def model_out_channels(model: torch.nn.Module) -> int | None:
    """Best-effort output-channel probe for TTA dispatch (None if unknown)."""
    for obj in (model, getattr(model, "decoder", None), getattr(model, "unet", None),
                getattr(model, "dynunet", None)):
        if obj is None:
            continue
        head = getattr(obj, "head", None)
        if head is not None and hasattr(head, "out_channels"):
            return int(head.out_channels)
        if hasattr(obj, "out_channels") and isinstance(obj.out_channels, int):
            return int(obj.out_channels)
    return None


def tta_predict_logits(model: torch.nn.Module, volume: torch.Tensor) -> torch.Tensor:
    """TTA dispatch: binary flipped-sigmoid average for C=1, softmax average
    for multi-class. Channel count comes from the model when known, else from
    a single probe forward (no grad)."""
    n_ch = model_out_channels(model)
    if n_ch is None:
        with torch.no_grad():
            n_ch = _model_logits(model, volume[:1]).shape[1]
    if n_ch == 1:
        return predict_with_tta_3d(model, volume)
    return predict_with_tta_multiclass_3d(model, volume)


def _run_tta_flips(
    model: torch.nn.Module,
    volume: torch.Tensor,
    multiclass: bool,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    probs_sum = None
    n_views = 0
    with torch.no_grad():
        for dims in TTA_FLIP_DIMS:
            v = torch.flip(volume, dims=dims) if dims is not None else volume
            logits = _model_logits(model, v)
            c = logits.shape[1]
            if multiclass:
                if c < 2:
                    raise ValueError(
                        f"predict_with_tta_multiclass_3d needs C>=2, got C={c}."
                    )
                probs = torch.softmax(logits.float(), dim=1)
            else:
                if c != 1:
                    raise ValueError(
                        f"predict_with_tta_3d is binary-only (C=1), got C={c}; "
                        f"use predict_with_tta_multiclass_3d for multi-class outputs."
                    )
                probs = torch.sigmoid(logits.float())
            if dims is not None:
                probs = torch.flip(probs, dims=dims)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            n_views += 1
    if was_training:
        model.train()

    out_dtype = volume.dtype if volume.is_floating_point() else torch.float32
    if multiclass:
        p_avg = (probs_sum / n_views).clamp(_LOGIT_EPS, 1.0)
        return torch.log(p_avg).to(dtype=out_dtype)
    p_avg = (probs_sum / n_views).clamp(_LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return torch.log(p_avg / (1.0 - p_avg)).to(dtype=out_dtype)


def predict_with_tta_3d(
    model: torch.nn.Module, volume: torch.Tensor
) -> torch.Tensor:
    r"""4-fold test-time augmentation via orthogonal reflections (binary, C=1)."""
    return _run_tta_flips(model, volume, multiclass=False)


def predict_with_tta_multiclass_3d(
    model: torch.nn.Module, volume: torch.Tensor
) -> torch.Tensor:
    r"""4-fold TTA for multi-class logits [B, C, D, H, W] (BraTS regions protocol)."""
    return _run_tta_flips(model, volume, multiclass=True)
