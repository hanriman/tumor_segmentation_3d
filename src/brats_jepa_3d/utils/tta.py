import torch

# Orthogonal reflection axes over [B, C, D, H, W]: sagittal (W), coronal (H), axial (D).
TTA_FLIP_DIMS: tuple[tuple[int, ...] | None, ...] = (None, (-1,), (-2,), (-3,))

_LOGIT_EPS = 1e-6


def predict_with_tta_3d(
    model: torch.nn.Module, volume: torch.Tensor
) -> torch.Tensor:
    r"""4-fold test-time augmentation via orthogonal reflections.

    Evaluates the model on the original volume plus flips along each spatial axis,
    un-flips each probability map, and returns ensemble-averaged logits so that
    downstream metric code (`from_logits=True`) works unchanged:

        p_avg = (p0 + px + py + pz) / 4,  out = logit(clamp(p_avg))

    Brain anatomy is approximately bilateral-symmetric (sagittal) and smoothly
    continuous (coronal/axial); averaging suppresses random boundary noise while
    preserving the exact tensor interface [B, 1, D, H, W].
    """
    was_training = model.training
    model.eval()
    probs_sum = None
    n_views = 0
    with torch.no_grad():
        for dims in TTA_FLIP_DIMS:
            v = torch.flip(volume, dims=dims) if dims is not None else volume
            out = model(v)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            probs = torch.sigmoid(logits.float())
            if dims is not None:
                probs = torch.flip(probs, dims=dims)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            n_views += 1
    if was_training:
        model.train()
    p_avg = (probs_sum / n_views).clamp(_LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return torch.log(p_avg / (1.0 - p_avg)).to(dtype=volume.dtype if volume.is_floating_point() else torch.float32)
