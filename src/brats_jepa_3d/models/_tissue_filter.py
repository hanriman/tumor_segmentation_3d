import logging

import torch

logger = logging.getLogger(__name__)


def filter_tissue_tokens(
    projected_tokens: torch.Tensor,
    tissue_mask: torch.Tensor | None,
    min_tokens: int = 32,
) -> torch.Tensor:
    r"""Keeps tissue rows of projected tokens for distribution regularization.

    Per-sample rule: sample `b` contributes its masked rows when it holds at
    least `min_tokens` tissue tokens, otherwise all its `N_ctx` rows (a thin
    volume must not poison — or starve — the batch). Output is 2D
    `[N_kept, proj_dim]` when filtering occurs (matching the loss flattening
    contract); with no (or malformed) mask the input is returned unchanged to
    preserve legacy `[B, N_ctx, proj_dim]` callers.

    Raises:
        ValueError: if `projected_tokens` is not 3D `[B, N_ctx, proj_dim]`
            (filters operate on batched context projections; a 2D tensor here
            means the caller flattened before filtering — a loud misuse that
            would otherwise silently misalign the mask).
    """
    if projected_tokens.dim() != 3:
        raise ValueError(
            f"filter_tissue_tokens expects 3D [B, N_ctx, proj_dim], got {tuple(projected_tokens.shape)}."
        )
    if tissue_mask is None:
        logger.warning(
            "filter_tissue_tokens: tissue_mask is None — regularizer sees unfiltered "
            "tokens including air padding."
        )
        return projected_tokens
    if tissue_mask.dim() != 2 or tissue_mask.shape != projected_tokens.shape[:2]:
        logger.warning(
            "filter_tissue_tokens: malformed tissue_mask %s vs tokens %s — "
            "returning unfiltered tokens.",
            tuple(tissue_mask.shape),
            tuple(projected_tokens.shape),
        )
        return projected_tokens
    mask = tissue_mask.to(device=projected_tokens.device, dtype=torch.bool)
    kept = [
        full_projected_b[m_b] if int(m_b.sum().item()) >= min_tokens else full_projected_b
        for full_projected_b, m_b in zip(projected_tokens.unbind(0), mask.unbind(0))
    ]
    return torch.cat(kept, dim=0)
