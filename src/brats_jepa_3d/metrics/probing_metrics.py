import torch
import torch.nn.functional as F


def compute_effective_rank(z: torch.Tensor) -> float:
    r"""
    Effective Rank (Spectral Entropy of Empirical Covariance) of 3D Latent Feature Tokens.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Information-Theoretic Dimensionality (Roy & Vetterli, 2007):
       Measures the effective dimensionality occupied by latent feature vectors z \in \mathbb{R}^{N \times D}.
       Let \tilde{z} = z - \bar{z} be centered feature tokens. The empirical covariance matrix is:
           \Sigma = \frac{1}{N - 1} \tilde{z}^\top \tilde{z}
       Its eigenvalues \lambda_k are proportional to the squared singular values S_k^2 from SVD(\tilde{z}):
           \lambda_k \propto S_k^2
       Normalizing eigenvalues produces a valid discrete probability distribution:
           p_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j} = \frac{S_k^2}{\sum_{j=1}^D S_k^2}
       The effective rank is the exponential of the Shannon entropy of this spectral distribution:
           \text{erank}(z) = \exp\left(-\sum_{k=1}^D p_k \ln(p_k)\right)

    2. Why Squared Singular Values (S^2) Are Mathematically Mandatory:
       Using linear singular values S_k artificially compresses dynamic range (since \sqrt{x} flattens peaks),
       masking dimensional collapse and spuriously reporting high ranks.
       Using S_k^2 strictly reflects the variance explained along each principal axis.
       - Maximum value: D = 384 (perfect isotropy across all orthogonal dimensions).
       - Minimum value: 1.0 (all variance concentrated along a single 1D ray; complete collapse).

    References:
    -----------
    - Roy, O., & Vetterli, M. (2007). "The effective rank: A measure of effective dimensionality."
      15th European Signal Processing Conference (EUSIPCO 2007), pp. 606-610.
    """
    # Upcast to float32 to ensure SVD is supported and numerically stable across CUDA, MPS, and CPU
    z_flat = z.reshape(-1, z.shape[-1]).float()
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        eigenvalues = S**2
        if eigenvalues.sum() < 1e-12:
            return 1.0  # Degenerate case: all-zero representations
        normalized = eigenvalues / eigenvalues.sum()
        entropy = -torch.sum(normalized * torch.log(normalized + 1e-12))
        eff_rank = torch.exp(entropy).item()
        return eff_rank
    except (RuntimeError, ValueError):
        return 1.0


def compute_representation_collapse_metrics(z: torch.Tensor) -> dict[str, float]:
    r"""
    Multi-Faceted Representation Collapse Diagnostic Suite.
    """
    z_flat = z.reshape(-1, z.shape[-1]).float()  # [N, D] in float32
    N_total = z_flat.shape[0]

    if N_total <= 1:
        eff_rank = compute_effective_rank(z_flat)
        return {
            "effective_rank": eff_rank,
            "avg_cosine_sim": 1.0 if N_total == 1 else 0.0,
            "avg_cosine_sim_centered": 0.0,
            "feature_variance": 0.0,
        }

    # Sample subset for pairwise similarity if token count is very large
    if N_total > 1000:
        indices = torch.randperm(N_total, device=z_flat.device)[:1000]
        z_sample = z_flat[indices]
    else:
        z_sample = z_flat

    N = z_sample.shape[0]
    mask = ~torch.eye(N, device=z.device, dtype=torch.bool)

    # 1. Uncentered cosine similarity
    z_norm = F.normalize(z_sample, p=2, dim=-1)
    sim_matrix = z_norm @ z_norm.T
    avg_cosine_sim = sim_matrix[mask].mean().item()

    # 2. Centered cosine similarity (Wang & Isola, ICML 2020)
    z_centered = z_sample - z_sample.mean(dim=0, keepdim=True)
    z_centered_norm = F.normalize(z_centered, p=2, dim=-1, eps=1e-8)
    sim_matrix_centered = z_centered_norm @ z_centered_norm.T
    avg_cosine_sim_centered = sim_matrix_centered[mask].mean().item()

    eff_rank = compute_effective_rank(z_flat)
    feature_var = z_flat.var(dim=0).mean().item()

    return {
        "effective_rank": eff_rank,
        "avg_cosine_sim": avg_cosine_sim,
        "avg_cosine_sim_centered": avg_cosine_sim_centered,
        "feature_variance": feature_var,
    }
