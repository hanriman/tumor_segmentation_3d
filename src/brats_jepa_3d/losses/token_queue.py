import torch
from torch import nn


class FIFOTokenQueue(nn.Module):
    r"""
    FIFO Token Memory Queue for 3D Self-Supervised Statistical Regularization.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Statistical Estimation Precision & Dimensional Decoupling:
       In statistical self-supervised learning (VISReg, Wu et al., 2026; SIGReg, Balestriero & LeCun, 2025),
       regularizers enforce zero-mean, unit-variance, and isotropic Gaussian distribution over
       latent tokens. Estimating sample covariance and distribution quantiles accurately requires
       a large sample of tokens.
       In 3D medical imaging, physical mini-batch size is constrained (B=2..16) due to volumetric VRAM.
       The FIFO Token Memory Queue buffers detached token coordinates from preceding steps,
       decoupling the gradient batch size (B=16 volumes = 3,072 tokens) from the empirical
       regularizer sample size (K=128 volumes = 24,576 tokens).

    2. Gradient Isolation:
       Gradients flow strictly into the current batch's tokens (`requires_grad=True`), while cached
       queue tokens serve purely as an empirical reference distribution (`requires_grad=False`).
       This incurs 0% backward pass activation overhead and requires only ~12 MB of VRAM.

    3. Evaluation & Checkpoint Safety:
       - In eval mode (`self.training == False`), the queue is bypassed to ensure deterministic
         evaluation on the evaluation batch alone.
       - Buffers are registered with `persistent=False` so transient cached tokens are never
         serialized into persistent model checkpoints.
    """

    def __init__(self, max_tokens: int = 24576, feature_dim: int | None = None):
        super().__init__()
        self.max_tokens = max_tokens
        self.feature_dim = feature_dim
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("current_size", torch.tensor(0, dtype=torch.long), persistent=False)

        initial_queue = (
            torch.zeros(max_tokens, feature_dim, dtype=torch.float32)
            if feature_dim is not None
            else None
        )
        self.register_buffer("queue", initial_queue, persistent=False)

    def _ensure_queue(self, cur: torch.Tensor) -> None:
        dim = cur.shape[-1]
        device = cur.device
        if (
            self.queue is None
            or self.queue.shape[1] != dim
            or self.queue.device != device
        ):
            self.feature_dim = dim
            self.queue = torch.zeros(
                self.max_tokens, dim, device=device, dtype=torch.float32
            )
            self.ptr.zero_()
            self.current_size.zero_()

    @torch.no_grad()
    def update(self, new_tokens: torch.Tensor) -> None:
        """Pushes detached new_tokens [N, D] into the circular buffer."""
        tokens = new_tokens.detach().float().reshape(-1, new_tokens.shape[-1])
        n = tokens.shape[0]
        if n == 0 or self.max_tokens <= 0:
            return

        self._ensure_queue(tokens)

        if n >= self.max_tokens:
            self.queue.copy_(tokens[-self.max_tokens:])
            self.ptr.zero_()
            self.current_size.fill_(self.max_tokens)
            return

        ptr = int(self.ptr.item())
        if ptr + n <= self.max_tokens:
            self.queue[ptr : ptr + n].copy_(tokens)
            self.ptr.add_(n)
        else:
            tail = self.max_tokens - ptr
            self.queue[ptr:].copy_(tokens[:tail])
            head = n - tail
            self.queue[:head].copy_(tokens[tail:])
            self.ptr.fill_(head)

        new_size = min(self.max_tokens, int(self.current_size.item()) + n)
        self.current_size.fill_(new_size)

    def get_all(self, current_batch: torch.Tensor) -> torch.Tensor:
        """
        Concatenates current_batch (preserving autograd history) with detached cached queue tokens.

        Returns:
            Concatenated tensor of shape [N_cur + N_queue, D].
        """
        cur = current_batch.reshape(-1, current_batch.shape[-1])
        if not self.training or self.max_tokens <= 0:
            return cur

        self._ensure_queue(cur)
        sz = int(self.current_size.item())
        if sz == 0:
            return cur

        cached = self.queue[:sz].to(device=cur.device, dtype=cur.dtype)
        return torch.cat([cur, cached], dim=0)

    def reset(self) -> None:
        """Resets the queue to empty state."""
        self.ptr.zero_()
        self.current_size.zero_()

    def __len__(self) -> int:
        return int(self.current_size.item())
