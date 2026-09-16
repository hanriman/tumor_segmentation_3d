import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    r"""
    Enforces deterministic reproducibility across Python, NumPy, PyTorch CPU, CUDA, and MPS.

    Mathematical & Methodological Justification:
    --------------------------------------------
    In multi-modal medical neuroimaging research, variance across stochastic operations
    (e.g., random patch block masking, modality dropout, spatial affine transforms, network weight initialization)
    can obscure true performance differences between self-supervised representation regularizations
    (I-JEPA vs SigReg vs VisReg). Strict pseudo-random number generator (PRNG) state seeding across
    Seeds 42, 43, 44 guarantees that all comparative benchmarks operate on statistically identical
    stochastic sequences, ensuring reproducible findings across diverse hardware platforms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
