import pytest
import numpy as np
import torch


@pytest.fixture
def sample_volume_3d() -> torch.Tensor:
    """Returns a synthetic 4-channel 3D volume [B=2, C=4, D=128, H=128, W=128]."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 128, 128, 128)


@pytest.fixture
def sample_mask_3d() -> torch.Tensor:
    """Returns a synthetic binary segmentation mask [B=2, C=1, D=128, H=128, W=128]."""
    mask = torch.zeros(2, 1, 128, 128, 128, dtype=torch.float32)
    # Insert a synthetic foreground tumor lesion
    mask[:, :, 50:75, 50:75, 50:75] = 1.0
    return mask


@pytest.fixture
def sample_small_volume_3d() -> torch.Tensor:
    """Returns a smaller 3D volume for rapid unit testing [B=2, C=4, D=32, H=32, W=32]."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 32, 32, 32)
