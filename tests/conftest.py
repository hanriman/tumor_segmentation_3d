import pytest
import torch


@pytest.fixture
def sample_volume_3d() -> torch.Tensor:
    """Returns a synthetic 4-channel 3D volume [B=2, C=4, D=128, H=128, W=128]."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 128, 128, 128)


@pytest.fixture
def sample_mask_3d() -> torch.Tensor:
    """Returns a synthetic 5-class segmentation mask [B=2, C=1, D=128, H=128, W=128] with BraTS labels (0-4)."""
    mask = torch.zeros(2, 1, 128, 128, 128, dtype=torch.long)
    # Insert synthetic tumor compartments (1=NCR, 2=ED, 3=NET, 4=ET)
    mask[:, :, 50:60, 50:60, 50:60] = 1
    mask[:, :, 60:70, 60:70, 60:70] = 2
    mask[:, :, 70:73, 70:73, 70:73] = 3
    mask[:, :, 73:75, 73:75, 73:75] = 4
    return mask


@pytest.fixture
def sample_small_volume_3d() -> torch.Tensor:
    """Returns a smaller 3D volume for rapid unit testing [B=2, C=4, D=32, H=32, W=32]."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 32, 32, 32)
