from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from brats_jepa_3d.config import get_dataset_dir, get_metadata_path


class BraTS3DDataset(Dataset):
    r"""
    Dataset loader for canonical 3D BraTS 2024 GLI multi-modal volumes (128x128x128).

    Supports:
    - Pre-computed .npz memory mapping & RAM caching
    - Integrated JEPAMaskingTransform3D (for self-supervised pre-training)
    - Integrated VolumetricAugmentations3D & RandomModalityDropout3D
    - Patient-stratified split filtering (train, val, test)
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str | None = "train",
        masking_transform: Callable | None = None,
        augmentations: Callable | None = None,
        cache_in_ram: bool = False,
        max_cache_size: int = 200,
        fraction: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir).resolve() if data_dir else get_dataset_dir("brats_gli_3d")
        self.split = split
        self.masking_transform = masking_transform
        self.augmentations = augmentations
        self.cache_in_ram = cache_in_ram
        self.max_cache_size = max_cache_size

        metadata_file = self.data_dir / "metadata.csv"
        if not metadata_file.exists():
            metadata_file = get_metadata_path("brats_gli_3d")

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        df = pd.read_csv(metadata_file)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        # Fraction subsampling for Low-Data Label Efficiency Benchmark
        # Technical Decision: In low-data regimes (1% to 50% labels), unstratified random sampling
        # risks sampling bias (e.g. over-representing micro-lesions or empty volumes). Stratifying
        # on tumor_quartile guarantees that every labeled fraction preserves the cohort's lesion size distribution.
        if fraction < 1.0 and len(df) > 0:
            n_samples = max(1, int(np.ceil(len(df) * fraction)))
            if (
                "tumor_quartile" in df.columns
                and df["tumor_quartile"].nunique() > 1
                and n_samples >= df["tumor_quartile"].nunique()
            ):
                try:
                    from sklearn.model_selection import train_test_split

                    sampled_df, _ = train_test_split(
                        df,
                        train_size=n_samples,
                        random_state=seed,
                        stratify=df["tumor_quartile"],
                    )
                    df = sampled_df.reset_index(drop=True)
                except (ValueError, KeyError, TypeError, ImportError):
                    df = df.sample(n=min(len(df), n_samples), random_state=seed).reset_index(drop=True)
            else:
                df = df.sample(n=min(len(df), n_samples), random_state=seed).reset_index(drop=True)

        self.df = df
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_volume(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx in self.cache:
            return self.cache[idx]

        row = self.df.iloc[idx]
        file_name = row["file_name"]
        file_path = self.data_dir / file_name

        if not file_path.exists():
            # Try searching anywhere under data_dir
            matches = list(self.data_dir.glob(f"**/{file_name}"))
            if matches:
                file_path = matches[0]
            else:
                raise FileNotFoundError(f"Volume file not found: {file_path}")

        with np.load(file_path, mmap_mode="r") as data:
            image = torch.from_numpy(data["image"].astype(np.float32).copy())  # [4, 128, 128, 128]
            mask = torch.from_numpy(data["mask"].astype(np.float32).copy())  # [1, 128, 128, 128]

        if self.cache_in_ram and len(self.cache) < self.max_cache_size:
            self.cache[idx] = (image, mask)

        return image, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, mask = self._load_volume(idx)
        pid = str(self.df.iloc[idx]["patient_id"])

        # Apply 3D spatial and modality dropout augmentations
        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)

        sample: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "patient_id": pid,
            "index": idx,
        }

        # Apply 3D JEPA multi-block masking
        if self.masking_transform is not None:
            mask_dict = self.masking_transform()
            sample["context_indices"] = mask_dict["context_indices"]
            sample["target_indices_list"] = mask_dict["target_indices_list"]

        return sample
