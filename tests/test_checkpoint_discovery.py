"""
tests/test_checkpoint_discovery.py
----------------------------------
Validates that:
1. Zero phantom or legacy directories (thesis_3d_outputs, visreg_pretrain_outputs)
   exist in get_checkpoint_search_dirs().
2. find_ssl_checkpoint() strictly checks .is_file() and is completely immune to IsADirectoryError.
3. Explicit folder paths are inspected internally for .pt checkpoints.
4. find_downstream_checkpoint() enforces .is_file() across all candidate globs.
5. _safe_torch_load() reliably loads checkpoints containing metadata dictionaries.
"""

import tempfile
from pathlib import Path
import torch

from brats_jepa_3d.config import get_checkpoint_search_dirs
from brats_jepa_3d.models.builder import (
    find_ssl_checkpoint,
    find_downstream_checkpoint,
    _safe_torch_load,
)


def test_no_phantom_directories():
    """Verify that get_checkpoint_search_dirs() never outputs legacy/phantom paths."""
    search_dirs = get_checkpoint_search_dirs()
    for s in search_dirs:
        s_str = str(s)
        assert "/content/drive/MyDrive/thesis_3d_outputs" not in s_str, (
            f"Phantom thesis_3d_outputs found in: {s_str}"
        )
        assert "/content/drive/MyDrive/thesis/visreg_pretrain_outputs" not in s_str, (
            f"Phantom visreg_pretrain_outputs found in: {s_str}"
        )


def test_find_ssl_checkpoint_explicit_file():
    """Verify find_ssl_checkpoint resolves an explicit valid .pt file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ckpt_file = tmp / "visreg_jepa_3d_best.pt"
        ckpt_file.write_bytes(b"dummy")
        res = find_ssl_checkpoint("visreg_jepa", checkpoints_dir=tmp, explicit_path=ckpt_file)
        assert res.resolve() == ckpt_file.resolve()
        assert res.is_file()


def test_find_ssl_checkpoint_dir_search():
    """Verify find_ssl_checkpoint searches inside a directory when a folder path is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        nested_dir = tmp / "pretrain model" / "checkpoints"
        nested_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = nested_dir / "visreg_jepa_3d_best.pt"
        ckpt_file.write_bytes(b"dummy")

        res = find_ssl_checkpoint("visreg_jepa", checkpoints_dir=tmp, explicit_path=nested_dir)
        assert res.resolve() == ckpt_file.resolve()
        assert res.is_file()


def test_is_a_directory_error_defense():
    """Verify that a directory named with a .pt extension is never returned as a checkpoint file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pretrain_dir = tmp / "pretrain model" / "checkpoints"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        real_file = pretrain_dir / "visreg_jepa_3d_best.pt"
        real_file.write_bytes(b"dummy")

        # Rogue directory ending in .pt (simulating mkdir -p .../visreg_jepa_3d_best.pt)
        rogue_dir = tmp / "outputs" / "visreg_jepa_3d_best.pt"
        rogue_dir.mkdir(parents=True, exist_ok=True)

        res = find_ssl_checkpoint("visreg_jepa", checkpoints_dir=pretrain_dir, explicit_path=rogue_dir)
        assert res.resolve() != rogue_dir.resolve()
        assert res.resolve() == real_file.resolve()
        assert res.is_file()


def test_empty_string_path_fallback():
    """Verify that an empty string explicit path falls back gracefully to search_dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ckpt_dir = tmp / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        real_file = ckpt_dir / "visreg_jepa_3d_best.pt"
        real_file.write_bytes(b"dummy")

        res = find_ssl_checkpoint("visreg_jepa", checkpoints_dir=ckpt_dir, explicit_path="")
        assert res.resolve() == real_file.resolve()
        assert res.is_file()


def test_find_downstream_checkpoint_is_file_guarded():
    """Verify that find_downstream_checkpoint ignores rogue directories matching glob patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ds_dir = tmp / "downstream" / "checkpoints"
        ds_dir.mkdir(parents=True, exist_ok=True)
        real_ds_file = ds_dir / "visreg_jepa_unetr_hybrid_frozen_best.pt"
        real_ds_file.write_bytes(b"dummy")

        # Rogue downstream directory
        rogue_ds_dir = tmp / "outputs" / "visreg_jepa_unetr_hybrid_frozen_best.pt"
        rogue_ds_dir.mkdir(parents=True, exist_ok=True)

        res = find_downstream_checkpoint(
            "visreg_jepa",
            decoder_type="unetr_hybrid",
            checkpoints_dir=ds_dir,
            is_frozen=True,
        )
        assert res.resolve() == real_ds_file.resolve()
        assert res.is_file()


def test_safe_torch_load_metadata():
    """Verify that _safe_torch_load loads state dicts with custom metadata without unpickling errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ckpt_path = tmp / "test_ckpt.pt"
        torch.save(
            {
                "model_state_dict": {},
                "pool_fingerprint": {"metadata_sha1": "abcdef123456", "n_rows": 100},
                "best_val_dice": 0.825,
                "epoch": 25,
            },
            ckpt_path,
        )
        loaded = _safe_torch_load(ckpt_path, map_location="cpu")
        assert loaded["epoch"] == 25
        assert loaded["best_val_dice"] == 0.825
        assert loaded["pool_fingerprint"]["metadata_sha1"] == "abcdef123456"


if __name__ == "__main__":
    test_no_phantom_directories()
    test_find_ssl_checkpoint_explicit_file()
    test_find_ssl_checkpoint_dir_search()
    test_is_a_directory_error_defense()
    test_empty_string_path_fallback()
    test_find_downstream_checkpoint_is_file_guarded()
    test_safe_torch_load_metadata()
    print("✓ All 7 checkpoint discovery and defense tests passed successfully!")
