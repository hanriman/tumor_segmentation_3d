import zipfile
from pathlib import Path

import pytest

from brats_jepa_3d.utils.export import (
    export_artifacts,
    import_artifacts,
    resolve_outputs_source_dir,
)


@pytest.fixture
def mock_outputs_structure(tmp_path: Path):
    """Creates a mock outputs directory structure with checkpoints, metrics, and logs."""
    outputs_dir = tmp_path / "outputs"
    ckpts = outputs_dir / "checkpoints"
    metrics = outputs_dir / "metrics"
    logs = outputs_dir / "logs"

    ckpts.mkdir(parents=True)
    metrics.mkdir(parents=True)
    logs.mkdir(parents=True)

    # Checkpoint files
    (ckpts / "visreg_jepa_best.pt").write_bytes(b"dummy_visreg_weights_12345")
    (ckpts / "visreg_jepa_multiscale_best.pt").write_bytes(b"dummy_visreg_downstream_weights")
    (ckpts / "nnunet_3d_best.pt").write_bytes(b"dummy_nnunet_weights")
    (ckpts / "unet_3d_best.pt").write_bytes(b"dummy_unet_weights")

    # Metrics files
    (metrics / "visreg_jepa_metrics.json").write_text('{"dice": 0.901}')
    (metrics / "master_3d_benchmark.csv").write_text("model,dice\nvisreg,0.901")
    (metrics / "nnunet_metrics.json").write_text('{"dice": 0.885}')

    # Logs
    (logs / "train_visreg_jepa.log").write_text("Epoch 1... Epoch 50 done.")

    return outputs_dir


def test_export_artifacts_model_prefix_filtering(tmp_path: Path, mock_outputs_structure: Path):
    export_dir = tmp_path / "export_visreg"
    zip_path = tmp_path / "visreg_outputs.zip"

    res = export_artifacts(
        export_name="visreg_outputs",
        export_dir=export_dir,
        zip_path=zip_path,
        model_prefix="visreg",
        source_dir=mock_outputs_structure,
        verbose=False,
    )

    assert res["total_files"] >= 4
    assert zip_path.exists()
    assert res["zip_size_bytes"] > 0

    # Verify only visreg checkpoints and master files were copied
    exported_ckpts = [f.name for f in (export_dir / "checkpoints").iterdir()]
    assert "visreg_jepa_best.pt" in exported_ckpts
    assert "visreg_jepa_multiscale_best.pt" in exported_ckpts
    assert "nnunet_3d_best.pt" not in exported_ckpts
    assert "unet_3d_best.pt" not in exported_ckpts

    # Verify zip archive contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert "checkpoints/visreg_jepa_best.pt" in names
        assert "metrics/visreg_jepa_metrics.json" in names
        assert "metrics/master_3d_benchmark.csv" in names


def test_export_artifacts_all_models(tmp_path: Path, mock_outputs_structure: Path):
    export_dir = tmp_path / "export_all"
    zip_path = tmp_path / "all_outputs.zip"

    res = export_artifacts(
        export_name="all_outputs",
        export_dir=export_dir,
        zip_path=zip_path,
        model_prefix=None,
        source_dir=mock_outputs_structure,
        verbose=False,
    )

    assert res["total_files"] >= 7
    exported_ckpts = [f.name for f in (export_dir / "checkpoints").iterdir()]
    assert "visreg_jepa_best.pt" in exported_ckpts
    assert "nnunet_3d_best.pt" in exported_ckpts
    assert "unet_3d_best.pt" in exported_ckpts


def test_import_artifacts_from_zip(tmp_path: Path, mock_outputs_structure: Path):
    # First create a zip archive to simulate a mounted Kaggle notebook output
    staging_dir = tmp_path / "staging"
    zip_path = tmp_path / "input_mount" / "visreg_outputs.zip"
    export_artifacts(
        export_name="visreg_outputs",
        export_dir=staging_dir,
        zip_path=zip_path,
        model_prefix="visreg",
        source_dir=mock_outputs_structure,
        verbose=False,
    )

    # Import into target directory
    import_target = tmp_path / "imported_outputs"
    imported = import_artifacts(
        input_dir=tmp_path / "input_mount",
        target_dir=import_target,
        mirror_to_local_outputs=False,
        verbose=False,
    )

    assert len(imported) >= 4
    assert (import_target / "checkpoints" / "visreg_jepa_best.pt").exists()
    assert (import_target / "metrics" / "visreg_jepa_metrics.json").exists()


def test_resolve_outputs_source_dir_fallback(tmp_path: Path):
    preferred = tmp_path / "custom_outputs"
    preferred.mkdir()
    resolved = resolve_outputs_source_dir(preferred)
    assert resolved == preferred.resolve()
