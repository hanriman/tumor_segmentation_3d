import logging
import os
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from brats_jepa_3d.config import (
    IN_KAGGLE,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


def resolve_outputs_source_dir(preferred_dir: Path | str | None = None) -> Path:
    """Resolves the active outputs directory, checking configured OUTPUTS_DIR and common fallbacks."""
    if preferred_dir is not None:
        p = Path(preferred_dir).resolve()
        if p.exists():
            return p

    candidates = [
        OUTPUTS_DIR,
        PROJECT_ROOT / "outputs",
        Path("/kaggle/working/outputs"),
        Path("outputs").resolve(),
    ]

    # Return the first candidate that exists and contains files/subdirectories
    for cand in candidates:
        if cand.exists() and any(cand.iterdir()):
            return cand

    # If none have contents, return OUTPUTS_DIR if it exists, else candidates[0]
    for cand in candidates:
        if cand.exists():
            return cand

    return OUTPUTS_DIR


def export_artifacts(
    export_name: str = "outputs",
    export_dir: Path | str | None = None,
    zip_path: Path | str | None = None,
    model_prefix: str | None = None,
    source_dir: Path | str | None = None,
    include_subdirs: Sequence[str] = ("checkpoints", "metrics", "logs", "figures"),
    verbose: bool = True,
) -> dict[str, Any]:
    """Collects trained checkpoints, metrics, and logs, stages them in export_dir,

    and compresses them into a zip archive with integrity checks.
    """
    src_base = resolve_outputs_source_dir(source_dir)

    base_working = Path("/kaggle/working") if IN_KAGGLE else PROJECT_ROOT
    if export_dir is None:
        prefix_tag = model_prefix.lower() if model_prefix else "all"
        target_export_dir = base_working / f"export_{prefix_tag}"
    else:
        target_export_dir = Path(export_dir).resolve()

    if zip_path is None:
        target_zip = target_export_dir.parent / f"{export_name}.zip"
    else:
        target_zip = Path(zip_path).resolve()

    target_export_dir.mkdir(parents=True, exist_ok=True)
    for sub in include_subdirs:
        (target_export_dir / sub).mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info("Packaging artifacts from outputs source: %s", src_base)
        logger.info("Staging directory: %s", target_export_dir)

    copied_files: list[Path] = []
    prefix_filter = model_prefix.lower() if model_prefix else None

    # Check primary source and candidate fallbacks to ensure no files are missed
    fallback_sources = [src_base]
    for cand in [OUTPUTS_DIR, PROJECT_ROOT / "outputs", Path("/kaggle/working/outputs"), Path("outputs").resolve()]:
        if cand.exists() and cand not in fallback_sources:
            fallback_sources.append(cand)

    for sub in include_subdirs:
        dst_subdir = target_export_dir / sub
        seen_dest_names = {f.name for f in dst_subdir.iterdir()}

        for s_base in fallback_sources:
            src_subdir = s_base / sub
            if not src_subdir.exists():
                continue

            for item in sorted(src_subdir.iterdir()):
                if not item.is_file():
                    continue
                if item.name in seen_dest_names:
                    continue

                # Filter by model_prefix if specified
                include_file = True
                if prefix_filter:
                    name_lower = item.name.lower()
                    matches_prefix = prefix_filter in name_lower
                    # Allow general summary tables/logs to be bundled with any model export
                    is_summary = sub in ("metrics", "logs") and any(
                        s in name_lower for s in ["master", "summary", "benchmark", "report"]
                    )
                    include_file = matches_prefix or is_summary

                if include_file:
                    dest_file = dst_subdir / item.name
                    shutil.copy2(item, dest_file)
                    copied_files.append(dest_file)
                    seen_dest_names.add(item.name)

    if verbose:
        logger.info("Collected %d artifact file(s) into %s:", len(copied_files), target_export_dir)
        for cf in copied_files:
            sz_mb = cf.stat().st_size / (1024 * 1024)
            logger.info("  • %s (%.2f MB)", cf.relative_to(target_export_dir), sz_mb)

        if "checkpoints" in include_subdirs:
            ckpt_count = sum(1 for f in copied_files if f.suffix == ".pt")
            if ckpt_count == 0:
                logger.warning("No model checkpoints (.pt) were found to export!")
            else:
                logger.info("Found %d checkpoint file(s).", ckpt_count)

    # Create zip archive
    if target_zip.exists():
        target_zip.unlink()

    target_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(target_export_dir):
            for f in files:
                fp = Path(root) / f
                arcname = fp.relative_to(target_export_dir)
                zf.write(fp, arcname=arcname)

    zip_size_bytes = target_zip.stat().st_size if target_zip.exists() else 0
    zip_size_mb = zip_size_bytes / (1024 * 1024)

    if verbose:
        logger.info("%s created! Total archive size: %.2f MB", target_zip.name, zip_size_mb)
        logger.info("Archive path: %s", target_zip)

    return {
        "export_dir": target_export_dir,
        "zip_path": target_zip,
        "copied_files": copied_files,
        "total_files": len(copied_files),
        "zip_size_bytes": zip_size_bytes,
        "zip_size_mb": zip_size_mb,
    }


def import_artifacts(
    input_dir: Path | str = "/kaggle/input",
    target_dir: Path | str | None = None,
    mirror_to_local_outputs: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Scans input_dir for mounted notebook exports or zip archives,

    unpacking and copying them to target_dir (OUTPUTS_DIR).
    Optionally mirrors to ./outputs so relative paths in notebooks also resolve.
    """
    in_path = Path(input_dir).resolve()
    dst_path = Path(target_dir).resolve() if target_dir is not None else OUTPUTS_DIR.resolve()
    dst_path.mkdir(parents=True, exist_ok=True)

    for sub in ["checkpoints", "metrics", "logs", "figures"]:
        (dst_path / sub).mkdir(parents=True, exist_ok=True)

    imported_files: list[Path] = []
    if not in_path.exists():
        if verbose:
            logger.warning("Input directory does not exist: %s", in_path)
        return imported_files

    if verbose:
        logger.info("=== DISCOVERING ARTIFACTS IN %s ===", in_path)

    # 1. Look for and extract any zip archives (e.g. visreg_outputs.zip, nnunet_outputs.zip)
    for zip_file in in_path.glob("**/*.zip"):
        if verbose:
            logger.info("  Extracting zip archive: %s", zip_file.name)
        try:
            with zipfile.ZipFile(zip_file, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    parts = Path(member.filename).parts
                    # Map into subdirectories if member path matches standard names
                    matched_sub = next(
                        (p for p in parts if p in ["checkpoints", "metrics", "logs", "figures"]),
                        None,
                    )
                    if matched_sub:
                        idx = parts.index(matched_sub)
                        rel = Path(*parts[idx:])
                        out_target = dst_path / rel
                    else:
                        out_target = dst_path / member.filename

                    out_target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src_f, open(out_target, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                    imported_files.append(out_target)
        except Exception as e:
            if verbose:
                logger.warning("Failed to extract %s: %s", zip_file, e)

    # 2. Look for directory trees (e.g. export_visreg/checkpoints/*, outputs/*)
    for sub in ["checkpoints", "metrics", "logs", "figures"]:
        for match in in_path.glob(f"**/{sub}/*"):
            if match.is_file():
                dest = dst_path / sub / match.name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(match, dest)
                    imported_files.append(dest)

    # 3. Mirror to local ./outputs if requested and different from dst_path
    local_outputs = Path("outputs").resolve()
    if mirror_to_local_outputs and local_outputs != dst_path:
        local_outputs.mkdir(parents=True, exist_ok=True)
        for sub in ["checkpoints", "metrics", "logs", "figures"]:
            src_sub = dst_path / sub
            tgt_sub = local_outputs / sub
            tgt_sub.mkdir(parents=True, exist_ok=True)
            if src_sub.exists():
                for f in src_sub.iterdir():
                    if f.is_file():
                        mirror_f = tgt_sub / f.name
                        if not mirror_f.exists():
                            shutil.copy2(f, mirror_f)

    if verbose:
        ckpts = list((dst_path / "checkpoints").glob("*.pt"))
        metrics = list((dst_path / "metrics").iterdir()) if (dst_path / "metrics").exists() else []
        logger.info("Imported %d files to %s", len(imported_files), dst_path)
        logger.info("Available Checkpoints (%d): %s", len(ckpts), [c.name for c in ckpts])
        logger.info("Available Metrics (%d): %s", len(metrics), [m.name for m in metrics])

    return imported_files
