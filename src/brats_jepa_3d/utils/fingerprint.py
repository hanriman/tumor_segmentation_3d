"""Dataset pool fingerprinting: guards against stale-checkpoint contamination.

A checkpoint carries the fingerprint of the pool it was trained on
(metadata sha + row/split counts). Loaders compare it against the current pool
and warn loudly on mismatch or absence (legacy files) — pretraining on volumes
that are test in the current split is silent leakage into reported numbers.
"""
import hashlib
from pathlib import Path

import pandas as pd


def dataset_fingerprint(data_dir: str | Path) -> dict:
    """Fingerprints a processed pool via its metadata.csv (stable, cheap)."""
    meta = Path(data_dir) / "metadata.csv"
    if not meta.exists():
        return {"metadata": None}
    raw = meta.read_bytes()
    try:
        df = pd.read_csv(meta)
        splits = df["split"].value_counts().to_dict() if "split" in df.columns else {}
        n_rows = len(df)
    except Exception:
        splits, n_rows = {}, -1
    return {
        "metadata_sha1": hashlib.sha1(raw).hexdigest()[:12],
        "n_rows": n_rows,
        "splits": {str(k): int(v) for k, v in splits.items()},
    }


def check_pool_match(ckpt_fp: dict | None, current_fp: dict, context: str) -> str:
    """Compares checkpoint pool fingerprint against the current pool.

    Returns "match" | "mismatch" | "unverifiable" (legacy checkpoint without a
    fingerprint). Callers log the result; mismatches must block reported runs.
    """
    import logging

    log = logging.getLogger(__name__)
    if not ckpt_fp or not ckpt_fp.get("metadata_sha1"):
        log.warning(
            f"[{context}] checkpoint carries no pool fingerprint (legacy file): "
            f"pretraining pool UNVERIFIED — do not report test numbers from this run."
        )
        return "unverifiable"
    if ckpt_fp.get("metadata_sha1") != current_fp.get("metadata_sha1"):
        log.warning(
            f"[{context}] pool MISMATCH: checkpoint trained on {ckpt_fp} vs "
            f"current pool {current_fp} — stale weights may leak test volumes."
        )
        return "mismatch"
    return "match"
