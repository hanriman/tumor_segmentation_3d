"""
Unified 3D Segmentation Model Factory and Checkpoint Discovery.

Consolidates model construction, patch-size normalization, and checkpoint
discovery/loading across training (`train_downstream_3d.py`, `train_unet_3d.py`,
`train_nnunet_3d.py`, `evaluate_low_data_3d.py`) and evaluation (`evaluate_3d.py`,
`evaluate_ood_3d.py`).
"""

import logging
from pathlib import Path
from typing import Any

import torch

from brats_jepa_3d.config import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    get_checkpoint_search_dirs,
    get_dataset_dir,
    load_yaml_config,
)
from brats_jepa_3d.utils import (
    check_pool_match,
    dataset_fingerprint,
    sort_checkpoints_by_epoch,
)

from .nnunet_3d import BraTS3DnnUNet
from .segmentation_head_3d import JEPASegmentationModel3D, load_downstream_state_dict
from .unet_3d import BraTS3DUNet

MODEL_ALIAS_MAP: dict[str, str] = {
    "visreg": "visreg_jepa",
    "visreg_jepa": "visreg_jepa",
    "sigreg": "sigreg_jepa",
    "sigreg_jepa": "sigreg_jepa",
    "unet": "unet_3d",
    "unet_3d": "unet_3d",
    "nnunet": "nnunet_3d",
    "nnunet_3d": "nnunet_3d",
    "scratch": "vit_scratch",
    "vit_scratch": "vit_scratch",
    "frozen": "vit_frozen",
    "vit_frozen": "vit_frozen",
    "all": "all",
    "all_models": "all",
}


def resolve_model_alias(model_type: str | None) -> str:
    """Normalizes a user-supplied model_type string to its canonical identifier."""
    if not model_type:
        return "all"
    key = model_type.strip().lower()
    return MODEL_ALIAS_MAP.get(key, key)


def normalize_patch_size(
    raw_patch: int | list[int] | tuple[int, ...] | None,
    default: tuple[int, int, int] = (8, 8, 8),
) -> tuple[int, int, int]:
    """Normalizes an integer, single-element sequence, or 3-tuple into a 3D patch_size tuple."""
    if raw_patch is None:
        return (int(default[0]), int(default[1]), int(default[2]))
    if isinstance(raw_patch, int):
        return (raw_patch, raw_patch, raw_patch)
    if len(raw_patch) == 1:
        p = int(raw_patch[0])
        return (p, p, p)
    return (int(raw_patch[0]), int(raw_patch[1]), int(raw_patch[2]))


def find_ssl_checkpoint(
    model_type: str,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    explicit_path: str | Path | None = None,
) -> Path | None:
    """Locates the best pre-trained SSL encoder checkpoint for `model_type`."""
    if explicit_path is not None:
        return Path(explicit_path)

    search_dirs = get_checkpoint_search_dirs(checkpoints_dir)
    excluded_tags = ("multiscale", "unetr_hybrid", "hybrid", "downstream", "scratch", "frozen")

    for sdir in search_dirs:
        if not sdir.exists():
            continue
        ssl_best = [
            p
            for p in sorted(sdir.glob(f"{model_type}*_3d_best.pt"))
            if not any(tag in p.name.lower() for tag in ("scratch", "frozen"))
        ]
        if ssl_best:
            return ssl_best[-1]

    for sdir in search_dirs:
        if not sdir.exists():
            continue
        ssl_epochs = [
            p
            for p in sort_checkpoints_by_epoch(list(sdir.glob(f"{model_type}*_epoch_*.pt")))
            if not any(tag in p.name.lower() for tag in ("scratch", "frozen"))
        ]
        if ssl_epochs:
            return ssl_epochs[-1]

    for sdir in search_dirs:
        if not sdir.exists():
            continue
        candidates = [
            p
            for p in sorted(sdir.glob(f"{model_type}*.pt"))
            if not any(tag in p.name.lower() for tag in excluded_tags)
        ]
        if candidates:
            return candidates[-1]

    return None


def find_downstream_checkpoint(
    model_type: str,
    decoder_type: str = "multiscale",
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    is_scratch: bool = False,
    is_frozen: bool = False,
    explicit_path: str | Path | None = None,
) -> Path:
    """Locates the best downstream segmentation checkpoint for evaluation."""
    if explicit_path is not None:
        return Path(explicit_path)

    search_dirs = get_checkpoint_search_dirs(checkpoints_dir)
    if model_type == "unet_3d":
        return next(
            (d / "unet_3d_best.pt" for d in search_dirs if (d / "unet_3d_best.pt").exists()),
            checkpoints_dir / "unet_3d_best.pt",
        )
    if model_type == "nnunet_3d":
        return next(
            (d / "nnunet_3d_best.pt" for d in search_dirs if (d / "nnunet_3d_best.pt").exists()),
            checkpoints_dir / "nnunet_3d_best.pt",
        )

    if is_scratch:
        patterns = [
            f"{model_type}_{decoder_type}_scratch_best.pt",
            f"*{model_type}*{decoder_type}*scratch*best.pt",
            f"*{decoder_type}*scratch*best.pt",
            f"*{model_type}*scratch*best.pt",
            "*scratch*best.pt",
        ]
        for pat in patterns:
            for d in search_dirs:
                if d.exists():
                    cands = sorted(d.glob(pat))
                    if cands:
                        return cands[-1]
        return checkpoints_dir / f"{model_type}_{decoder_type}_scratch_best.pt"

    if is_frozen:
        patterns = [
            f"{model_type}_{decoder_type}_frozen_best.pt",
            f"*{model_type}*{decoder_type}*frozen*best.pt",
            f"*{decoder_type}*frozen*best.pt",
            f"*{model_type}*frozen*best.pt",
            "*frozen*best.pt",
        ]
        for pat in patterns:
            for d in search_dirs:
                if d.exists():
                    cands = sorted(d.glob(pat))
                    if cands:
                        return cands[-1]
        return checkpoints_dir / f"{model_type}_{decoder_type}_frozen_best.pt"

    patterns = [
        f"{model_type}_{decoder_type}_best.pt",
        f"*{model_type}*{decoder_type}*best.pt",
        f"*{decoder_type}*best.pt",
        f"{model_type}*best.pt",
    ]
    for pat in patterns:
        for d in search_dirs:
            if d.exists():
                cands = [
                    p
                    for p in sorted(d.glob(pat))
                    if not any(x in p.name.lower() for x in ("scratch", "frozen"))
                ]
                if cands:
                    return cands[-1]
    return checkpoints_dir / f"{model_type}_{decoder_type}_best.pt"


def _peek_deep_supervision(ckpt_file: Path, default: bool = False) -> bool:
    """Detects whether a saved JEPA segmentation checkpoint contains deep supervision heads."""
    if not ckpt_file.exists():
        return default
    try:
        sd_peek = torch.load(ckpt_file, map_location="cpu")
        sd_peek = sd_peek.get("model_state_dict", sd_peek)
        if isinstance(sd_peek, dict) and any(k.startswith("decoder.ds") for k in sd_peek):
            return True
    except (KeyError, OSError, RuntimeError, AttributeError):
        pass
    return default


def build_segmentation_model_for_training(
    model_type: str,
    device: torch.device,
    decoder_type: str = "multiscale",
    from_scratch: bool = False,
    freeze_encoder: bool = False,
    deep_supervision: bool = True,
    patch_size: int | list[int] | tuple[int, ...] | None = None,
    out_channels: int | None = None,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    pretrained_checkpoint: str | Path | None = None,
    resolved_data_dir: Path | None = None,
    logger: logging.Logger | None = None,
    require_ssl_for_pretrain: bool = False,
    model_overrides: dict[str, Any] | None = None,
    label: str | None = None,
) -> tuple[torch.nn.Module, int] | None:
    """Constructs a 3D segmentation model for downstream fine-tuning or low-data training.

    Returns `(model, out_channels)` or `None` when `require_ssl_for_pretrain=True`
    and no pre-trained SSL checkpoint is available.
    """
    overrides = model_overrides or {}

    if model_type == "unet_3d":
        unet_cfg_path = CONFIGS_DIR / "model" / "unet_3d.yaml"
        cfg = load_yaml_config(unet_cfg_path) if unet_cfg_path.exists() else {}
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        n_classes = int(out_channels if out_channels is not None else cfg.get("out_channels", 5))
        model = BraTS3DUNet(
            in_channels=int(cfg.get("in_channels", 4)),
            out_channels=n_classes,
            channels=tuple(cfg.get("channels", (32, 64, 128, 256, 512))),
            strides=tuple(cfg.get("strides", (2, 2, 2, 2))),
            num_res_units=int(cfg.get("num_res_units", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            deep_supervision=deep_supervision,
        ).to(device)
        return model, n_classes

    if model_type == "nnunet_3d":
        nnunet_cfg_path = CONFIGS_DIR / "model" / "nnunet_3d.yaml"
        cfg = load_yaml_config(nnunet_cfg_path) if nnunet_cfg_path.exists() else {}
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        n_classes = int(out_channels if out_channels is not None else cfg.get("out_channels", 5))
        model = BraTS3DnnUNet(
            in_channels=int(cfg.get("in_channels", 4)),
            out_channels=n_classes,
            deep_supervision=deep_supervision,
            deep_supr_num=int(cfg.get("deep_supr_num", 3)),
            res_block=bool(cfg.get("res_block", True)),
        ).to(device)
        return model, n_classes

    # JEPA / ViT Segmentation Model
    cfg_prefix = "visreg_jepa" if model_type in ("vit_scratch", "vit_frozen") else model_type
    jepa_cfg_path = CONFIGS_DIR / "model" / f"{cfg_prefix}_3d.yaml"
    cfg = load_yaml_config(jepa_cfg_path) if jepa_cfg_path.exists() else {}
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    n_classes = int(out_channels if out_channels is not None else cfg.get("out_channels", 5))
    resolved_patch = normalize_patch_size(patch_size, default=tuple(cfg.get("patch_size", (8, 8, 8))))
    display = label or model_type

    ckpt_path: Path | None = None
    if from_scratch:
        if logger:
            logger.info(
                f"Initializing {display} from SCRATCH (random initialization, no pre-trained checkpoint) [--from_scratch active]"
            )
    else:
        ckpt_path = find_ssl_checkpoint(cfg_prefix, checkpoints_dir=checkpoints_dir, explicit_path=pretrained_checkpoint)
        if (ckpt_path is None or not ckpt_path.exists()) and require_ssl_for_pretrain:
            if logger:
                logger.warning(
                    f"No pre-trained SSL weights found for {display} ({cfg_prefix}_3d_best.pt / {cfg_prefix}*_epoch_*.pt). "
                    f"Skipping evaluation."
                )
            return None

    model = JEPASegmentationModel3D(
        img_size=tuple(cfg.get("spatial_shape", (128, 128, 128))),
        patch_size=resolved_patch,
        in_channels=int(cfg.get("in_channels", 4)),
        embed_dim=int(cfg.get("embed_dim", 384)),
        encoder_depth=int(cfg.get("encoder_depth", 8)),
        num_heads=int(cfg.get("num_heads", 6)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        out_channels=n_classes,
        freeze_encoder=freeze_encoder,
        decoder_type=decoder_type,
        deep_supervision=deep_supervision,
    ).to(device)

    if not from_scratch and ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        check_pool_match(
            ckpt.get("pool_fingerprint"),
            dataset_fingerprint(resolved_data_dir or get_dataset_dir()),
            f"downstream/{display}",
        )
        res = model.load_pretrained_encoder(ckpt)
        if logger:
            logger.info(
                f"Loaded pre-trained encoder weights from {ckpt_path} ({res['loaded_keys']} keys matched)"
            )
    elif not from_scratch:
        if freeze_encoder:
            raise FileNotFoundError(
                f"Cannot freeze encoder without a pre-trained checkpoint! "
                f"No pre-trained checkpoint found for '{cfg_prefix}' (specified: '{pretrained_checkpoint}')."
            )
        if logger:
            logger.warning(
                "No pre-trained checkpoint specified or found. Training encoder from random initialization."
            )

    return model, n_classes


def load_segmentation_model_for_eval(
    model_type: str,
    label: str,
    device: torch.device,
    decoder_type: str = "multiscale",
    is_scratch: bool = False,
    is_frozen: bool = False,
    patch_size: int | list[int] | tuple[int, ...] | None = None,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    explicit_checkpoint: str | Path | None = None,
    resolved_data_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> torch.nn.Module | None:
    """Discovers and loads a trained 3D segmentation checkpoint for evaluation."""
    cfg_prefix = "visreg_jepa" if model_type in ("vit_scratch", "vit_frozen") else model_type
    ckpt_file = find_downstream_checkpoint(
        model_type=cfg_prefix,
        decoder_type=decoder_type,
        checkpoints_dir=checkpoints_dir,
        is_scratch=is_scratch,
        is_frozen=is_frozen,
        explicit_path=explicit_checkpoint,
    )

    if not ckpt_file.exists():
        if logger:
            logger.warning(f"Checkpoint for {label} not found at {ckpt_file}. Skipping evaluation.")
        return None

    if cfg_prefix == "unet_3d":
        unet_cfg_path = CONFIGS_DIR / "model" / "unet_3d.yaml"
        unet_cfg = load_yaml_config(unet_cfg_path) if unet_cfg_path.exists() else {}
        model: torch.nn.Module = BraTS3DUNet(
            in_channels=unet_cfg.get("in_channels", 4),
            out_channels=unet_cfg.get("out_channels", 5),
            channels=tuple(unet_cfg.get("channels", (32, 64, 128, 256, 512))),
            strides=tuple(unet_cfg.get("strides", (2, 2, 2, 2))),
            num_res_units=unet_cfg.get("num_res_units", 2),
            dropout=unet_cfg.get("dropout", 0.1),
        ).to(device)
        sd = torch.load(ckpt_file, map_location=device)
        sd = sd.get("model_state_dict", sd)
        missing, _unexpected = model.load_state_dict(sd, strict=False)
        matched = [k for k in model.state_dict() if k not in missing]
        if len(matched) == 0:
            raise RuntimeError(f"Failed to load any weights for {label} from {ckpt_file}")
        if logger:
            logger.info(f"Loaded {label} weights from {ckpt_file.name} ({len(matched)} matched keys)")
        return model

    if cfg_prefix == "nnunet_3d":
        nnunet_cfg_path = CONFIGS_DIR / "model" / "nnunet_3d.yaml"
        nnunet_cfg = load_yaml_config(nnunet_cfg_path) if nnunet_cfg_path.exists() else {}
        model = BraTS3DnnUNet(
            in_channels=nnunet_cfg.get("in_channels", 4),
            out_channels=nnunet_cfg.get("out_channels", 5),
            deep_supervision=nnunet_cfg.get("deep_supervision", True),
            deep_supr_num=nnunet_cfg.get("deep_supr_num", 3),
            res_block=nnunet_cfg.get("res_block", True),
        ).to(device)
        sd = torch.load(ckpt_file, map_location=device)
        sd = sd.get("model_state_dict", sd)
        missing, _unexpected = model.load_state_dict(sd, strict=False)
        matched = [k for k in model.state_dict() if k not in missing]
        if len(matched) == 0:
            raise RuntimeError(f"Failed to load any weights for {label} from {ckpt_file}")
        if logger:
            logger.info(f"Loaded {label} weights from {ckpt_file.name} ({len(matched)} matched keys)")
        return model

    # JEPA / ViT Downstream Model
    jepa_cfg_path = CONFIGS_DIR / "model" / f"{cfg_prefix}_3d.yaml"
    jepa_cfg = load_yaml_config(jepa_cfg_path) if jepa_cfg_path.exists() else {}
    ds_flag = _peek_deep_supervision(ckpt_file, default=False)
    resolved_patch = normalize_patch_size(patch_size, default=tuple(jepa_cfg.get("patch_size", (8, 8, 8))))

    model = JEPASegmentationModel3D(
        img_size=tuple(jepa_cfg.get("spatial_shape", (128, 128, 128))),
        patch_size=resolved_patch,
        in_channels=jepa_cfg.get("in_channels", 4),
        embed_dim=jepa_cfg.get("embed_dim", 384),
        encoder_depth=jepa_cfg.get("encoder_depth", 8),
        num_heads=jepa_cfg.get("num_heads", 6),
        mlp_ratio=jepa_cfg.get("mlp_ratio", 4.0),
        out_channels=jepa_cfg.get("out_channels", 5),
        decoder_type=decoder_type or "multiscale",
        deep_supervision=ds_flag,
        freeze_encoder=is_frozen,
    ).to(device)

    ckpt_raw = torch.load(ckpt_file, map_location=device)
    if isinstance(ckpt_raw, dict) and "pool_fingerprint" in ckpt_raw:
        check_pool_match(
            ckpt_raw.get("pool_fingerprint"),
            dataset_fingerprint(resolved_data_dir or get_dataset_dir()),
            f"eval/{label}",
        )
    sd = ckpt_raw.get("model_state_dict", ckpt_raw) if isinstance(ckpt_raw, dict) else ckpt_raw
    has_downstream = any(k.startswith("encoder.") for k in sd) or any(k.startswith("decoder.") for k in sd)
    has_pretrain = any(k.startswith("context_encoder.") for k in sd)
    if has_pretrain and not has_downstream:
        raise RuntimeError(
            f"{label}: Checkpoint {ckpt_file.name} contains only pre-trained encoder weights. "
            f"Evaluating a randomly initialized downstream decoder produces invalid benchmark numbers. "
            f"Train downstream model with train_downstream_3d.py first."
        )

    missing, _unexpected = load_downstream_state_dict(model, sd, label)
    matched = [k for k in model.state_dict() if k not in missing]
    if logger:
        logger.info(f"Loaded {label} weights from {ckpt_file.name} ({len(matched)} matched keys)")
    return model
