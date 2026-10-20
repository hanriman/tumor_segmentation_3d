import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Root directory of thesis_3d
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Detect Google Colab execution environment
IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()

DEFAULT_NUM_WORKERS = int(
    os.environ.get(
        "BRATS3D_NUM_WORKERS", "2" if not sys.platform.startswith("darwin") else "0"
    )
)
DEFAULT_SEED = int(os.environ.get("BRATS3D_SEED", "42"))

# Standard directory locations
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "brats_gli_3d"


def get_dataset_dir(dataset_name: str = "brats_gli_3d") -> Path:
    """Dynamically resolves the path to a 3D dataset directory across Colab, env vars, and local storage."""
    # 1. Environment variable overrides
    env_key = f"BRATS_{dataset_name.upper()}_DIR"
    if os.environ.get(env_key):
        p = Path(os.environ[env_key]).resolve()
        if p.exists():
            return p
    if os.environ.get("BRATS3D_DATA_DIR"):
        p = Path(os.environ["BRATS3D_DATA_DIR"]).resolve()
        if p.exists():
            return p
    if os.environ.get("BRATS_DATA_DIR"):
        p = (Path(os.environ["BRATS_DATA_DIR"]) / dataset_name).resolve()
        if p.exists():
            return p

    # 2. Check Google Colab local & Drive mounts
    if Path("/content").exists():
        colab_candidates = [
            Path("/content") / "data" / "processed" / dataset_name,
            Path("/content") / "data" / "processed" / "brats_gli_3d_v2",
            Path("/content") / dataset_name,
            Path("/content") / "brats_3d_datasets" / dataset_name,
            Path("/content/drive/MyDrive") / "thesis_3d" / "data" / "processed" / dataset_name,
            Path("/content/drive/MyDrive") / dataset_name,
        ]
        for cand in colab_candidates:
            if (cand / "metadata.csv").exists():
                return cand.resolve()
        colab_matches = list(Path("/content").glob(f"**/{dataset_name}"))
        if colab_matches and (colab_matches[0] / "metadata.csv").exists():
            return colab_matches[0].resolve()

    # 3. Local project processed dataset directory
    local_variants = ["brats_gli_3d_v2", "brats_gli_3d_full", dataset_name]
    for variant in local_variants:
        cand = (DATA_DIR / "processed" / variant).resolve()
        if cand.is_dir() and (cand / "metadata.csv").exists():
            return cand

    return (DATA_DIR / "processed" / dataset_name).resolve()


def get_metadata_path(dataset_name: str = "brats_gli_3d") -> Path:
    """Returns the resolved metadata.csv path for a given dataset."""
    return get_dataset_dir(dataset_name) / "metadata.csv"


# Output directory resolution
if IN_COLAB:
    drive_out = Path("/content/drive/MyDrive/thesis_3d_outputs")
    OUTPUTS_DIR = drive_out if Path("/content/drive/MyDrive").exists() else Path("/content/outputs")
else:
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CONFIGS_DIR = PROJECT_ROOT / "configs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_default_output_dir(
    explicit_output_dir: str | Path | None = None,
    experiment_name: str | None = None,
    is_smoke_test: bool = False,
) -> Path:
    """
    Intelligently resolves the output directory:
    1. If explicit_output_dir is provided -> use it directly.
    2. If running under test/smoke-test (pytest active, BRATS3D_TEST_MODE set, or is_smoke_test)
       -> automatically isolates to `tests/test_outputs/.smoke_tmp` (ZERO output flags needed).
    3. If standard experiment -> defaults to `outputs/experiments/<experiment_name>`
       (never pollutes the `outputs/` root).
    """
    if explicit_output_dir:
        return Path(explicit_output_dir).resolve()

    # Detect test / smoke test execution
    is_test_env = (
        is_smoke_test
        or ("pytest" in sys.modules)
        or ("PYTEST_CURRENT_TEST" in os.environ)
        or os.environ.get("BRATS3D_TEST_MODE") == "1"
    )
    if is_test_env:
        return (PROJECT_ROOT / "tests" / "test_outputs" / ".smoke_tmp").resolve()

    # Real run: default to dedicated experiment subfolder
    subfolder = experiment_name or "default_experiment"
    return (OUTPUTS_DIR / "experiments" / subfolder).resolve()


def ensure_directories(base_output_dir: Path | str | None = None):
    """Creates required project output directories if they do not exist."""
    out_dir = Path(base_output_dir).resolve() if base_output_dir else OUTPUTS_DIR
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    if out_dir == OUTPUTS_DIR.resolve():
        # Top-level outputs directory should only ever contain experiments and reports
        (OUTPUTS_DIR / "experiments").mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "reports").mkdir(parents=True, exist_ok=True)
    else:
        for sub in ["checkpoints", "figures", "metrics", "logs"]:
            (out_dir / sub).mkdir(parents=True, exist_ok=True)


def get_checkpoint_search_dirs(primary_dir: Path | str | None = None) -> list[Path]:
    """Returns candidate checkpoint search directories in priority order across Colab & local layouts."""
    primary = Path(primary_dir).resolve() if primary_dir else None
    candidates: list[Path] = []

    # If an explicit custom primary directory was provided (not the default CHECKPOINTS_DIR), prioritize it
    if primary is not None and primary != CHECKPOINTS_DIR.resolve():
        candidates.append(primary)

    # Search in experiment subdirectories first (where fully trained checkpoints reside)
    search_roots = [
        OUTPUTS_DIR,
        PROJECT_ROOT / "outputs",
        Path.cwd() / "outputs",
        Path("outputs").resolve(),
    ]
    if IN_COLAB:
        search_roots = [
            Path("/content/drive/MyDrive/thesis/pretrain model"),
            Path("/content/drive/MyDrive/thesis/visreg_pretrain_outputs"),
            Path("/content/drive/MyDrive/thesis/downstream"),
            Path("/content/drive/MyDrive/thesis"),
            Path("/content/outputs"),
            Path("/content/drive/MyDrive/thesis_3d_outputs"),
        ] + search_roots

    for root in search_roots:
        if root.exists():
            if (root / "checkpoints").is_dir():
                candidates.append((root / "checkpoints").resolve())
            candidates.append(root.resolve())
            exp_dir = root / "experiments"
            if exp_dir.exists():
                for sub in sorted(exp_dir.glob("*/checkpoints")):
                    if sub.is_dir():
                        candidates.append(sub.resolve())

    # Then search in standard checkpoints directories
    candidates.extend([
        CHECKPOINTS_DIR,
        Path("outputs/checkpoints").resolve(),
        Path.cwd() / "outputs" / "checkpoints",
        PROJECT_ROOT / "outputs" / "checkpoints",
    ])
    if primary is not None:
        candidates.append(primary)

    seen: set[Path] = set()
    result: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            if c.exists():
                result.append(c)
    if not result:
        result.append(primary if primary else CHECKPOINTS_DIR)
    return result


def merge_config_with_args(
    config: dict[str, Any],
    args: Any,
    cli_args: list[str] | None = None,
) -> Any:
    """
    Merges a loaded YAML configuration dictionary with an argparse Namespace.
    Explicit command-line arguments take precedence over YAML values.
    """
    if cli_args is None:
        cli_args = sys.argv[1:]

    explicit_cli_flags: set[str] = set()
    for arg in cli_args:
        if arg.startswith("--"):
            flag = arg.lstrip("-").split("=")[0].replace("-", "_")
            explicit_cli_flags.add(flag)
            if flag.startswith("no_"):
                explicit_cli_flags.add(flag[3:])
        elif arg.startswith("-"):
            flag = arg.lstrip("-").split("=")[0].replace("-", "_")
            explicit_cli_flags.add(flag)

    args_dict = vars(args) if hasattr(args, "__dict__") else args

    def apply_kv(key: str, val: Any) -> None:
        target_keys = [key]
        if key == "name":
            target_keys.append("model_type")
        elif key == "model_type":
            target_keys.append("name")

        if any(k in explicit_cli_flags for k in target_keys):
            return

        for k in target_keys:
            if hasattr(args, k) or (isinstance(args_dict, dict) and k in args_dict):
                setattr(args, k, val)
        setattr(args, key, val)

    for k, v in config.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                apply_kv(sub_k, sub_v)
            setattr(args, k, v)
        else:
            apply_kv(k, v)

    if "BRATS3D_SEED" in os.environ and "seed" not in explicit_cli_flags:
        apply_kv("seed", int(os.environ["BRATS3D_SEED"]))

    if "BRATS3D_NUM_WORKERS" in os.environ and "num_workers" not in explicit_cli_flags:
        apply_kv("num_workers", int(os.environ["BRATS3D_NUM_WORKERS"]))

    return args
