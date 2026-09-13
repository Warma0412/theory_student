import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RAW_DATA_ROOT = DATA_ROOT / "raw"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
DOCS_ROOT = PROJECT_ROOT / "docs"


def get_data_root() -> Path:
    override = os.environ.get("LASSONET_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_RAW_DATA_ROOT


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
