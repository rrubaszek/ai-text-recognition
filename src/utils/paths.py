from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATASET_DIR: Final[Path] = PROJECT_ROOT / "2025-smigiel" / "data"
STYLOMETRY_DATASET_DIR: Final[Path] = PROJECT_ROOT / "stylometric_dataset"