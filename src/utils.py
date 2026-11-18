"""Shared utilities: paths, seeding, and small helpers.

Everything that touches randomness must go through `set_seed` so that the
graders' re-runs reproduce our reported numbers.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/utils.py -> project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "Celebrity Faces Dataset"
DATA_DIR = PROJECT_ROOT / "data"
FACES_DIR = DATA_DIR / "faces"          # cached YOLO crops (generated later)
RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_PATH = DATA_DIR / "folds.json"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# Single source of truth for the seed. Logged into the manifest and the report.
SEED = 402


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch not installed yet (e.g. during fold generation) - fine.
        pass


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def list_identities() -> list[str]:
    """Return the sorted list of celebrity identities (class names)."""
    return sorted(
        d.name
        for d in RAW_DATA_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def list_images(identity: str) -> list[Path]:
    """Return the sorted list of image paths for one identity."""
    folder = RAW_DATA_DIR / identity
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def ensure_dirs() -> None:
    for d in (DATA_DIR, FACES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_manifest_or_raise(path: Path = MANIFEST_PATH) -> dict:
    """Load the fold manifest, with a clear error if it hasn't been built."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fold manifest not found at {path}. "
            "Run `python -m src.make_folds` first."
        )
    return load_json(path)
