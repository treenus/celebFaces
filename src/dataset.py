"""Consume `data/folds.json`: per-fold train/test splits + PyTorch loaders.

The fold manifest (built by `src.make_folds`, seed 402) is the single source
of truth. This module NEVER re-shuffles anything — it only reads the manifest.

Image resolution policy (detection-failure fallback):
  1. Prefer the cached face crop  -> data/faces/<identity>/<filename>
  2. If no crop exists (YOLO missed the face, or detection hasn't run yet),
     fall back to the FULL raw image -> Celebrity Faces Dataset/<identity>/<filename>

Because crops keep the exact raw filenames, the fold assignment stays valid
regardless of which source an image comes from. `source_of()` reports which
bucket each image landed in so we can measure the fallback rate and decide
whether a secondary detector is needed.

Usage:
    from src.dataset import make_fold_loaders
    train_loader, test_loader, class_to_idx = make_fold_loaders(fold=0)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from .utils import (
    FACES_DIR,
    MANIFEST_PATH,
    RAW_DATA_DIR,
    SEED,
    load_json,
    load_manifest_or_raise,
)

IMAGE_SIZE = 224  # ResNet18 / AlexNet input

# ImageNet statistics (pretrained torchvision backbones expect these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_transforms(train: bool) -> transforms.Compose:
    """Train: light augmentation (small data -> flip + jitter + crop).
    Test: deterministic resize + center crop. No randomness at eval time.
    """
    if train:
        return transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Manifest / split helpers
# ---------------------------------------------------------------------------


def load_manifest() -> dict:
    return load_manifest_or_raise(MANIFEST_PATH)


def class_mapping(manifest: dict) -> dict[str, int]:
    """Stable identity -> label index, taken from the manifest order."""
    return {name: i for i, name in enumerate(manifest["identities"])}


def split_for_fold(manifest: dict, fold: int) -> tuple[list, list]:
    """Return (train_items, test_items) as lists of (identity, filename)."""
    n_folds = manifest["n_folds"]
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")

    train, test = [], []
    for identity in manifest["identities"]:
        for filename, f in sorted(manifest["fold_of"][identity].items()):
            (test if f == fold else train).append((identity, filename))
    return train, test


def source_of(identity: str, filename: str) -> tuple[Path, str]:
    """Resolve an image to (path, source) where source is 'crop' or 'full'."""
    crop = FACES_DIR / identity / filename
    if crop.exists():
        return crop, "crop"
    return RAW_DATA_DIR / identity / filename, "full"


def fallback_stats(items: list[tuple[str, str]]) -> dict:
    """Count crop vs full-image fallbacks for a list of (identity, filename)."""
    counts = {"crop": 0, "full": 0}
    for identity, filename in items:
        counts[source_of(identity, filename)[1]] += 1
    counts["fallback_rate"] = counts["full"] / max(len(items), 1)
    return counts


# ---------------------------------------------------------------------------
# Dataset / loaders
# ---------------------------------------------------------------------------


class FacesDataset(Dataset):
    """One fold-split of the celebrity faces, with crop-or-full fallback."""

    def __init__(
        self,
        items: list[tuple[str, str]],
        class_to_idx: dict[str, int],
        transform=None,
    ):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform if transform is not None else default_transforms(train=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        identity, filename = self.items[idx]
        path, _ = source_of(identity, filename)
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.class_to_idx[identity]


def _seed_worker(worker_id: int) -> None:
    """Deterministic per-worker seeding for DataLoader augmentation."""
    seed = torch.initial_seed() % (2**32)
    import random

    import numpy as np

    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


class EpochSeededDataset(Dataset):
    """Wraps a dataset so augmentation depends ONLY on (seed, epoch, index).

    torchvision transforms draw from the global torch RNG, which makes
    batches depend on iteration order/history. Instead we re-seed the RNG
    per sample from (base_seed, epoch, idx) — every sample's augmentation
    is then a pure function of its identity, fully reproducible no matter
    how loaders are interleaved or when epochs start.

    Call `set_epoch(e)` at the start of each training epoch so augmentation
    varies across epochs while staying deterministic.
    """

    def __init__(self, base: Dataset, base_seed: int):
        self.base = base
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        torch.manual_seed(self.base_seed + 100_000 * self.epoch + idx)
        return self.base[idx]


def make_fold_loaders(
    fold: int,
    batch_size: int = 32,
    num_workers: int = 0,
    manifest: dict | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Build (train_loader, test_loader, class_to_idx) for one LOOCV round.

    Defaults tuned for an 8 GB unified-memory Mac: batch 32, workers 0
    (worker processes duplicate memory; 0 keeps the footprint small and
    is fully deterministic).
    """
    manifest = manifest if manifest is not None else load_manifest()
    c2i = class_mapping(manifest)
    train_items, test_items = split_for_fold(manifest, fold)

    train_ds = EpochSeededDataset(
        FacesDataset(train_items, c2i, default_transforms(train=True)),
        base_seed=SEED + 1000 * fold,
    )
    test_ds = FacesDataset(test_items, c2i, default_transforms(train=False))

    g = torch.Generator()
    g.manual_seed(SEED + fold)  # per-fold shuffle, still fully reproducible

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader, c2i


def main() -> None:
    """Sanity-check every fold: sizes, balance, and fallback rate."""
    manifest = load_manifest()
    c2i = class_mapping(manifest)
    print(f"Manifest: seed={manifest['seed']} folds={manifest['n_folds']} classes={len(c2i)}")
    for fold in range(manifest["n_folds"]):
        train, test = split_for_fold(manifest, fold)
        assert len(train) == manifest["train_size"], (fold, len(train))
        assert len(test) == manifest["test_size"], (fold, len(test))
        # Every identity contributes exactly 10 test images.
        per_class = [0] * len(c2i)
        for identity, _ in test:
            per_class[c2i[identity]] += 1
        assert per_class == [10] * len(c2i), (fold, per_class)
        stats = fallback_stats(train + test)
        print(
            f"fold {fold}: train={len(train)} test={len(test)} "
            f"crop={stats['crop']} full-fallback={stats['full']} "
            f"({stats['fallback_rate']:.1%})"
        )
    print("All folds OK.")


if __name__ == "__main__":
    main()