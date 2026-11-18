"""Generate the 10-fold leave-one-out cross-validation (LOOCV) split.

Spec (from Final.pdf):
  - Each identity has 100 images.
  - Split each identity's 100 images into 10 folds of 10.
  - Train on 9 folds  -> 9 * 10 * 17 = 1530 images.
  - Test  on 1 fold   -> 1 * 10 * 17 = 170 images.
  - Repeat until every fold has served as the test set (10 rounds).
  - Average the 10 accuracies -> overall LOOCV accuracy.

Because every identity contributes exactly 10 images to every fold, the split
is inherently stratified: each fold is a balanced 170-image test set.

The assignment is written to `data/folds.json` so that training/eval never
re-derive it (and therefore can never drift between runs).

Usage:
    python -m src.make_folds
"""

from __future__ import annotations

import random
from collections import defaultdict

from .utils import (
    MANIFEST_PATH,
    SEED,
    ensure_dirs,
    list_identities,
    list_images,
    save_json,
    set_seed,
)

N_FOLDS = 10


def build_folds(seed: int = SEED) -> dict:
    """Return the fold manifest as a plain dict (JSON-serialisable)."""
    set_seed(seed)
    identities = list_identities()

    # fold_of[identity][filename] = fold index in [0, N_FOLDS)
    fold_of: dict[str, dict[str, int]] = {}
    per_fold_counts: dict[str, list[int]] = {}

    for identity in identities:
        images = list_images(identity)
        if len(images) != 100:
            raise ValueError(
                f"{identity!r} has {len(images)} images, expected 100"
            )

        # Shuffle with the shared seed, then deal round-robin into 10 folds.
        # Round-robin guarantees exactly 10 images per fold per identity.
        names = [p.name for p in images]
        random.shuffle(names)

        assignment = {name: i % N_FOLDS for i, name in enumerate(names)}
        fold_of[identity] = assignment

        counts = [0] * N_FOLDS
        for f in assignment.values():
            counts[f] += 1
        per_fold_counts[identity] = counts

    # Sanity checks ---------------------------------------------------------
    for identity, counts in per_fold_counts.items():
        assert counts == [10] * N_FOLDS, (identity, counts)

    manifest = {
        "seed": seed,
        "n_folds": N_FOLDS,
        "n_identities": len(identities),
        "identities": identities,
        "images_per_identity": 100,
        "train_size": 9 * 10 * len(identities),   # 1530
        "test_size": 1 * 10 * len(identities),    # 170
        "fold_of": fold_of,
    }
    return manifest


def summarize(manifest: dict) -> None:
    """Print a human-readable summary of the split."""
    n_folds = manifest["n_folds"]
    identities = manifest["identities"]
    fold_of = manifest["fold_of"]

    print(f"Seed: {manifest['seed']}")
    print(f"Identities: {manifest['n_identities']}")
    print(f"Train size per round: {manifest['train_size']}")
    print(f"Test size per round:  {manifest['test_size']}")
    print()

    # Per-fold totals across all identities.
    totals = defaultdict(int)
    for identity in identities:
        for fold in fold_of[identity].values():
            totals[fold] += 1

    print(f"{'fold':>4} | {'test':>5} | {'train':>5}")
    print("-" * 20)
    for f in range(n_folds):
        test = totals[f]
        print(f"{f:>4} | {test:>5} | {manifest['train_size']:>5}")


def main() -> None:
    ensure_dirs()
    manifest = build_folds()
    save_json(manifest, MANIFEST_PATH)
    summarize(manifest)
    print(f"\nWrote manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
