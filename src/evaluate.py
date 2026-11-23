"""LOOCV evaluation: aggregate per-fold predictions -> accuracy + confusion matrix.

Each fold's best checkpoint (results/checkpoints/resnet18_foldN.pt) predicts
its own 170-image test split. Every image is predicted exactly once, by a
model that never saw it in training -> the pooled 1700 predictions form the
LOOCV confusion matrix.

Also breaks errors down by detection bucket (haar / yolo / full-image
fallback) to test whether misses/pose correlate with errors.

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .dataset import make_fold_loaders, source_of, split_for_fold, load_manifest
from .train import build_model, evaluate, pick_device
from .utils import RESULTS_DIR, load_json, save_json

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


def collect_predictions(device: torch.device | None = None, checkpoint_dir=None) -> dict:
    """Run every fold's checkpoint on its test split; pool the predictions."""
    device = device or pick_device()
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    manifest = load_manifest()
    identities = manifest["identities"]

    all_preds, all_labels, records = [], [], []
    fold_accs = []

    for fold in range(manifest["n_folds"]):
        ckpt_path = checkpoint_dir / f"resnet18_fold{fold}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {ckpt_path}. Run `python -m src.train --fold {fold}` first."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(len(identities))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        _, test_loader, c2i = make_fold_loaders(fold)
        acc, preds, labels = evaluate(model, test_loader, device)
        fold_accs.append(acc)

        # Recover per-image filenames in test-loader order (shuffle=False).
        _, test_items = split_for_fold(manifest, fold)
        assert len(test_items) == len(preds)
        for (identity, filename), p, y in zip(test_items, preds.tolist(), labels.tolist()):
            all_preds.append(p)
            all_labels.append(y)
            records.append({
                "identity": identity,
                "filename": filename,
                "fold": fold,
                "true": y,
                "pred": p,
                "correct": p == y,
                "source": source_of(identity, filename)[1],
            })
        print(f"fold {fold}: acc={acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(identities))))
    overall = float(np.mean(all_preds == np.array(all_labels)))
    return {
        "identities": identities,
        "fold_accs": fold_accs,
        "mean_fold_acc": float(np.mean(fold_accs)),
        "pooled_acc": overall,
        "confusion_matrix": cm,
        "records": records,
    }


def plot_confusion_matrix(cm: np.ndarray, identities: list[str], path, title: str = "LOOCV Confusion Matrix (10-fold, ResNet18)") -> None:
    """Row-normalized (recall per identity) heatmap with raw counts annotated."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(identities)))
    ax.set_yticks(range(len(identities)))
    ax.set_xticklabels(identities, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(identities, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\ncolor = row-normalized recall, number = count")

    for i in range(len(identities)):
        for j in range(len(identities)):
            if cm[i, j] == 0:
                continue
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyze(result: dict) -> dict:
    """Error breakdowns: per-identity recall, top confused pairs, detection bucket."""
    identities = result["identities"]
    cm = result["confusion_matrix"]
    records = result["records"]

    per_identity = {
        ident: {
            "recall": float(cm[i, i] / cm[i].sum()),
            "errors": int(cm[i].sum() - cm[i, i]),
        }
        for i, ident in enumerate(identities)
    }

    pairs = []
    for i in range(len(identities)):
        for j in range(len(identities)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": identities[i], "pred": identities[j], "count": int(cm[i, j])})
    pairs.sort(key=lambda p: p["count"], reverse=True)

    # Detection-bucket error rates (crop vs full-image fallback).
    det_report = load_json(RESULTS_DIR / "detection_report.json")["images"]
    bucket_stats: dict[str, Counter] = {}
    for r in records:
        key = f"{r['identity']}/{r['filename']}"
        bucket = det_report.get(key, {}).get("status", "unknown")
        b = bucket_stats.setdefault(bucket, Counter())
        b["total"] += 1
        b["errors"] += int(not r["correct"])
    buckets = {
        k: {"total": v["total"], "errors": v["errors"], "error_rate": v["errors"] / v["total"]}
        for k, v in bucket_stats.items()
    }

    return {"per_identity": per_identity, "top_confused_pairs": pairs[:15], "detection_buckets": buckets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--title", type=str, default="LOOCV Confusion Matrix (10-fold, ResNet18)")
    parser.add_argument("--hn", action="store_true", help="evaluate hard-negative checkpoints")
    args = parser.parse_args()

    ckpt_dir = RESULTS_DIR / "checkpoints_hn" if args.hn else CHECKPOINT_DIR
    suffix = "_hn" if args.hn else ""
    result = collect_predictions(pick_device(args.device), checkpoint_dir=ckpt_dir)
    analysis = analyze(result)

    print(f"\nMean fold accuracy: {result['mean_fold_acc']:.4f}")
    print(f"Pooled accuracy:    {result['pooled_acc']:.4f}  (n=1700)")

    print("\nPer-identity recall (worst first):")
    for ident, s in sorted(analysis["per_identity"].items(), key=lambda kv: kv[1]["recall"]):
        print(f"  {ident:22s} {s['recall']:.3f}  ({s['errors']} errors)")

    print("\nTop confused pairs (true -> pred):")
    for p in analysis["top_confused_pairs"][:10]:
        print(f"  {p['true']:22s} -> {p['pred']:22s} x{p['count']}")

    print("\nError rate by detection bucket:")
    for k, v in analysis["detection_buckets"].items():
        print(f"  {k:10s} {v['errors']}/{v['total']} = {v['error_rate']:.3f}")

    save_json(
        {
            "mean_fold_acc": result["mean_fold_acc"],
            "pooled_acc": result["pooled_acc"],
            "fold_accs": result["fold_accs"],
            "identities": result["identities"],
            "confusion_matrix": result["confusion_matrix"].tolist(),
            "analysis": analysis,
            "errors": [r for r in result["records"] if not r["correct"]],
        },
        RESULTS_DIR / f"loocv_evaluation{suffix}.json",
    )
    plot_confusion_matrix(
        result["confusion_matrix"], result["identities"], RESULTS_DIR / f"confusion_matrix{suffix}.png",
        title=args.title,
    )
    print(f"\nWrote {RESULTS_DIR / f'loocv_evaluation{suffix}.json'}")
    print(f"Wrote {RESULTS_DIR / f'confusion_matrix{suffix}.png'}")


if __name__ == "__main__":
    main()