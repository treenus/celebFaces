"""ResNet18 baseline: model factory + per-fold training (Milestone 4).

Route A from the plan: ImageNet-pretrained ResNet18, final FC replaced with
a 17-way head, fine-tuned on each fold's 1530-image train split.

8 GB unified-memory defaults: batch 32, num_workers 0, MPS if available.
Determinism: seeds fixed per fold; augmentation is per-sample seeded by
src.dataset.EpochSeededDataset (set_epoch each epoch).

Usage:
    python -m src.train --fold 0 --epochs 10
    python -m src.train --fold 0 --epochs 10 --device cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from .dataset import make_fold_loaders
from .utils import RESULTS_DIR, SEED, ensure_dirs, save_json, set_seed

N_CLASSES = 17
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(n_classes: int = N_CLASSES) -> nn.Module:
    """ImageNet-pretrained ResNet18 with a fresh n-class head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Return (accuracy, all_preds, all_labels) on a loader."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = (preds == labels).float().mean().item()
    return acc, preds, labels


def train_fold(
    fold: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    save_checkpoint: bool = True,
) -> dict:
    """Fine-tune ResNet18 on one LOOCV fold. Returns a metrics dict."""
    set_seed(SEED + fold)
    device = device or pick_device()

    train_loader, test_loader, c2i = make_fold_loaders(fold, batch_size=batch_size)
    model = build_model(len(c2i)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_loader.dataset.set_epoch(epoch)  # deterministic augmentation
        running_loss, seen, correct = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"fold {fold} epoch {epoch + 1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            pbar.set_postfix(loss=f"{running_loss / seen:.3f}", acc=f"{correct / seen:.3f}")

        scheduler.step()
        test_acc, _, _ = evaluate(model, test_loader, device)
        train_acc = correct / seen
        history.append({
            "epoch": epoch + 1,
            "train_loss": running_loss / seen,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
        })
        print(
            f"fold {fold} epoch {epoch + 1:2d}: "
            f"loss={running_loss / seen:.3f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    metrics = {
        "fold": fold,
        "seed": SEED + fold,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_test_acc": best_acc,
        "final_test_acc": history[-1]["test_acc"],
        "elapsed_sec": round(elapsed, 1),
        "history": history,
    }

    ensure_dirs()
    save_json(metrics, RESULTS_DIR / f"train_fold{fold}.json")
    if save_checkpoint and best_state is not None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": best_state, "class_to_idx": c2i, "metrics": {k: v for k, v in metrics.items() if k != "history"}},
            CHECKPOINT_DIR / f"resnet18_fold{fold}.pt",
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, help="test fold index 0-9")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None, help="mps | cpu | cuda")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    metrics = train_fold(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=pick_device(args.device),
        save_checkpoint=not args.no_checkpoint,
    )
    print(f"\nBest test acc: {metrics['best_test_acc']:.4f} in {metrics['elapsed_sec']}s")


if __name__ == "__main__":
    main()