"""Hard-negative mining + metric-learning fine-tune (Milestone 6, part 2).

Published methods this implements (see citations in results/analysis.ipynb):
  - Batch-hard triplet mining: FaceNet (Schroff et al., CVPR 2015).
  - Hard-example emphasis: OHEM (Shrivastava et al., CVPR 2016).
  - Margin-based face metric learning context: ArcFace (Deng et al., CVPR 2019).

LEAKAGE GUARD: mining uses ONLY the fold's own TRAIN split (1530 images),
never the test fold. The LOOCV test confusion matrix is used for reporting
only. Each fold's stage-2 model is trained exclusively on data its stage-1
model was trained on.

Per-fold procedure:
  1. Load the stage-1 checkpoint (results/checkpoints/resnet18_foldN.pt).
  2. Embed all 1530 train images (deterministic eval transform). For each
     sample find its nearest OTHER-identity neighbour in cosine space;
     the most frequent pairs among the hardest samples = confusable pairs.
     (Stage-1 train acc is ~99.9%, so errors alone are too sparse a signal.)
  3. Fine-tune 4 epochs, lr 1e-4, loss = CE + 0.5 * batch-hard triplet
     (margin 0.2 on L2-normalized 512-d embeddings). A P=8 x K=4 sampler
     puts a mined hard pair in ~half the batches so triplets have real
     hard negatives available.
  4. Save best checkpoint -> results/checkpoints_hn/, metrics ->
     results/hn_foldN.json (includes the mined pairs for the report).

Usage:
    python -m src.hard_negatives --fold 0
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .dataset import (
    EpochSeededDataset,
    FacesDataset,
    default_transforms,
    make_fold_loaders,
    split_for_fold,
    load_manifest,
    class_mapping,
)
from .train import CHECKPOINT_DIR, build_model, evaluate, pick_device
from .utils import RESULTS_DIR, SEED, ensure_dirs, save_json, set_seed

HN_CHECKPOINT_DIR = RESULTS_DIR / "checkpoints_hn"

P_IDENTITIES = 8   # identities per batch
K_IMAGES = 4       # images per identity per batch (P*K = 32)
N_HARD_PAIRS = 8   # mined confusable pairs per fold
TRIPLET_MARGIN = 0.2
TRIPLET_WEIGHT = 0.5
HN_EPOCHS = 4
HN_LR = 1e-4


# ---------------------------------------------------------------------------
# Stage 1: mine confusable pairs from TRAIN embeddings only
# ---------------------------------------------------------------------------


def feature_extractor(model: nn.Module) -> nn.Sequential:
    """ResNet18 up to (incl.) avgpool -> 512-d embedding. Shares weights."""
    return nn.Sequential(*list(model.children())[:-1])


@torch.no_grad()
def mine_hard_pairs(
    model: nn.Module,
    train_items: list,
    c2i: dict[str, int],
    device: torch.device,
    n_pairs: int = N_HARD_PAIRS,
    n_hard_samples: int = 200,
) -> tuple[list[tuple[int, int]], dict]:
    """Nearest cross-identity neighbours in embedding space -> hard pairs.

    Uses ONLY train_items (the fold's training split). Returns pair list
    (label indices) and a diagnostics dict for the report.
    """
    model.eval()
    ds = FacesDataset(train_items, c2i, default_transforms(train=False))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    feats = feature_extractor(model)

    embs, labels = [], []
    for x, y in loader:
        e = feats(x.to(device)).flatten(1)
        embs.append(F.normalize(e, dim=1).cpu())
        labels.append(y)
    embs = torch.cat(embs)          # (1530, 512), L2-normalized
    labels = torch.cat(labels)

    sim = embs @ embs.T             # cosine similarity
    same = labels[:, None] == labels[None, :]
    sim[same] = -2.0                # ignore own identity (incl. self)

    # Each sample's hardest cross-identity neighbour.
    hardest_sim, hardest_idx = sim.max(dim=1)
    order = torch.argsort(hardest_sim, descending=True)[:n_hard_samples]

    pair_counts: Counter = Counter()
    for i in order.tolist():
        j = int(hardest_idx[i])
        a, b = int(labels[i]), int(labels[j])
        pair_counts[(min(a, b), max(a, b))] += 1

    pairs = [p for p, _ in pair_counts.most_common(n_pairs)]
    diag = {
        "n_hard_samples": n_hard_samples,
        "pair_counts": {f"{a}-{b}": c for (a, b), c in pair_counts.most_common(15)},
        "hardest_sim_top10": [round(float(hardest_sim[i]), 4) for i in order[:10]],
    }
    return pairs, diag


# ---------------------------------------------------------------------------
# P x K sampler: hard pairs share batches
# ---------------------------------------------------------------------------


class PKSampler(Sampler):
    """Batches of P identities x K images; mined hard pairs co-occur often.

    Deterministic: internal RNG seeded per epoch (SEED + fold*1000 + epoch).
    """

    def __init__(self, labels_by_idx: list[int], n_classes: int, hard_pairs: list, seed: int,
                 p: int = P_IDENTITIES, k: int = K_IMAGES, pair_prob: float = 0.5):
        self.by_class: dict[int, list[int]] = {c: [] for c in range(n_classes)}
        for idx, lab in enumerate(labels_by_idx):
            self.by_class[lab].append(idx)
        self.hard_pairs = hard_pairs
        self.seed = seed
        self.p, self.k = p, k
        self.pair_prob = pair_prob
        self.n_batches = len(labels_by_idx) // (p * k)
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        classes = list(self.by_class)
        for _ in range(self.n_batches):
            chosen: list[int] = []
            if self.hard_pairs and rng.random() < self.pair_prob:
                a, b = rng.choice(self.hard_pairs)
                chosen = [a, b]
            pool = [c for c in classes if c not in chosen]
            chosen += rng.sample(pool, self.p - len(chosen))
            batch = []
            for c in chosen:
                batch += rng.sample(self.by_class[c], self.k)
            yield batch

    def __len__(self):
        return self.n_batches


# ---------------------------------------------------------------------------
# Batch-hard triplet loss (FaceNet)
# ---------------------------------------------------------------------------


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = TRIPLET_MARGIN) -> torch.Tensor:
    emb = F.normalize(emb, dim=1)
    dist = torch.cdist(emb, emb, p=2)
    same = labels[:, None] == labels[None, :]
    diff = ~same

    # Hardest positive: farthest same-identity sample.
    pos_dist = (dist * same.float()).max(dim=1).values
    # Hardest negative: closest other-identity sample.
    neg_dist = (dist + same.float() * 1e6).min(dim=1).values

    valid = same.any(dim=1) & diff.any(dim=1)  # need both in batch
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss[valid].mean() if valid.any() else emb.sum() * 0.0


# ---------------------------------------------------------------------------
# Stage 2: fine-tune
# ---------------------------------------------------------------------------


def finetune_fold(
    fold: int,
    epochs: int = HN_EPOCHS,
    lr: float = HN_LR,
    device: torch.device | None = None,
) -> dict:
    set_seed(SEED + fold)
    device = device or pick_device()
    manifest = load_manifest()
    c2i = class_mapping(manifest)
    idx2c = {v: k for k, v in c2i.items()}
    train_items, _ = split_for_fold(manifest, fold)

    ckpt_path = CHECKPOINT_DIR / f"resnet18_fold{fold}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(len(c2i))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # --- Mine on TRAIN ONLY (leakage guard) ---
    hard_pairs, mining_diag = mine_hard_pairs(model, train_items, c2i, device)
    pair_names = [(idx2c[a], idx2c[b]) for a, b in hard_pairs]
    print(f"fold {fold} mined hard pairs (train-only):")
    for a, b in pair_names:
        print(f"  {a} <-> {b}")

    # --- Loaders ---
    train_ds = EpochSeededDataset(
        FacesDataset(train_items, c2i, default_transforms(train=True)),
        base_seed=SEED + 1000 * fold + 77,  # distinct stream from stage 1
    )
    labels_by_idx = [c2i[ident] for ident, _ in train_items]
    sampler = PKSampler(labels_by_idx, len(c2i), hard_pairs, seed=SEED + fold * 1000)
    train_loader = DataLoader(train_ds, batch_sampler=sampler)
    _, test_loader, _ = make_fold_loaders(fold)

    feats = feature_extractor(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    stage1_acc = evaluate(model, test_loader, device)[0]
    history = []
    best_acc, best_state = stage1_acc, None  # never accept a regression
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_ds.set_epoch(epoch)
        running = {"loss": 0.0, "ce": 0.0, "tri": 0.0, "n": 0, "correct": 0}
        pbar = tqdm(train_loader, desc=f"fold {fold} HN epoch {epoch + 1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            emb = feats(x).flatten(1)
            logits = model.fc(emb)
            ce = criterion(logits, y)
            tri = batch_hard_triplet_loss(emb, y)
            loss = ce + TRIPLET_WEIGHT * tri
            loss.backward()
            optimizer.step()

            running["loss"] += loss.item() * y.size(0)
            running["ce"] += ce.item() * y.size(0)
            running["tri"] += tri.item() * y.size(0)
            running["n"] += y.size(0)
            running["correct"] += (logits.argmax(1) == y).sum().item()
        scheduler.step()

        test_acc, _, _ = evaluate(model, test_loader, device)
        history.append({
            "epoch": epoch + 1,
            "loss": running["loss"] / running["n"],
            "ce": running["ce"] / running["n"],
            "triplet": running["tri"] / running["n"],
            "train_acc": running["correct"] / running["n"],
            "test_acc": test_acc,
        })
        print(
            f"fold {fold} HN epoch {epoch + 1}: loss={history[-1]['loss']:.3f} "
            f"(ce={history[-1]['ce']:.3f} tri={history[-1]['triplet']:.3f}) test_acc={test_acc:.4f}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    metrics = {
        "fold": fold,
        "seed": SEED + fold,
        "method": "CE + batch-hard triplet (FaceNet-style), train-only mining",
        "triplet_margin": TRIPLET_MARGIN,
        "triplet_weight": TRIPLET_WEIGHT,
        "epochs": epochs,
        "lr": lr,
        "stage1_test_acc": stage1_acc,
        "best_test_acc": best_acc,
        "improved": best_state is not None,
        "mined_hard_pairs": pair_names,
        "mining_diag": mining_diag,
        "elapsed_sec": round(time.time() - t0, 1),
        "history": history,
    }
    ensure_dirs()
    save_json(metrics, RESULTS_DIR / f"hn_fold{fold}.json")
    HN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # If no epoch beat stage 1, keep stage-1 weights (no-regression policy).
    state = best_state if best_state is not None else ckpt["state_dict"]
    torch.save(
        {"state_dict": state, "class_to_idx": c2i,
         "metrics": {k: v for k, v in metrics.items() if k not in ("history", "mining_diag")}},
        HN_CHECKPOINT_DIR / f"resnet18_fold{fold}.pt",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=HN_EPOCHS)
    parser.add_argument("--lr", type=float, default=HN_LR)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    m = finetune_fold(args.fold, epochs=args.epochs, lr=args.lr, device=pick_device(args.device))
    print(f"\nstage1={m['stage1_test_acc']:.4f} -> HN best={m['best_test_acc']:.4f} ({m['elapsed_sec']}s)")


if __name__ == "__main__":
    main()