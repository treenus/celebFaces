"""Route B reference: pretrained ArcFace embeddings + cosine k-NN LOOCV.

Upper-bound comparison for the report/presentation — NOT the primary
submission (our own fine-tuned ResNet18 remains that).

Model: insightface Buffalo_L (SCRFD detector + ArcFace ResNet50 recognition,
trained on WebFace600K / MS1M-scale data). 512-d L2-normalized embeddings.

Protocol (leakage-free by construction):
  - Embed every image ONCE (cached to data/embeddings.npz). The backbone is
    frozen and was never trained on this dataset, so embeddings are
    fold-independent.
  - For each fold: classify the 170 test embeddings against ONLY the 1530
    train embeddings via cosine-similarity k-NN (majority vote, ties broken
    by summed similarity). No training, no model selection on test data —
    this route has NO best-epoch caveat.

Detection note: SCRFD at det_size 640 misses faces that fill the frame
(our tight crops); det_size 320 works. We try 320 -> 160 -> 640 and fall
back to the raw image if all fail.

Usage:
    python -m src.embeddings            # embed (cached) + evaluate
    python -m src.embeddings --force    # re-embed everything
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from .dataset import load_manifest, class_mapping, split_for_fold, source_of
from .utils import (
    DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    ensure_dirs,
    list_identities,
    list_images,
    save_json,
)

EMB_PATH = DATA_DIR / "embeddings.npz"
DET_SIZES = [(320, 320), (160, 160), (640, 640)]
K_NEIGHBORS = 5


def _load_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=DET_SIZES[0])
    return app


def _embed_one(app, img_bgr: np.ndarray) -> tuple[np.ndarray, str] | None:
    """Try det sizes until a face is found; return (embedding, det_size_tag)."""
    for ds in DET_SIZES:
        app.prepare(ctx_id=-1, det_size=ds)
        faces = app.get(img_bgr)
        if faces:
            best = max(faces, key=lambda f: f.det_score)
            return best.normed_embedding.astype(np.float32), f"{ds[0]}"
    return None


def build_embeddings(force: bool = False) -> dict:
    """Embed all 1700 images (crop first, raw fallback). Cache to npz."""
    if EMB_PATH.exists() and not force:
        z = np.load(EMB_PATH, allow_pickle=True)
        return {"keys": list(z["keys"]), "embs": z["embs"], "meta": z["meta"].tolist()}

    app = _load_app()
    keys, embs, meta = [], [], []
    stats: Counter = Counter()

    for identity in list_identities():
        for path in tqdm(list_images(identity), desc=f"embed {identity}", leave=False):
            key = f"{identity}/{path.name}"
            crop_path, _ = source_of(identity, path.name)
            img = cv2.imread(str(crop_path))
            got = _embed_one(app, img) if img is not None else None
            src = "crop"
            if got is None:
                img = cv2.imread(str(RAW_DATA_DIR / key))
                got = _embed_one(app, img) if img is not None else None
                src = "raw"
            if got is None:
                stats["failed"] += 1
                meta.append({"source": "failed", "det": None})
                keys.append(key)
                embs.append(np.zeros(512, dtype=np.float32))
                continue
            emb, det_tag = got
            stats[f"{src}_{det_tag}"] += 1
            keys.append(key)
            embs.append(emb)
            meta.append({"source": src, "det": det_tag})

    embs = np.stack(embs)
    ensure_dirs()
    np.savez_compressed(EMB_PATH, keys=np.array(keys), embs=embs, meta=np.array(meta, dtype=object))
    print("embedding stats:", dict(stats))
    return {"keys": keys, "embs": embs, "meta": meta}


def knn_loocv(data: dict, k: int = K_NEIGHBORS) -> dict:
    """Per-fold cosine k-NN against train embeddings only."""
    manifest = load_manifest()
    c2i = class_mapping(manifest)
    identities = manifest["identities"]
    key_to_row = {key: i for i, key in enumerate(data["keys"])}
    embs = data["embs"]

    fold_accs, all_preds, all_labels, records = [], [], [], []
    for fold in range(manifest["n_folds"]):
        train_items, test_items = split_for_fold(manifest, fold)
        train_rows = [key_to_row[f"{i}/{f}"] for i, f in train_items]
        test_rows = [key_to_row[f"{i}/{f}"] for i, f in test_items]

        train_embs = embs[train_rows]                    # (1530, 512)
        train_labels = np.array([c2i[i] for i, _ in train_items])
        sims = embs[test_rows] @ train_embs.T            # (170, 1530)

        preds = []
        for row in sims:
            top = np.argsort(row)[::-1][:k]
            votes = Counter(train_labels[top].tolist())
            best = max(votes.values())
            tied = [c for c, v in votes.items() if v == best]
            if len(tied) == 1:
                preds.append(tied[0])
            else:  # tie-break: summed similarity among tied classes
                preds.append(max(tied, key=lambda c: row[top][train_labels[top] == c].sum()))
        preds = np.array(preds)
        labels = np.array([c2i[i] for i, _ in test_items])
        acc = float((preds == labels).mean())
        fold_accs.append(acc)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        for (identity, filename), p, y in zip(test_items, preds, labels):
            records.append({"identity": identity, "filename": filename, "fold": fold,
                            "true": int(y), "pred": int(p), "correct": bool(p == y)})
        print(f"fold {fold}: kNN acc={acc:.4f}")

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(identities))))
    return {
        "identities": identities,
        "fold_accs": fold_accs,
        "mean_fold_acc": float(np.mean(fold_accs)),
        "pooled_acc": float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "confusion_matrix": cm,
        "records": records,
        "k": k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS)
    args = parser.parse_args()

    t0 = time.time()
    data = build_embeddings(force=args.force)
    result = knn_loocv(data, k=args.k)

    from .evaluate import plot_confusion_matrix, analyze

    analysis = analyze(result)
    print(f"\nArcFace kNN (k={args.k}) mean LOOCV: {result['mean_fold_acc']:.4f}")
    print(f"elapsed: {time.time() - t0:.0f}s")
    print("\nTop confused pairs:")
    for p in analysis["top_confused_pairs"][:8]:
        print(f"  {p['true']:22s} -> {p['pred']:22s} x{p['count']}")

    save_json(
        {
            "model": "insightface buffalo_l (ArcFace ResNet50) + cosine kNN",
            "k": args.k,
            "mean_fold_acc": result["mean_fold_acc"],
            "pooled_acc": result["pooled_acc"],
            "fold_accs": result["fold_accs"],
            "identities": result["identities"],
            "confusion_matrix": result["confusion_matrix"].tolist(),
            "analysis": analysis,
            "errors": [r for r in result["records"] if not r["correct"]],
        },
        RESULTS_DIR / "loocv_evaluation_arcface.json",
    )
    plot_confusion_matrix(
        result["confusion_matrix"], result["identities"],
        RESULTS_DIR / "confusion_matrix_arcface.png",
        title=f"LOOCV Confusion Matrix (ArcFace embeddings + {args.k}-NN, reference)",
    )
    print(f"\nWrote {RESULTS_DIR / 'loocv_evaluation_arcface.json'}")
    print(f"Wrote {RESULTS_DIR / 'confusion_matrix_arcface.png'}")


if __name__ == "__main__":
    main()