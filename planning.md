# CSE 402 Final Project — Plan

## 1. Assignment Summary (from `Final.pdf`)

**Task:** Face recognition on the *Celebrity Faces Dataset* — 17 celebrity classes, 100 images each (1700 total).

**Required deliverables (all three must be in the code):**

1. **Data preparation** — 10-fold leave-one-out cross-validation (LOOCV):
   - Split each identity's 100 images into 10 folds of 10.
   - Train on 9 folds → `9 folds × 10 imgs × 17 classes = 1530` images.
   - Test on 1 fold → `1 fold × 10 imgs × 17 classes = 170` images.
   - Repeat until every fold has been the test set → average accuracy = overall LOOCV accuracy.
2. **Face recognition model** — choose one:
   - **Option 1:** Pre-trained models (e.g. `torchvision.models.alexnet(pretrained=True)`).
   - **Option 2:** Develop our own model using any internet resources.
3. **Evaluation** — report average accuracy under LOOCV + draw a confusion matrix.

**Report must include:** team contributions, model description + improvement schemes, LOOCV accuracy, confusion matrix + discussion of confusable identities, submitted as `LASTNAME_FIRSTNAME.pdf`.

**Grading (10 pts):** data prep 1 · model 2 · LOOCV scheme 0.5 · confusion matrix 0.5 · contributions 1 · model description 2 · improvement schemes 0.5 · accuracy 0.5 · confusion matrix 0.5 · discussion 0.5 · readability 1.
**Bonus:** top-3 teams get +5 pts each (split across members).

> ⚠️ Code is randomly re-run by the graders. Reported results must reproduce (small variance OK) or it's treated as academic dishonesty → score 0. **Fix all random seeds.**

---

## 2. Key Observation: Images Are NOT Tightly Cropped

Sample inspection (`Angelina Jolie/001`, `Tom Cruise/050`) shows many images are **upper-body / full-scene shots**, not face crops. Feeding these raw into a classifier wastes capacity on background/clothing and hurts accuracy.

**→ A face detection + alignment stage is required before recognition.** This is also the single biggest accuracy lever (DeepFace benchmarks: detection adds up to ~42%, alignment ~6%).

---

## 3. Proposed Pipeline (all Python)

```
raw image
   │
   ▼
[1] Face detection  ──►  bounding box
   │
   ▼
[2] Crop + align    ──►  normalized face chip (e.g. 224×224)
   │
   ▼
[3] Embedding / classifier
   │
   ▼
[4] LOOCV evaluation ──►  accuracy + confusion matrix
```

### Stage 1 — Face Detection (YOLOv3)

- **YOLOv3** (Darknet) as requested. Options:
  - `yolov3` / `yolov3-tiny` weights via OpenCV `dnn` module (`cv2.dnn.readNetFromDarknet`) — no heavy deps, CPU-friendly.
  - Or `ultralytics` YOLO (v8/v11) which is easier to install and better maintained; DeepFace also wraps `yolov8n…yolov12l` as detector backends.
- **Recommendation:** start with **YOLOv3-tiny via OpenCV DNN** (matches the "import YOLOv3" request, minimal setup). Keep `ultralytics` as a fallback if detection quality is poor.
- **Fallback detector:** OpenCV Haar cascade or `retinaface` (RetinaFace is the strongest in DeepFace's portfolio) if YOLO misses faces.
- **Cache the crops to disk** (`data/faces/<identity>/<file>.jpg`) so detection runs once, not per fold. This also makes grading re-runs fast and deterministic.

### Stage 2 — Crop + Align

- Expand the YOLO box by a margin (~20–30%) to include chin/forehead.
- Resize to the model's expected input (224×224 for AlexNet/ResNet; 112×112 for ArcFace).
- Optional alignment via 5-point landmarks (eyes/nose/mouth) — improves accuracy but adds a dependency (`face_alignment` / `insightface`). Mark as a stretch goal.

### Stage 3 — Recognition Model

The assignment's Option 1 example is **AlexNet** (ImageNet-pretrained, 1000-class head). Two viable routes:

| Route | Approach | Pros | Cons |
|---|---|---|---|
| **A. Fine-tune CNN classifier** | AlexNet / ResNet18 / EfficientNet-B0, replace final layer with 17-way head, train on the 1530-image train split | Simple, matches assignment example, easy to explain | Needs training per fold (10×); small data |
| **B. Pretrained face embeddings + classifier** | ArcFace / FaceNet512 / Buffalo_L embeddings → cosine k-NN or linear SVM | State-of-the-art accuracy, no per-fold training of the backbone | Extra deps (`deepface`/`insightface`), less "our own model" |

**Recommendation:** do **both** and compare — Route A as the baseline (satisfies "develop your own model"), Route B as the improvement scheme. This directly feeds the "schemes for improving performance" rubric item.

**On "Darknet" for recognition:** Darknet is the *framework* YOLOv3 is built on, not a face-recognition model. For recognition, the strong modern choices are **ArcFace** (best accuracy, margin-based loss) and **FaceNet512**. We'll use ArcFace/FaceNet via `deepface` or `insightface` rather than Darknet.

**On the "train off lookalikes" idea (live augmentation):** This is essentially **hard-negative mining / metric learning with confusable pairs**. It's a real, published technique (triplet loss, ArcFace margin). Practical version for us:
- After a first pass, find the most-confused identity pairs from the confusion matrix.
- Mine those pairs as hard negatives and fine-tune with a triplet/contrastive objective.
- This is a strong "improvement scheme" write-up and is more defensible than a vague lookalike idea.

### Stage 4 — LOOCV Evaluation

- Implement the 10-fold split **exactly** as specified (10 imgs/identity/fold).
- For each fold: train (or fit classifier) on 1530, predict 170, record.
- Aggregate → average accuracy + full 17×17 confusion matrix.
- **Seed everything** (`random`, `numpy`, `torch`) and log the seed in the report.

---

## 4. Repo / File Layout (proposed)

```
finalproj/
├── planning.md
├── Final.pdf
├── Celebrity Faces Dataset/        # raw (given)
├── data/
│   └── faces/                      # cached YOLO crops (generated)
├── src/
│   ├── detect.py                   # YOLOv3 detection + crop + cache
│   ├── dataset.py                  # LOOCV fold generation, Dataset/DataLoader
│   ├── models.py                   # AlexNet/ResNet head + embedding wrappers
│   ├── train.py                    # per-fold training (Route A)
│   ├── evaluate.py                 # LOOCV loop, accuracy, confusion matrix
│   └── utils.py                    # seeding, plotting, metrics
├── notebooks/
│   └── exploration.ipynb           # EDA, sample crops, confusion matrix viz
├── results/                        # metrics json, confusion matrix png
├── requirements.txt
└── report/                         # LASTNAME_FIRSTNAME.pdf source
```

---

## 5. Milestones

| # | Milestone | Output |
|---|---|---|
| 1 | Environment + deps (`torch`, `torchvision`, `opencv-python`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) | `requirements.txt` |
| 2 | YOLOv3 detection + crop caching over all 1700 images | `data/faces/` |
| 3 | LOOCV splitter + dataset loader | `src/dataset.py` |
| 4 | Baseline classifier (AlexNet/ResNet18) + per-fold training | `src/train.py` |
| 5 | LOOCV evaluation + confusion matrix | `results/` |
| 6 | Improvement pass (ArcFace embeddings and/or hard-negative fine-tune) | comparison table |
| 7 | Report + contributions section | `report/*.pdf` |

---

## 6. Decisions (locked)

1. **Detector:** **YOLOv3-tiny via OpenCV DNN** for v1. Rationale: the paper is a required integration, so shipping YOLOv3 now keeps us within our rights to *upgrade* to a newer YOLO (v8/v11) in the improvement pass.
2. **Recognition route:** **ResNet18** (or a newer AlexNet) fine-tuned for face recognition, tuned for a **low false-positive rate**, then **hard-negative fine-tuning** in the improvement pass.
3. **Alignment:** skip 5-point landmark alignment for v1; revisit as a stretch goal.
4. **Compute:** Apple Silicon Mac, **8 GB unified memory**, MPS available. Train ResNet18 on MPS with batch 32, `num_workers=0` (worker processes duplicate memory). Keep the backbone small; frozen-feature extraction is the fallback if fine-tuning is too slow.
5. **Team:** single-person project (solo).
6. **Detection fallback:** if YOLO misses a face, use the **full raw image** (dataset resolves crop → full automatically). Track the fallback rate; if it's high, add a secondary detector (Haar cascade / RetinaFace) for that bucket.
7. **Vendored weights:** `weights/yolov3-tiny.{cfg,weights}` + `coco.names` are committed in-repo so graders can re-run offline. OpenCV pinned to `>=4.8,<5` (5.x removed `readNetFromDarknet`).

## 8. Progress Log

- [x] **Milestone 1** — venv + deps installed (`numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `opencv-python`, `Pillow`, `torch` 2.15.0, `torchvision` 0.29.2; MPS available).
- [x] **Milestone 3** — `src/make_folds.py` → `data/folds.json` (seed 402, sha256 stable). `src/dataset.py` consumes the manifest: per-fold train/test `DataLoader`s (1530/170, balanced), crop→full-image fallback, per-sample deterministic augmentation (`set_epoch`), verified bit-identical batches across runs.
- [x] **Weights vendored** — `weights/yolov3-tiny.cfg` (1.9 KB), `yolov3-tiny.weights` (35.4 MB), `coco.names` (80 classes); loads via `cv2.dnn.readNetFromDarknet` (OpenCV 4.14).
- [x] **Milestone 2** — `src/detect.py`: Haar face box → YOLOv3-tiny person fallback → full-image fallback. **Measured:** YOLO person boxes hit only ~1% (images are tight head shots; YOLO needs a body) and can be torso-only on group shots, so Haar runs FIRST. Full pass: 1604 Haar + 2 YOLO crops, 94 misses (5.5% full-image fallback). Report: `results/detection_report.json`. Crops cached in `data/faces/` (~25 s full pass).
- [ ] **Milestone 4** — ResNet18 baseline + per-fold training.
- [ ] **Milestone 5** — LOOCV evaluation + confusion matrix.
- [ ] **Milestone 6** — hard-negative improvement pass.
- [ ] **Milestone 7** — report.

---

## 7. Risks

- **Reproducibility (grading risk):** must seed + cache crops so re-runs match.
- **Detection failures:** YOLO may miss profile/occluded faces → need fallback + manual review of failures.
- **Small data:** 1530 train images is tiny → heavy augmentation (flip, crop, color jitter) + transfer learning essential.
- **Per-fold training cost:** 10× training runs; keep model small or use frozen-backbone embeddings.
