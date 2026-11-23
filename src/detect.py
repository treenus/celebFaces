"""YOLOv3-tiny person detection + Haar face fallback + crop caching (Milestone 2).

Measured on this dataset (85-image sample, 5 per identity):
  - YOLOv3-tiny person boxes hit only ~1%: the images are mostly tight
    head-and-shoulders crops, and YOLO needs a substantial body to fire.
  - Haar frontal-face cascade hits ~88% on the same sample.

So the pipeline is:
  1. Haar cascade face box (largest face), expanded by MARGIN.
  2. If Haar misses -> YuNet (cv2.FaceDetectorYN, pose-robust ONNX face
     detector). Measured: YuNet recovers 96/96 images that Haar+YOLO
     missed or cropped as torso-only (conf 0.9+).
  3. If YuNet misses -> YOLOv3-tiny person box (COCO class 0), expanded.
     Kept as a last-resort safety net (and satisfies the YOLOv3 paper
     integration); in practice YuNet catches everything Haar misses.
  4. If all miss -> write nothing; `src.dataset.source_of()` falls back
     to the full raw image for that file.

A per-image report is written to results/detection_report.json so every
bucket (yolo / haar / miss) is auditable.

Crops keep the exact raw filenames, so data/folds.json stays valid.

Usage:
    python -m src.detect            # skips images already cropped
    python -m src.detect --force    # re-detect everything
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .utils import (
    FACES_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    ensure_dirs,
    list_identities,
    list_images,
    save_json,
)

WEIGHTS_DIR = PROJECT_ROOT / "weights"
CFG_PATH = WEIGHTS_DIR / "yolov3-tiny.cfg"
WEIGHTS_PATH = WEIGHTS_DIR / "yolov3-tiny.weights"
YUNET_PATH = WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"

PERSON_CLASS_ID = 0     # COCO class 0 = person
INPUT_SIZE = 416        # YOLOv3-tiny default
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
MARGIN = 0.20           # expand box by 20% per side (chin/forehead headroom)
JPEG_QUALITY = 95


def load_net() -> cv2.dnn_Net:
    net = cv2.dnn.readNetFromDarknet(str(CFG_PATH), str(WEIGHTS_PATH))
    # CPU backend: deterministic and memory-light (8 GB machine).
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def load_haar() -> cv2.CascadeClassifier:
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {path}")
    return cascade


def detect_face_haar(cascade: cv2.CascadeClassifier, img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest frontal-face box (x, y, w, h), or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def load_yunet() -> cv2.FaceDetectorYN:
    if not YUNET_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model not found at {YUNET_PATH}. Download "
            "face_detection_yunet_2023mar.onnx from opencv/opencv_zoo."
        )
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 5000)


def detect_face_yunet(det: cv2.FaceDetectorYN, img_bgr: np.ndarray) -> tuple[float, tuple[int, int, int, int]] | None:
    """Largest YuNet face: (conf, (x, y, w, h)), or None. Pose-robust."""
    h, w = img_bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, bw, bh = (int(v) for v in best[:4])
    return float(best[-1]), (x, y, bw, bh)


def detect_persons(net: cv2.dnn_Net, img_bgr: np.ndarray) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Return [(conf, (x, y, w, h)), ...] for person detections, best first."""
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img_bgr, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_names = net.getLayerNames()
    out_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outs = net.forward(out_names)

    boxes, confs = [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if class_id != PERSON_CLASS_ID or conf < CONF_THRESHOLD:
                continue
            cx, cy, bw, bh = det[:4] * np.array([w, h, w, h])
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)
            boxes.append([x, y, int(bw), int(bh)])
            confs.append(conf)

    if not boxes:
        return []

    keep = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD)
    keep = np.array(keep).flatten().tolist()
    dets = [(confs[i], tuple(boxes[i])) for i in keep]
    dets.sort(key=lambda d: d[0], reverse=True)
    return dets


def expand_box(box: tuple[int, int, int, int], img_shape, margin: float = MARGIN) -> tuple[int, int, int, int]:
    """Expand (x, y, w, h) by `margin` per side, clamped to image bounds."""
    h, w = img_shape[:2]
    x, y, bw, bh = box
    mx, my = int(bw * margin), int(bh * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w, x + bw + mx)
    y1 = min(h, y + bh + my)
    return x0, y0, x1 - x0, y1 - y0


def run(force: bool = False) -> dict:
    ensure_dirs()
    net = load_net()
    haar = load_haar()
    yunet = load_yunet()

    report: dict[str, dict] = {}
    stats: Counter = Counter()

    identities = list_identities()
    all_images = [(ident, p) for ident in identities for p in list_images(ident)]

    for identity, path in tqdm(all_images, desc="detect", unit="img"):
        dest = FACES_DIR / identity / path.name
        key = f"{identity}/{path.name}"

        if dest.exists() and not force:
            stats["cached"] += 1
            report.setdefault(key, {"status": "cached"})
            continue

        img = cv2.imread(str(path))
        if img is None:
            stats["unreadable"] += 1
            report[key] = {"status": "unreadable"}
            continue

        # Stage 1: Haar (fast, frontal). Stage 2: YuNet (pose-robust).
        # Stage 3: YOLO person box (last resort). Else: full-image fallback.
        box = detect_face_haar(haar, img)
        source = "haar"
        extra: dict = {}
        if box is None:
            yn = detect_face_yunet(yunet, img)
            if yn is not None:
                conf, box = yn
                source = "yunet"
                extra = {"conf": round(conf, 4)}
        if box is None:
            dets = detect_persons(net, img)
            if dets:
                conf, box = dets[0]
                source = "yolo"
                extra = {"conf": round(conf, 4), "n_persons": len(dets)}

        if box is None:
            stats["miss"] += 1
            report[key] = {"status": "miss", "n_persons": 0}
            continue

        x, y, w, h = expand_box(box, img.shape)
        crop = img[y : y + h, x : x + w]

        dest.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            stats["write_failed"] += 1
            report[key] = {"status": "write_failed"}
            continue

        stats[source] += 1
        report[key] = {
            "status": source,
            "box_xywh": [x, y, w, h],
            "img_wh": [img.shape[1], img.shape[0]],
            **extra,
        }

    total = len(all_images)
    n_miss = stats["miss"] + stats["unreadable"] + stats["write_failed"]
    summary = {
        "detector": "haar -> yunet -> yolov3-tiny person (margin=0.20)",
        "conf_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "total_images": total,
        "counts": dict(stats),
        "fallback_rate": n_miss / total,
        "multi_person_images": sum(
            1 for r in report.values() if r.get("n_persons", 0) > 1
        ),
    }
    save_json({"summary": summary, "images": report}, RESULTS_DIR / "detection_report.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-detect cached images")
    args = parser.parse_args()

    summary = run(force=args.force)
    print("\n=== Detection summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nReport -> {RESULTS_DIR / 'detection_report.json'}")


if __name__ == "__main__":
    main()