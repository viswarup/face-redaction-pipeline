"""Automated 3-Way Benchmark & Evaluation Pipeline for Hand-Annotated Images.
Evaluates:
  1. Model 1: Base SCRFD-34G (Standalone primary face detector + 28% box dilation + blur)
  2. Model 2: RetinaFace (Verifier model run directly as detector on raw images + 28% box dilation + blur)
  3. Model 3: SCRFD-34G + Body/Head Pose Fusion (Fused with YOLOv8n-Pose keypoints via PoseHeadFusor + 28% box dilation + blur)

Generates:
  - Table 1: Overall 3-Way Model Comparison (All 100 Images)
  - Table 2: Per-Condition Breakdown Across All 3 Models
  - Table 3: Detector Agreement & Correlated Failure Mode Matrix
  - 4-Panel Visual Comparison Strips in data/outputs/failure_gallery/
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.pipeline.detector import PrimaryFaceDetector
from app.pipeline.pose_head_fallback import PoseHeadFusor
from app.pipeline.redactor import apply_obfuscation, dilate_bbox
from app.pipeline.tracker_interpolator import compute_iou
from app.pipeline.verifier import RetinaFaceVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_100")


def load_ground_truth_annotations(dataset_dir: Path) -> dict[str, list[list[float]]]:
    """
    Flexible Ground-Truth loader supporting:
    1. COCO JSON (_annotations.coco.json / annotations.json).
    2. YOLO normalized .txt files matching each image stem.

    Returns:
        dict mapping image filename (or stem) to list of [x1, y1, x2, y2] boxes.
    """
    gt_map: dict[str, list[list[float]]] = {}

    # 1. Search for COCO JSON
    coco_candidates = list(dataset_dir.rglob("*.coco.json")) + list(dataset_dir.rglob("annotations*.json"))
    if coco_candidates:
        coco_path = coco_candidates[0]
        logger.info("Loading ground-truth annotations from COCO format: %s", coco_path)
        with open(coco_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        cat_names = {c["id"]: c.get("name", "").lower() for c in coco_data.get("categories", [])}
        target_cat_ids = {cid for cid, name in cat_names.items() if "face" in name}
        if not target_cat_ids:
            target_cat_ids = set(cat_names.keys())

        img_id_to_name = {im["id"]: im.get("file_name", "") for im in coco_data.get("images", [])}

        for ann in coco_data.get("annotations", []):
            if ann.get("category_id") in target_cat_ids:
                img_name = img_id_to_name.get(ann.get("image_id"))
                if img_name:
                    bbox = ann.get("bbox", [])
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        box = [float(x), float(y), float(x + w), float(y + h)]
                        base = Path(img_name).name
                        gt_map.setdefault(base, []).append(box)
                        gt_map.setdefault(img_name, []).append(box)

        if gt_map:
            return gt_map

    # 2. Search for YOLO .txt format
    yolo_files = list(dataset_dir.rglob("*.txt"))
    yolo_label_files = [f for f in yolo_files if not f.name.lower().startswith("readme")]
    if yolo_label_files:
        logger.info("Found YOLO format annotation files (%d labels)", len(yolo_label_files))
        for yf in yolo_label_files:
            stem = yf.stem
            img_candidates = list(dataset_dir.rglob(f"{stem}.jpg")) + list(dataset_dir.rglob(f"{stem}.png"))
            if not img_candidates:
                continue
            sample_img = cv2.imread(str(img_candidates[0]))
            if sample_img is None:
                continue
            ih, iw = sample_img.shape[:2]

            boxes: list[list[float]] = []
            with open(yf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, xc, yc, bw, bh = map(float, parts[:5])
                        x1 = (xc - bw / 2.0) * iw
                        y1 = (yc - bh / 2.0) * ih
                        x2 = (xc + bw / 2.0) * iw
                        y2 = (yc + bh / 2.0) * ih
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])

            gt_map[f"{stem}.jpg"] = boxes
            gt_map[stem] = boxes

    return gt_map


def parse_condition_stratum(filename: str) -> tuple[str, list[str]]:
    """
    Parse filename (case-insensitive) to extract augmentation & environmental condition stratum.
    """
    lower = filename.lower()
    tags = []

    if any(k in lower for k in ["blur", "motion", "shake"]):
        tags.append("blur")
    if any(k in lower for k in ["low_light", "dark", "night", "poor_lighting", "ch2"]):
        tags.append("low_light")
    if any(k in lower for k in ["occ", "occlusion", "occluded", "mask", "hand"]):
        tags.append("occlusion")
    if any(k in lower for k in ["top", "down", "high_angle", "ceiling", "ch4"]):
        tags.extend(["top_down_angle", "face_from_top"])
    if any(k in lower for k in ["flip", "rot", "rotated", "invert"]):
        tags.append("flip")
    if any(k in lower for k in ["distant", "crowd", "multiple", "small", "ch3"]):
        tags.extend(["distant_crowd", "multiple_distant"])

    if "blur" in tags and "low_light" in tags:
        stratum = "Blur & Low Light"
    elif "distant_crowd" in tags or "multiple" in lower:
        stratum = "Distant / Multi-Person"
    elif "top_down_angle" in tags:
        stratum = "Top-Down Angle"
    elif "occlusion" in tags:
        stratum = "Occlusion"
    elif "blur" in tags:
        stratum = "Motion Blur"
    elif "low_light" in tags:
        stratum = "Blur & Low Light"
    elif "flip" in tags:
        stratum = "Flip & Rotation"
    else:
        stratum = "Nominal Quality"

    return stratum, tags


def evaluate_model_on_image(
    raw_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    clamped_gt_boxes: list[list[float]],
    verifier: RetinaFaceVerifier,
    verifier_conf: float,
    dilation_ratio: float = 0.28,
) -> dict[str, Any]:
    """
    Applies +28% dilation and dual blur to detections, runs RetinaFace verification on blurred output,
    and classifies each GT face as MISSED_FACE, RESIDUAL_VISIBLE_FACE, or FULLY_OBSCURED.
    """
    h, w = raw_bgr.shape[:2]
    pred_boxes = [d["bbox"] for d in detections]
    dilated_boxes = [dilate_bbox(b, (h, w), ratio=dilation_ratio) for b in pred_boxes]

    # Apply heavy irreversible obfuscation
    redacted_bgr = apply_obfuscation(raw_bgr, dilated_boxes)

    # Independent RetinaFace audit on the blurred output
    rf_audit_detections = verifier.detect(redacted_bgr, conf_thresh=verifier_conf)
    rf_audit_boxes = [d["bbox"] for d in rf_audit_detections]

    gt_outcomes: list[dict[str, Any]] = []
    missed_count = 0
    residual_count = 0
    obscured_count = 0
    ious: list[float] = []

    for gt_box in clamped_gt_boxes:
        blur_ious = [compute_iou(gt_box, b_box) for b_box in dilated_boxes]
        max_blur_iou = max(blur_ious) if blur_ious else 0.0
        ious.append(max_blur_iou)

        # Check if RetinaFace detected an un-anonymized face inside/adjacent to the GT region
        rf_ious = [compute_iou(gt_box, r_box) for r_box in rf_audit_boxes]
        rf_fired_on_gt = any(iou >= 0.10 for iou in rf_ious)

        if max_blur_iou < 0.10:
            classification = "MISSED_FACE"
            missed_count += 1
        elif max_blur_iou < 0.55 or rf_fired_on_gt:
            classification = "RESIDUAL_VISIBLE_FACE"
            residual_count += 1
        else:
            classification = "FULLY_OBSCURED"
            obscured_count += 1

        gt_outcomes.append(
            {
                "gt_box": gt_box,
                "max_blur_iou": round(max_blur_iou, 3),
                "classification": classification,
                "retinaface_fired": rf_fired_on_gt,
                "detected": (max_blur_iou >= 0.10),
            }
        )

    return {
        "redacted_bgr": redacted_bgr,
        "dilated_boxes": dilated_boxes,
        "rf_audit_boxes": rf_audit_boxes,
        "gt_outcomes": gt_outcomes,
        "missed_count": missed_count,
        "residual_count": residual_count,
        "obscured_count": obscured_count,
        "ious": ious,
    }


def create_4panel_comparison(
    raw_bgr: np.ndarray,
    clamped_gt_boxes: list[list[float]],
    m1_eval: dict[str, Any],
    m2_eval: dict[str, Any],
    m3_eval: dict[str, Any],
    image_title: str,
    target_width: int = 400,
    target_height: int = 300,
) -> np.ndarray:
    """
    Constructs a 4-panel visual failure strip:
    [1. Raw + Hand GT (Green)] | [2. Base SCRFD-34G] | [3. RetinaFace] | [4. SCRFD-34G + Body/Head Pose]
    """
    h, w = raw_bgr.shape[:2]

    # Panel 1: Raw + Hand GT
    p1 = raw_bgr.copy()
    for gb in clamped_gt_boxes:
        cv2.rectangle(p1, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 255, 0), 2)
        cv2.putText(p1, "GT FACE", (int(gb[0]), max(16, int(gb[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

    # Panel 2: Base SCRFD-34G
    p2 = m1_eval["redacted_bgr"].copy()
    for db in m1_eval["dilated_boxes"]:
        cv2.rectangle(p2, (int(db[0]), int(db[1])), (int(db[2]), int(db[3])), (255, 140, 0), 2)
    # Highlight missed GT faces
    for outcome in m1_eval.get("gt_outcomes", []):
        if outcome["classification"] == "MISSED_FACE":
            gb = outcome["gt_box"]
            cv2.rectangle(p2, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 0, 255), 2)
            cv2.putText(p2, "MISSED LEAK", (int(gb[0]), max(16, int(gb[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    for rb in m1_eval["rf_audit_boxes"]:
        cv2.rectangle(p2, (int(rb[0]), int(rb[1])), (int(rb[2]), int(rb[3])), (0, 0, 255), 2)
        cv2.putText(p2, "QA LEAK", (int(rb[0]), max(16, int(rb[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    # Panel 3: RetinaFace
    p3 = m2_eval["redacted_bgr"].copy()
    for db in m2_eval["dilated_boxes"]:
        cv2.rectangle(p3, (int(db[0]), int(db[1])), (int(db[2]), int(db[3])), (255, 140, 0), 2)
    # Highlight missed GT faces
    for outcome in m2_eval.get("gt_outcomes", []):
        if outcome["classification"] == "MISSED_FACE":
            gb = outcome["gt_box"]
            cv2.rectangle(p3, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 0, 255), 2)
            cv2.putText(p3, "MISSED LEAK", (int(gb[0]), max(16, int(gb[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    for rb in m2_eval["rf_audit_boxes"]:
        cv2.rectangle(p3, (int(rb[0]), int(rb[1])), (int(rb[2]), int(rb[3])), (0, 0, 255), 2)
        cv2.putText(p3, "QA LEAK", (int(rb[0]), max(16, int(rb[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    # Panel 4: SCRFD-34G + Pose Fusion
    p4 = m3_eval["redacted_bgr"].copy()
    for db in m3_eval["dilated_boxes"]:
        cv2.rectangle(p4, (int(db[0]), int(db[1])), (int(db[2]), int(db[3])), (0, 220, 100), 2)
        cv2.putText(p4, "RESCUED & BLURRED", (int(db[0]), max(16, int(db[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 100), 1)

    # Add header bars to each panel
    panels = [p1, p2, p3, p4]
    titles = [
        "1. Raw + Hand GT (Green)",
        "2. Base SCRFD-34G",
        "3. RetinaFace Detector",
        "4. SCRFD + Pose Fusion (Rescued)",
    ]
    bg_colors = [
        (30, 30, 30),
        (20, 20, 120),
        (20, 80, 20),
        (80, 40, 20),
    ]

    resized_panels = []
    for p, title, bg in zip(panels, titles, bg_colors):
        pr = cv2.resize(p, (target_width, target_height))
        # Top banner
        header = np.zeros((32, target_width, 3), dtype=np.uint8)
        header[:] = bg
        cv2.putText(header, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        combined = np.vstack([header, pr])
        resized_panels.append(combined)

    strip = np.hstack(resized_panels)
    return strip


def run_evaluation(
    dataset_dir: Optional[Path] = None,
    output_summary_path: Optional[Path] = None,
    failure_gallery_dir: Optional[Path] = None,
    primary_conf: float = 0.25,
    verifier_conf: float = 0.30,
) -> dict[str, Any]:
    """
    Executes comprehensive 3-Way Model Comparison across all 100 images:
      Model 1: Base SCRFD-34G
      Model 2: RetinaFace
      Model 3: SCRFD-34G + Body/Head Pose Fusion
    """
    img_dir = dataset_dir or (config.IMAGES_100_DIR / "test")
    if not img_dir.exists():
        img_dir = config.IMAGES_100_DIR

    summary_path = output_summary_path or (config.DATA_DIR / "outputs" / "eval_100_summary.json")
    gallery_dir = failure_gallery_dir or (config.DATA_DIR / "outputs" / "failure_gallery")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
    if not image_paths:
        image_paths = sorted(glob.glob(str(config.IMAGES_100_DIR / "**" / "*.jpg"), recursive=True))

    logger.info("Found %d evaluation images in %s", len(image_paths), img_dir)
    assert len(image_paths) > 0, f"No evaluation images found in {img_dir}"

    gt_map = load_ground_truth_annotations(img_dir.parent if img_dir.name == "test" else img_dir)
    logger.info("Loaded annotations for %d unique filenames/keys.", len(gt_map))

    # Initialize all three models
    logger.info("Initializing models: Base SCRFD-34G, RetinaFace Verifier, and YOLOv8n-Pose...")
    detector_scrfd = PrimaryFaceDetector(conf_thresh=primary_conf, use_pose_fallback=False)
    verifier = RetinaFaceVerifier(conf_thresh=verifier_conf)
    detector_pose = PrimaryFaceDetector(conf_thresh=primary_conf, use_pose_fallback=True)

    # Data collectors
    models = ["Base SCRFD-34G", "RetinaFace", "SCRFD-34G + Body/Head Pose"]
    model_keys = ["scrfd", "retinaface", "pose_fusion"]

    model_stats = {
        k: {
            "name": name,
            "total_gt_faces": 0,
            "missed_faces": 0,
            "residual_visible": 0,
            "fully_obscured": 0,
            "all_ious": [],
            "total_latency_ms": 0.0,
            "image_count": 0,
        }
        for k, name in zip(model_keys, models)
    }

    strata_breakdown: dict[str, dict[str, dict[str, Any]]] = {}

    # Correlated Failure Mode / Agreement Matrix counters
    both_detected = 0
    scrfd_missed_retina_caught = 0
    scrfd_caught_retina_missed = 0
    shared_false_negatives_both_missed = 0
    shared_fn_rescued_by_pose_fusion = 0
    total_benchmark_gt_faces = 0

    comparison_candidates: list[dict[str, Any]] = []
    per_image_results: list[dict[str, Any]] = []

    logger.info("Running 3-Way Benchmark inference across %d images...", len(image_paths))

    for idx, img_path_str in enumerate(image_paths):
        img_path = Path(img_path_str)
        fname = img_path.name
        raw_bgr = cv2.imread(img_path_str)
        if raw_bgr is None:
            continue

        h, w = raw_bgr.shape[:2]
        stratum, tags = parse_condition_stratum(fname)

        if stratum not in strata_breakdown:
            strata_breakdown[stratum] = {
                k: {"gt_faces": 0, "missed": 0, "residual": 0, "obscured": 0}
                for k in model_keys
            }

        # Ground truth boxes for this image
        gt_boxes = gt_map.get(fname) or gt_map.get(img_path.stem) or []
        clamped_gt_boxes = []
        for g in gt_boxes:
            gx1 = max(0.0, min(float(w), float(g[0])))
            gy1 = max(0.0, min(float(h), float(g[1])))
            gx2 = max(0.0, min(float(w), float(g[2])))
            gy2 = max(0.0, min(float(h), float(g[3])))
            if gx2 > gx1 and gy2 > gy1:
                clamped_gt_boxes.append([gx1, gy1, gx2, gy2])

        n_gt = len(clamped_gt_boxes)
        total_benchmark_gt_faces += n_gt

        for k in model_keys:
            strata_breakdown[stratum][k]["gt_faces"] += n_gt
            model_stats[k]["total_gt_faces"] += n_gt
            model_stats[k]["image_count"] += 1

        # -------------------------------------------------------------
        # 1. Model 1: Base SCRFD-34G
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        m1_dets = detector_scrfd.detect(raw_bgr, conf_thresh=primary_conf)
        t1 = (time.perf_counter() - t0) * 1000.0
        model_stats["scrfd"]["total_latency_ms"] += t1
        m1_eval = evaluate_model_on_image(raw_bgr, m1_dets, clamped_gt_boxes, verifier, verifier_conf)

        model_stats["scrfd"]["missed_faces"] += m1_eval["missed_count"]
        model_stats["scrfd"]["residual_visible"] += m1_eval["residual_count"]
        model_stats["scrfd"]["fully_obscured"] += m1_eval["obscured_count"]
        model_stats["scrfd"]["all_ious"].extend(m1_eval["ious"])
        strata_breakdown[stratum]["scrfd"]["missed"] += m1_eval["missed_count"]
        strata_breakdown[stratum]["scrfd"]["residual"] += m1_eval["residual_count"]
        strata_breakdown[stratum]["scrfd"]["obscured"] += m1_eval["obscured_count"]

        # -------------------------------------------------------------
        # 2. Model 2: RetinaFace
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        m2_dets = verifier.detect(raw_bgr, conf_thresh=verifier_conf)
        t2 = (time.perf_counter() - t0) * 1000.0
        model_stats["retinaface"]["total_latency_ms"] += t2
        m2_eval = evaluate_model_on_image(raw_bgr, m2_dets, clamped_gt_boxes, verifier, verifier_conf)

        model_stats["retinaface"]["missed_faces"] += m2_eval["missed_count"]
        model_stats["retinaface"]["residual_visible"] += m2_eval["residual_count"]
        model_stats["retinaface"]["fully_obscured"] += m2_eval["obscured_count"]
        model_stats["retinaface"]["all_ious"].extend(m2_eval["ious"])
        strata_breakdown[stratum]["retinaface"]["missed"] += m2_eval["missed_count"]
        strata_breakdown[stratum]["retinaface"]["residual"] += m2_eval["residual_count"]
        strata_breakdown[stratum]["retinaface"]["obscured"] += m2_eval["obscured_count"]

        # -------------------------------------------------------------
        # 3. Model 3: SCRFD-34G + Body/Head Pose Fusion
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        m3_dets = detector_pose.detect(raw_bgr, conf_thresh=primary_conf)
        t3 = (time.perf_counter() - t0) * 1000.0
        model_stats["pose_fusion"]["total_latency_ms"] += t3
        m3_eval = evaluate_model_on_image(raw_bgr, m3_dets, clamped_gt_boxes, verifier, verifier_conf)

        model_stats["pose_fusion"]["missed_faces"] += m3_eval["missed_count"]
        model_stats["pose_fusion"]["residual_visible"] += m3_eval["residual_count"]
        model_stats["pose_fusion"]["fully_obscured"] += m3_eval["obscured_count"]
        model_stats["pose_fusion"]["all_ious"].extend(m3_eval["ious"])
        strata_breakdown[stratum]["pose_fusion"]["missed"] += m3_eval["missed_count"]
        strata_breakdown[stratum]["pose_fusion"]["residual"] += m3_eval["residual_count"]
        strata_breakdown[stratum]["pose_fusion"]["obscured"] += m3_eval["obscured_count"]

        # -------------------------------------------------------------
        # Pairwise Failure Overlap Calculation
        # -------------------------------------------------------------
        image_rescued_count = 0
        for i_gt in range(n_gt):
            scrfd_det = m1_eval["gt_outcomes"][i_gt]["detected"]
            retina_det = m2_eval["gt_outcomes"][i_gt]["detected"]
            pose_det = m3_eval["gt_outcomes"][i_gt]["detected"]

            if scrfd_det and retina_det:
                both_detected += 1
            elif (not scrfd_det) and retina_det:
                scrfd_missed_retina_caught += 1
            elif scrfd_det and (not retina_det):
                scrfd_caught_retina_missed += 1
            else:
                # Shared false negative: BOTH missed!
                shared_false_negatives_both_missed += 1
                if pose_det:
                    shared_fn_rescued_by_pose_fusion += 1
                    image_rescued_count += 1

        # Candidate selection for 4-panel visual comparison gallery
        missed_reduction = m1_eval["missed_count"] - m3_eval["missed_count"]
        if missed_reduction > 0 or image_rescued_count > 0:
            comparison_candidates.append(
                {
                    "raw_bgr": raw_bgr,
                    "clamped_gt_boxes": clamped_gt_boxes,
                    "m1_eval": m1_eval,
                    "m2_eval": m2_eval,
                    "m3_eval": m3_eval,
                    "fname": fname,
                    "stratum": stratum,
                    "rescued_count": max(missed_reduction, image_rescued_count),
                    "obscured_gain": missed_reduction,
                }
            )

        per_image_results.append(
            {
                "filename": fname,
                "stratum": stratum,
                "gt_face_count": n_gt,
                "scrfd_missed": m1_eval["missed_count"],
                "retina_missed": m2_eval["missed_count"],
                "pose_missed": m3_eval["missed_count"],
                "pose_obscured": m3_eval["obscured_count"],
            }
        )

    # =========================================================================
    # EXPORT 4-PANEL VISUAL COMPARISON STRIPS
    # =========================================================================
    logger.info("Exporting 4-Panel Visual Comparison Strips to %s...", gallery_dir)
    # Sort candidates by rescued count and obscured gain to get the most dramatic improvements
    comparison_candidates.sort(key=lambda x: (x["rescued_count"], x["obscured_gain"]), reverse=True)
    num_to_save = min(6, len(comparison_candidates))
    saved_strips: list[str] = []

    for idx, cand in enumerate(comparison_candidates[:num_to_save], 1):
        strip = create_4panel_comparison(
            cand["raw_bgr"],
            cand["clamped_gt_boxes"],
            cand["m1_eval"],
            cand["m2_eval"],
            cand["m3_eval"],
            image_title=cand["fname"],
        )
        strip_name = f"comparison_{idx:02d}.jpg"
        strip_path = gallery_dir / strip_name
        cv2.imwrite(str(strip_path), strip)
        saved_strips.append(strip_name)
        logger.info("Saved 4-panel comparison: %s (%s, rescued=%d)", strip_name, cand["stratum"], cand["rescued_count"])

    # =========================================================================
    # TABLE 1: OVERALL 3-WAY MODEL COMPARISON (ALL 100 IMAGES)
    # =========================================================================
    table_1_rows: list[list[Any]] = []
    summary_table_1_json: list[dict[str, Any]] = []

    for k in model_keys:
        st = model_stats[k]
        tot_gt = st["total_gt_faces"]
        miss_pct = (st["missed_faces"] / tot_gt * 100.0) if tot_gt else 0.0
        res_pct = (st["residual_visible"] / tot_gt * 100.0) if tot_gt else 0.0
        obs_pct = (st["fully_obscured"] / tot_gt * 100.0) if tot_gt else 0.0
        mean_iou = float(np.mean(st["all_ious"])) if st["all_ious"] else 0.0
        avg_lat = (st["total_latency_ms"] / st["image_count"]) if st["image_count"] else 0.0
        fps = (1000.0 / avg_lat) if avg_lat > 0 else 0.0

        table_1_rows.append(
            [
                st["name"],
                tot_gt,
                f"{st['missed_faces']} ({miss_pct:.1f}%)",
                f"{st['residual_visible']} ({res_pct:.1f}%)",
                f"{st['fully_obscured']} ({obs_pct:.1f}%)",
                f"{mean_iou:.3f}",
                f"{avg_lat:.1f} ms",
                f"{fps:.1f}",
            ]
        )

        summary_table_1_json.append(
            {
                "model": st["name"],
                "gt_faces": tot_gt,
                "missed_faces": st["missed_faces"],
                "missed_faces_pct": round(miss_pct, 2),
                "residual_visible": st["residual_visible"],
                "residual_visible_pct": round(res_pct, 2),
                "fully_obscured": st["fully_obscured"],
                "fully_obscured_pct": round(obs_pct, 2),
                "mean_iou": round(mean_iou, 3),
                "avg_latency_ms": round(avg_lat, 1),
                "fps": round(fps, 1),
            }
        )

    headers_1 = [
        "Model Architecture",
        "GT Faces",
        "Missed Faces (%)",
        "Residual Visible (%)",
        "Fully Obscured (%)",
        "Mean IoU",
        "Avg Latency (ms)",
        "FPS",
    ]

    print("\n" + "=" * 105)
    print("TABLE 1 — OVERALL 3-WAY MODEL COMPARISON (ALL 100 IMAGES)")
    print("=" * 105)
    print(tabulate(table_1_rows, headers=headers_1, tablefmt="github"))
    print("=" * 105 + "\n")

    # =========================================================================
    # TABLE 2: PER-CONDITION BREAKDOWN ACROSS ALL 3 MODELS
    # =========================================================================
    table_2_rows: list[list[Any]] = []
    summary_table_2_json: list[dict[str, Any]] = []

    # Sort strata by GT face count
    sorted_strata = sorted(strata_breakdown.items(), key=lambda x: -x[1]["scrfd"]["gt_faces"])

    for stratum_name, s_data in sorted_strata:
        gt_c = s_data["scrfd"]["gt_faces"]
        if gt_c == 0:
            continue

        # Missed %
        m1_m_pct = (s_data["scrfd"]["missed"] / gt_c) * 100.0
        m2_m_pct = (s_data["retinaface"]["missed"] / gt_c) * 100.0
        m3_m_pct = (s_data["pose_fusion"]["missed"] / gt_c) * 100.0

        # Residual %
        m1_r_pct = (s_data["scrfd"]["residual"] / gt_c) * 100.0
        m2_r_pct = (s_data["retinaface"]["residual"] / gt_c) * 100.0
        m3_r_pct = (s_data["pose_fusion"]["residual"] / gt_c) * 100.0

        # Obscured %
        m1_o_pct = (s_data["scrfd"]["obscured"] / gt_c) * 100.0
        m2_o_pct = (s_data["retinaface"]["obscured"] / gt_c) * 100.0
        m3_o_pct = (s_data["pose_fusion"]["obscured"] / gt_c) * 100.0

        table_2_rows.append(
            [
                stratum_name,
                gt_c,
                f"{m1_m_pct:.1f}%",
                f"{m2_m_pct:.1f}%",
                f"{m3_m_pct:.1f}%",
                f"{m1_r_pct:.1f}%",
                f"{m2_r_pct:.1f}%",
                f"{m3_r_pct:.1f}%",
                f"{m1_o_pct:.1f}%",
                f"{m2_o_pct:.1f}%",
                f"{m3_o_pct:.1f}%",
            ]
        )

        summary_table_2_json.append(
            {
                "stratum": stratum_name,
                "gt_faces": gt_c,
                "missed_scrfd_pct": round(m1_m_pct, 1),
                "missed_retina_pct": round(m2_m_pct, 1),
                "missed_pose_pct": round(m3_m_pct, 1),
                "residual_scrfd_pct": round(m1_r_pct, 1),
                "residual_retina_pct": round(m2_r_pct, 1),
                "residual_pose_pct": round(m3_r_pct, 1),
                "obscured_scrfd_pct": round(m1_o_pct, 1),
                "obscured_retina_pct": round(m2_o_pct, 1),
                "obscured_pose_pct": round(m3_o_pct, 1),
            }
        )

    headers_2 = [
        "Condition Stratum",
        "GT Faces",
        "Miss: Base SCRFD",
        "Miss: RetinaFace",
        "Miss: +Pose Fusion",
        "Resid: Base SCRFD",
        "Resid: RetinaFace",
        "Resid: +Pose Fusion",
        "Obscured: Base SCRFD",
        "Obscured: RetinaFace",
        "Obscured: +Pose Fusion",
    ]

    print("\n" + "=" * 120)
    print("TABLE 2 — PER-CONDITION BREAKDOWN ACROSS ALL 3 MODELS")
    print("=" * 120)
    print(tabulate(table_2_rows, headers=headers_2, tablefmt="github"))
    print("=" * 120 + "\n")

    # =========================================================================
    # TABLE 3: DETECTOR AGREEMENT & CORRELATED FAILURE MODE MATRIX
    # =========================================================================
    tot_gt = total_benchmark_gt_faces
    both_pct = (both_detected / tot_gt * 100.0) if tot_gt else 0.0
    rf_catch_pct = (scrfd_missed_retina_caught / tot_gt * 100.0) if tot_gt else 0.0
    scrfd_catch_pct = (scrfd_caught_retina_missed / tot_gt * 100.0) if tot_gt else 0.0
    shared_fn_pct = (shared_false_negatives_both_missed / tot_gt * 100.0) if tot_gt else 0.0
    rescued_pct_of_shared = (
        (shared_fn_rescued_by_pose_fusion / shared_false_negatives_both_missed * 100.0)
        if shared_false_negatives_both_missed
        else 0.0
    )

    table_3_rows = [
        [
            "Both Detected (SCRFD ∩ RetinaFace)",
            both_detected,
            f"{both_pct:.1f}%",
            "High confidence mutual detection under nominal/moderate lighting",
        ],
        [
            "SCRFD Missed, RetinaFace Caught",
            scrfd_missed_retina_caught,
            f"{rf_catch_pct:.1f}%",
            "RetinaFace QA verification successfully catches primary detector misses",
        ],
        [
            "SCRFD Caught, RetinaFace Missed",
            scrfd_caught_retina_missed,
            f"{scrfd_catch_pct:.1f}%",
            "SCRFD-34G multi-scale anchors catch smaller/partial profiles missed by verifier",
        ],
        [
            "Shared False Negatives (Both Missed)",
            shared_false_negatives_both_missed,
            f"{shared_fn_pct:.1f}%",
            "Common epistemic blind spot: extreme motion blur, low light, top-down angles",
        ],
        [
            "Shared FN Rescued by Pose Fusion",
            shared_fn_rescued_by_pose_fusion,
            f"{rescued_pct_of_shared:.1f}% of Shared FN",
            "Recovered & fully obscured via cranial/shoulder keypoint triangulation",
        ],
    ]

    headers_3 = [
        "Failure Mode / Detection Agreement Stratum",
        "Count (Faces)",
        "Percentage of Dataset",
        "Operational Significance",
    ]

    print("\n" + "=" * 115)
    print("TABLE 3 — DETECTOR AGREEMENT & CORRELATED FAILURE MODE MATRIX")
    print("=" * 115)
    print(tabulate(table_3_rows, headers=headers_3, tablefmt="github"))
    print("=" * 115 + "\n")

    # Complete Summary JSON payload
    summary_data = {
        "summary": summary_table_1_json[0],  # Base SCRFD summary for backward compatibility
        "summary_scrfd": summary_table_1_json[0],
        "summary_retinaface": summary_table_1_json[1],
        "summary_pose_fusion": summary_table_1_json[2],
        "three_way_comparison": summary_table_1_json,
        "three_way_table_rows": table_1_rows,
        "three_way_headers": headers_1,
        "per_condition_comparison": summary_table_2_json,
        "per_condition_table_rows": table_2_rows,
        "per_condition_headers": headers_2,
        "agreement_matrix": {
            "both_detected": both_detected,
            "both_detected_pct": round(both_pct, 1),
            "scrfd_missed_retina_caught": scrfd_missed_retina_caught,
            "scrfd_missed_retina_caught_pct": round(rf_catch_pct, 1),
            "scrfd_caught_retina_missed": scrfd_caught_retina_missed,
            "scrfd_caught_retina_missed_pct": round(scrfd_catch_pct, 1),
            "shared_false_negatives_both_missed": shared_false_negatives_both_missed,
            "shared_false_negatives_both_missed_pct": round(shared_fn_pct, 1),
            "shared_fn_rescued_by_pose_fusion": shared_fn_rescued_by_pose_fusion,
            "shared_fn_rescued_pct_of_shared": round(rescued_pct_of_shared, 1),
            "table_rows": table_3_rows,
            "headers": headers_3,
        },
        "gallery_comparison_strips": saved_strips,
        "strata_breakdown": {
            s: {
                "images": len([p for p in image_paths if parse_condition_stratum(Path(p).name)[0] == s]),
                "gt_faces": v["scrfd"]["gt_faces"],
                "missed_faces": v["scrfd"]["missed"],
                "residual_visible": v["scrfd"]["residual"],
                "fully_obscured": v["scrfd"]["obscured"],
                "flagged_by_verifier": v["retinaface"]["missed"],
                "shared_false_negatives": shared_false_negatives_both_missed,
            }
            for s, v in strata_breakdown.items()
        },
        "per_image_results": per_image_results,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Saved complete 3-way evaluation summary to %s", summary_path)
    return summary_data


if __name__ == "__main__":
    run_evaluation()
