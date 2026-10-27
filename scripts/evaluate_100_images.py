"""Automated Dual Evaluation Pipeline for Hand-Annotated Images.
Evaluates Primary Anonymizer against Ground-Truth faces and RetinaFace QA Verifier,
generating the failure taxonomy, correlated blind-spot metrics, failure gallery,
and the Before vs. After Ablation Comparison (SCRFD-34G vs. SCRFD-34G + YOLOv8-Pose Fusion).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.pipeline.detector import PrimaryFaceDetector
from app.pipeline.redactor import redact_single_image
from app.pipeline.tracker_interpolator import compute_iou
from app.pipeline.verifier import RetinaFaceVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_100")


def load_ground_truth_annotations(dataset_dir: Path) -> dict[str, list[list[float]]]:
    """
    Flexible Ground-Truth loader supporting:
    1. COCO JSON (_annotations.coco.json / annotations.json).
    2. YOLO normalized .txt files matching each image stem.
    3. Custom CSV/JSON bounding box formats.

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


def evaluate_single_mode(
    detector: PrimaryFaceDetector,
    verifier: RetinaFaceVerifier,
    image_paths: list[str],
    gt_map: dict[str, list[list[float]]],
    primary_conf: float,
    verifier_conf: float,
    save_gallery: bool = False,
    gallery_dir: Optional[Path] = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """
    Evaluates detector against dataset and verifier.
    """
    strata_stats: dict[str, dict[str, Any]] = {}
    per_image_results: list[dict[str, Any]] = []

    total_gt_faces = 0
    total_missed = 0
    total_residual = 0
    total_fully_obscured = 0
    total_flagged_by_verifier = 0
    total_shared_false_negatives = 0
    gallery_count = 0

    for idx, img_path_str in enumerate(image_paths):
        img_path = Path(img_path_str)
        fname = img_path.name

        raw_bgr = cv2.imread(img_path_str)
        if raw_bgr is None:
            continue

        h, w = raw_bgr.shape[:2]
        stratum, tags = parse_condition_stratum(fname)

        if stratum not in strata_stats:
            strata_stats[stratum] = {
                "images": 0,
                "gt_faces": 0,
                "missed_faces": 0,
                "residual_visible": 0,
                "fully_obscured": 0,
                "flagged_by_verifier": 0,
                "shared_false_negatives": 0,
            }
        strata_stats[stratum]["images"] += 1

        # 1. Primary Anonymization
        redacted_bgr, detection_records = redact_single_image(
            raw_bgr,
            detector=detector,
            conf_thresh=primary_conf,
            dilation_ratio=config.BBOX_DILATION_RATIO,
        )
        applied_blur_boxes = [d["dilated_bbox"] for d in detection_records]

        # 2. Independent RetinaFace Verification
        rf_detections = verifier.detect(redacted_bgr, conf_thresh=verifier_conf)
        rf_boxes = [d["bbox"] for d in rf_detections]

        # 3. Ground Truth Matching
        gt_boxes = gt_map.get(fname) or gt_map.get(img_path.stem) or []
        clamped_gt_boxes = []
        for g in gt_boxes:
            gx1 = max(0.0, min(float(w), float(g[0])))
            gy1 = max(0.0, min(float(h), float(g[1])))
            gx2 = max(0.0, min(float(w), float(g[2])))
            gy2 = max(0.0, min(float(h), float(g[3])))
            if gx2 > gx1 and gy2 > gy1:
                clamped_gt_boxes.append([gx1, gy1, gx2, gy2])

        strata_stats[stratum]["gt_faces"] += len(clamped_gt_boxes)
        total_gt_faces += len(clamped_gt_boxes)

        img_has_leak = False
        image_eval_records = []

        for gt_box in clamped_gt_boxes:
            blur_ious = [compute_iou(gt_box, b_box) for b_box in applied_blur_boxes]
            max_blur_iou = max(blur_ious) if blur_ious else 0.0

            rf_ious = [compute_iou(gt_box, r_box) for r_box in rf_boxes]
            rf_fired_on_gt = any(iou >= 0.10 for iou in rf_ious)

            if max_blur_iou < 0.10:
                classification = "MISSED_FACE"
                strata_stats[stratum]["missed_faces"] += 1
                total_missed += 1
                img_has_leak = True

                if not rf_fired_on_gt:
                    strata_stats[stratum]["shared_false_negatives"] += 1
                    total_shared_false_negatives += 1

            elif rf_fired_on_gt:
                classification = "RESIDUAL_VISIBLE_FACE"
                strata_stats[stratum]["residual_visible"] += 1
                total_residual += 1
                img_has_leak = True
            else:
                classification = "FULLY_OBSCURED"
                strata_stats[stratum]["fully_obscured"] += 1
                total_fully_obscured += 1

            image_eval_records.append(
                {
                    "gt_box": gt_box,
                    "max_blur_iou": round(max_blur_iou, 3),
                    "classification": classification,
                    "retinaface_detected": rf_fired_on_gt,
                }
            )

        if len(rf_boxes) > 0:
            strata_stats[stratum]["flagged_by_verifier"] += len(rf_boxes)
            total_flagged_by_verifier += len(rf_boxes)

        # Export failure composite if requested
        if save_gallery and gallery_dir and (img_has_leak or len(rf_boxes) > 0) and gallery_count < 30:
            left_pane = raw_bgr.copy()
            for gb in clamped_gt_boxes:
                cv2.rectangle(
                    left_pane,
                    (int(gb[0]), int(gb[1])),
                    (int(gb[2]), int(gb[3])),
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                left_pane,
                f"RAW + GT (Green: {len(clamped_gt_boxes)})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            right_pane = redacted_bgr.copy()
            for ab in applied_blur_boxes:
                cv2.rectangle(
                    right_pane,
                    (int(ab[0]), int(ab[1])),
                    (int(ab[2]), int(ab[3])),
                    (255, 0, 0),
                    2,
                )
            for rb in rf_boxes:
                cv2.rectangle(
                    right_pane,
                    (int(rb[0]), int(rb[1])),
                    (int(rb[2]), int(rb[3])),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    right_pane,
                    "QA LEAK",
                    (int(rb[0]), max(15, int(rb[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            cv2.putText(
                right_pane,
                "REDACTED + QA (Blur: Blue | Leak: Red)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            composite = np.hstack([left_pane, right_pane])
            comp_name = f"{stratum.replace(' ', '_').replace('/', '_')}_{fname}"
            comp_path = gallery_dir / comp_name
            cv2.imwrite(str(comp_path), composite)
            gallery_count += 1

        per_image_results.append(
            {
                "filename": fname,
                "stratum": stratum,
                "tags": tags,
                "gt_face_count": len(clamped_gt_boxes),
                "applied_blur_count": len(applied_blur_boxes),
                "retinaface_leak_count": len(rf_boxes),
                "outcomes": image_eval_records,
            }
        )

    overall_missed_pct = (total_missed / total_gt_faces * 100.0) if total_gt_faces > 0 else 0.0
    overall_residual_pct = (total_residual / total_gt_faces * 100.0) if total_gt_faces > 0 else 0.0
    overall_obscured_pct = (total_fully_obscured / total_gt_faces * 100.0) if total_gt_faces > 0 else 100.0

    summary = {
        "total_images": len(image_paths),
        "total_gt_faces": total_gt_faces,
        "missed_faces": total_missed,
        "missed_faces_pct": round(overall_missed_pct, 2),
        "residual_visible": total_residual,
        "residual_visible_pct": round(overall_residual_pct, 2),
        "fully_obscured": total_fully_obscured,
        "fully_obscured_pct": round(overall_obscured_pct, 2),
        "flagged_by_verifier": total_flagged_by_verifier,
        "shared_false_negatives": total_shared_false_negatives,
        "gallery_saved_count": gallery_count,
        "primary_model": detector.model_name,
        "primary_conf": primary_conf,
        "verifier_conf": verifier_conf,
    }

    return summary, strata_stats, per_image_results


def run_evaluation(
    dataset_dir: Optional[Path] = None,
    output_summary_path: Optional[Path] = None,
    failure_gallery_dir: Optional[Path] = None,
    primary_conf: float = 0.25,
    verifier_conf: float = 0.30,
) -> dict[str, Any]:
    """
    Executes full evaluation in two passes:
      Mode A: Baseline SCRFD-34G
      Mode B: SCRFD-34G + YOLOv8-Pose Body/Head Fusion
    Prints both the Baseline Breakdown and the Before vs. After Ablation Comparison Table.
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

    verifier = RetinaFaceVerifier(conf_thresh=verifier_conf)

    # =========================================================================
    # PASS 1: Mode A - Baseline SCRFD-34G
    # =========================================================================
    logger.info(">>> Running Mode A: Baseline SCRFD-34G Evaluation...")
    detector_baseline = PrimaryFaceDetector(conf_thresh=primary_conf, use_pose_fallback=False)
    summary_base, strata_base, per_image_base = evaluate_single_mode(
        detector=detector_baseline,
        verifier=verifier,
        image_paths=image_paths,
        gt_map=gt_map,
        primary_conf=primary_conf,
        verifier_conf=verifier_conf,
        save_gallery=True,
        gallery_dir=gallery_dir,
    )

    # =========================================================================
    # PASS 2: Mode B - SCRFD-34G + YOLOv8-Pose Body/Head Fusion
    # =========================================================================
    logger.info(">>> Running Mode B: SCRFD-34G + YOLOv8-Pose Head Fusion Evaluation...")
    detector_fusion = PrimaryFaceDetector(conf_thresh=primary_conf, use_pose_fallback=True)
    summary_fusion, strata_fusion, per_image_fusion = evaluate_single_mode(
        detector=detector_fusion,
        verifier=verifier,
        image_paths=image_paths,
        gt_map=gt_map,
        primary_conf=primary_conf,
        verifier_conf=verifier_conf,
        save_gallery=False,
    )

    # =========================================================================
    # PRINT REPORT 1: Baseline SCRFD-34G Failure Taxonomy
    # =========================================================================
    table_base: list[list[Any]] = []
    for st, m in sorted(strata_base.items(), key=lambda x: -x[1]["gt_faces"]):
        gt_count = m["gt_faces"]
        missed_pct = (m["missed_faces"] / gt_count * 100.0) if gt_count > 0 else 0.0
        residual_pct = (m["residual_visible"] / gt_count * 100.0) if gt_count > 0 else 0.0
        obscured_pct = (m["fully_obscured"] / gt_count * 100.0) if gt_count > 0 else 100.0
        table_base.append(
            [
                st,
                m["images"],
                gt_count,
                f"{m['missed_faces']} ({missed_pct:.1f}%)",
                f"{m['residual_visible']} ({residual_pct:.1f}%)",
                f"{m['fully_obscured']} ({obscured_pct:.1f}%)",
                m["flagged_by_verifier"],
                m["shared_false_negatives"],
            ]
        )

    s_b = summary_base
    table_base.append(
        [
            "**OVERALL TOTALS**",
            s_b["total_images"],
            s_b["total_gt_faces"],
            f"**{s_b['missed_faces']} ({s_b['missed_faces_pct']:.1f}%)**",
            f"**{s_b['residual_visible']} ({s_b['residual_visible_pct']:.1f}%)**",
            f"**{s_b['fully_obscured']} ({s_b['fully_obscured_pct']:.1f}%)**",
            f"**{s_b['flagged_by_verifier']}**",
            f"**{s_b['shared_false_negatives']}**",
        ]
    )

    headers_base = [
        "Stratum / Condition",
        "Images",
        "GT Faces",
        "Missed Faces (%)",
        "Residual Visible (%)",
        "Fully Obscured (%)",
        "Flagged by Verifier",
        "Shared FN (Both Missed)",
    ]

    print("\n" + "=" * 90)
    print("REPORT 1: BASELINE SCRFD-34G PER-CONDITION FAILURE BREAKDOWN")
    print("=" * 90)
    print(tabulate(table_base, headers=headers_base, tablefmt="github"))
    print("=" * 90 + "\n")

    # =========================================================================
    # PRINT REPORT 2: Before vs. After Ablation Comparison Table
    # =========================================================================
    ablation_rows: list[list[Any]] = []
    all_strata = sorted(list(strata_base.keys()), key=lambda k: -strata_base[k]["gt_faces"])

    for st in all_strata:
        b_m = strata_base.get(st, {})
        f_m = strata_fusion.get(st, {})
        gt_count = b_m.get("gt_faces", 0)

        b_miss_pct = (b_m.get("missed_faces", 0) / gt_count * 100.0) if gt_count else 0.0
        f_miss_pct = (f_m.get("missed_faces", 0) / gt_count * 100.0) if gt_count else 0.0
        miss_delta = f_miss_pct - b_miss_pct

        b_res_pct = (b_m.get("residual_visible", 0) / gt_count * 100.0) if gt_count else 0.0
        f_res_pct = (f_m.get("residual_visible", 0) / gt_count * 100.0) if gt_count else 0.0
        res_delta = f_res_pct - b_res_pct

        b_fn = b_m.get("shared_false_negatives", 0)
        f_fn = f_m.get("shared_false_negatives", 0)
        fn_delta = f_fn - b_fn

        ablation_rows.append(
            [
                st,
                gt_count,
                f"{b_miss_pct:.1f}%",
                f"{f_miss_pct:.1f}% ({miss_delta:+.1f}%)",
                f"{b_res_pct:.1f}%",
                f"{f_res_pct:.1f}% ({res_delta:+.1f}%)",
                b_fn,
                f"{f_fn} ({fn_delta:+d})",
            ]
        )

    # Overall Ablation Totals
    tot_gt = s_b["total_gt_faces"]
    s_f = summary_fusion
    tot_miss_delta = s_f["missed_faces_pct"] - s_b["missed_faces_pct"]
    tot_res_delta = s_f["residual_visible_pct"] - s_b["residual_visible_pct"]
    tot_fn_delta = s_f["shared_false_negatives"] - s_b["shared_false_negatives"]

    ablation_rows.append(
        [
            "**OVERALL TOTALS**",
            tot_gt,
            f"**{s_b['missed_faces_pct']:.1f}%**",
            f"**{s_f['missed_faces_pct']:.1f}% ({tot_miss_delta:+.1f}%)**",
            f"**{s_b['residual_visible_pct']:.1f}%**",
            f"**{s_f['residual_visible_pct']:.1f}% ({tot_res_delta:+.1f}%)**",
            f"**{s_b['shared_false_negatives']}**",
            f"**{s_f['shared_false_negatives']} ({tot_fn_delta:+d})**",
        ]
    )

    headers_ablation = [
        "Stratum / Condition",
        "GT Faces",
        "Missed: Baseline",
        "Missed: +Pose Fusion (Δ)",
        "Residual: Baseline",
        "Residual: +Pose Fusion (Δ)",
        "Shared FN: Base",
        "Shared FN: Fusion (Δ)",
    ]

    print("\n" + "=" * 95)
    print("REPORT 2: BEFORE vs. AFTER ABLATION (SCRFD-34G vs. SCRFD-34G + YOLOv8-Pose Fusion)")
    print("=" * 95)
    print(tabulate(ablation_rows, headers=headers_ablation, tablefmt="github"))
    print("=" * 95 + "\n")

    # Complete Summary JSON payload
    summary_data = {
        "summary": summary_base,  # Baseline for backward-compatible dashboard display
        "strata_breakdown": strata_base,
        "per_image_results": per_image_base,
        "ablation_comparison": {
            "baseline": summary_base,
            "pose_fusion": summary_fusion,
            "overall_missed_delta_pct": round(tot_miss_delta, 2),
            "overall_shared_fn_delta": tot_fn_delta,
            "table_rows": ablation_rows,
            "headers": headers_ablation,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Saved complete evaluation and ablation summary to %s", summary_path)
    return summary_data


if __name__ == "__main__":
    run_evaluation()
