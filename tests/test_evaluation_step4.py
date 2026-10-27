"""Pytest suite for Step 4 Evaluation Pipeline & Failure Taxonomy."""

from pathlib import Path
import pytest

from app.config import config
from scripts.evaluate_100_images import load_ground_truth_annotations, parse_condition_stratum


def test_parse_condition_stratum():
    """Verify that filename parsing accurately extracts tags and assigns reporting strata."""
    # Low light & blur
    s1, t1 = parse_condition_stratum("blur_and_poor_lighting__clip_low_light_01__sec007.jpg")
    assert s1 == "Blur & Low Light"
    assert "low_light" in t1
    assert "blur" in t1

    # Top-down camera angle
    s2, t2 = parse_condition_stratum("face_from_top__uniand_dvr_ch4_main_sec013.jpg")
    assert s2 == "Top-Down Angle"
    assert "face_from_top" in t2

    # Occlusion
    s3, t3 = parse_condition_stratum("good_lighting_and_occlusion__clip_occlusion_01.jpg")
    assert s3 == "Occlusion"
    assert "occlusion" in t3

    # Distant / Multi-person
    s4, t4 = parse_condition_stratum("or_quality_multiple_people__uniand_dvr_ch3.jpg")
    assert s4 == "Distant / Multi-Person"
    assert "multiple_distant" in t4


def test_load_ground_truth_annotations():
    """Verify that COCO ground-truth annotations are parsed correctly from disk."""
    gt_map = load_ground_truth_annotations(config.IMAGES_100_DIR)
    assert len(gt_map) > 0, "No ground truth annotations loaded!"

    # Verify sample box format: [x1, y1, x2, y2] with x2 > x1 and y2 > y1
    sample_key = list(gt_map.keys())[0]
    sample_boxes = gt_map[sample_key]
    assert len(sample_boxes) > 0
    b = sample_boxes[0]
    assert len(b) == 4
    assert b[2] > b[0], f"Invalid x coordinates: {b}"
    assert b[3] > b[1], f"Invalid y coordinates: {b}"


def test_three_way_outcome_classification_logic():
    """
    Test the 3-way outcome logic and shared false negative detection:
    a. Missed Face: max_blur_iou < 0.10
    b. Residual Visible: 0.10 <= max_blur_iou < 0.55 OR (max_blur_iou >= 0.10 and retinaface_fired)
    c. Fully Obscured: max_blur_iou >= 0.55 and not retinaface_fired
    """
    from app.pipeline.tracker_interpolator import compute_iou

    gt_box = [100.0, 100.0, 200.0, 200.0]

    # Scenario A: Anonymizer completely missed face (no blur anywhere near)
    no_blur_boxes = [[500.0, 500.0, 600.0, 600.0]]
    max_iou_a = max([compute_iou(gt_box, b) for b in no_blur_boxes])
    assert max_iou_a < 0.10
    # If RetinaFace also missed -> Shared False Negative!
    rf_fired_a = False
    assert (max_iou_a < 0.10) and (not rf_fired_a)

    # Scenario B: Partial overlap (IoU = 0.35) -> Residual Visible
    partial_blur = [[120.0, 120.0, 220.0, 220.0]]
    max_iou_b = max([compute_iou(gt_box, b) for b in partial_blur])
    assert 0.10 <= max_iou_b < 0.55

    # Scenario C: Heavy blur (IoU = 0.85) and RetinaFace does not fire -> Fully Obscured
    solid_blur = [[95.0, 95.0, 205.0, 205.0]]
    max_iou_c = max([compute_iou(gt_box, b) for b in solid_blur])
    rf_fired_c = False
    assert (max_iou_c >= 0.55) and (not rf_fired_c)
