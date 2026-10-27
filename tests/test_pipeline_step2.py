"""Pytest suite for Step 2 Primary Spatio-Temporal Anonymization Engine."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from app.pipeline.detector import PrimaryFaceDetector
from app.pipeline.tracker_interpolator import (
    compute_iou,
    KalmanBoxTracker,
    TwoPassTrajectorySmoother,
    VideoFaceTracker,
)
from app.pipeline.redactor import apply_obfuscation, dilate_bbox, redact_single_image


def test_dilate_bbox_math():
    """Verify that dilate_bbox expands width and height by 28% and clamps to frame edges."""
    frame_shape = (1000, 1000, 3)
    bbox = [100.0, 100.0, 200.0, 200.0]  # width = 100, height = 100
    # ratio = 0.28 -> dw = 14, dh = 14 -> [86, 86, 214, 214]
    dilated = dilate_bbox(bbox, frame_shape, ratio=0.28)
    assert dilated == [86, 86, 214, 214]

    # Boundary test: box near top-left edge
    bbox_edge = [5.0, 2.0, 50.0, 60.0]
    dilated_edge = dilate_bbox(bbox_edge, frame_shape, ratio=0.28)
    assert dilated_edge[0] == 0
    assert dilated_edge[1] == 0
    assert dilated_edge[2] > 50
    assert dilated_edge[3] > 60


def test_apply_obfuscation():
    """Verify that obfuscation modifies pixel values inside dilated boxes."""
    # Synthetic frame with distinct values
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Draw high contrast patch
    frame[50:100, 50:100] = 255

    dilated_boxes = [[45, 45, 105, 105]]
    redacted = apply_obfuscation(frame, dilated_boxes, mode="gaussian_or_mosaic")

    assert redacted.shape == frame.shape
    # Ensure blurred ROI is not identical to original sharp patch
    assert not np.array_equal(redacted[50:100, 50:100], frame[50:100, 50:100])
    # Ensure regions outside box remain untouched
    assert np.array_equal(redacted[:30, :30], frame[:30, :30])


def test_iou_and_kalman_box_tracker():
    """Test IoU calculation and Kalman filter state estimation."""
    box_a = [10.0, 10.0, 50.0, 50.0]
    box_b = [10.0, 10.0, 50.0, 50.0]
    assert compute_iou(box_a, box_b) == 1.0

    box_c = [100.0, 100.0, 150.0, 150.0]
    assert compute_iou(box_a, box_c) == 0.0

    # Test tracker update and prediction
    tracker = KalmanBoxTracker(box_a, score=0.9)
    pred_box = tracker.predict()
    assert len(pred_box) == 4
    # State update
    tracker.update([12.0, 12.0, 52.0, 52.0], score=0.92)
    state = tracker.get_state()
    assert state[0] > 10.0 and state[1] > 10.0


def test_two_pass_trajectory_gap_interpolation():
    """
    Test bidirectional trajectory gap interpolation for t2 - t1 <= 12 frames.
    """
    smoother = TwoPassTrajectorySmoother(max_gap=12, coast_frames=3, interior_margin=25)
    total_frames = 20
    raw_detections = [[] for _ in range(total_frames)]

    # Track 1 detected at frame 2 and frame 6 (gap = 4 frames, <= 12)
    raw_detections[2] = [{"track_id": 1, "bbox": [100.0, 100.0, 150.0, 150.0], "score": 0.8, "source": "detection"}]
    raw_detections[6] = [{"track_id": 1, "bbox": [140.0, 140.0, 190.0, 190.0], "score": 0.8, "source": "detection"}]

    smoothed = smoother.smooth(raw_detections, frame_shape=(500, 500))

    # Assert intermediate frames (3, 4, 5) were filled by interpolation
    for f in (3, 4, 5):
        boxes_at_f = smoothed[f]
        interp_boxes = [b for b in boxes_at_f if b.get("source") == "interpolated_coaster"]
        assert len(interp_boxes) == 1, f"Frame {f} did not receive interpolated box"
        box = interp_boxes[0]
        assert box["track_id"] == 1
        # Check linear progression of coordinates
        assert 100.0 < box["bbox"][0] < 140.0


def test_interior_birth_death_coasting():
    """
    Test 3-frame coasting at track birth and death when faces are in the interior of the frame.
    """
    smoother = TwoPassTrajectorySmoother(max_gap=12, coast_frames=3, interior_margin=25)
    total_frames = 15
    raw_detections = [[] for _ in range(total_frames)]

    # Face strictly in the interior (centered at 200, 200 inside 600x600 frame)
    # Appears at frame 5 and ends at frame 7
    for f in (5, 6, 7):
        raw_detections[f] = [{"track_id": 2, "bbox": [200.0, 200.0, 250.0, 250.0], "score": 0.9, "source": "detection"}]

    smoothed = smoother.smooth(raw_detections, frame_shape=(600, 600))

    # Coast backward should populate frames 2, 3, 4
    for f in (2, 3, 4):
        coast_b = [b for b in smoothed[f] if b.get("source") == "coast_backward"]
        assert len(coast_b) == 1, f"Frame {f} missing backward coasting box"

    # Coast forward should populate frames 8, 9, 10
    for f in (8, 9, 10):
        coast_f = [b for b in smoothed[f] if b.get("source") == "coast_forward"]
        assert len(coast_f) == 1, f"Frame {f} missing forward coasting box"


def test_primary_detector_inference():
    """Test PrimaryFaceDetector inference returns compliant schema."""
    detector = PrimaryFaceDetector(conf_thresh=0.25)
    assert detector.loaded_weight_path is not None

    # Synthetic image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_img, (320, 240), 50, (200, 200, 200), -1)

    dets = detector.detect(test_img)
    assert isinstance(dets, list)
    for d in dets:
        assert "bbox" in d
        assert len(d["bbox"]) == 4
        assert "score" in d
        assert "landmarks" in d
