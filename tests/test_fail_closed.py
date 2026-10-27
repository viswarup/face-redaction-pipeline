"""Unit tests verifying the Fail-Closed QA Verifier state machine."""

import glob
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import config
from app.pipeline.verifier import RetinaFaceVerifier, verify_rendered_video


def test_corrupted_or_nonexistent_mp4_fails_closed(tmp_path: Path):
    """
    Assert that passing a non-existent or corrupted file to verify_rendered_video
    always marks the job NEEDS_REVIEW (never SAFE).
    """
    # 1. Non-existent path
    non_existent = tmp_path / "does_not_exist.mp4"
    res1 = verify_rendered_video(non_existent)
    assert res1["status"] == "NEEDS_REVIEW"
    assert res1["reason"] == "VERIFICATION_OR_CODEC_ERROR"
    assert res1["decoded_frames"] == 0

    # 2. Corrupted file (garbage bytes)
    corrupted = tmp_path / "corrupted.mp4"
    corrupted.write_bytes(b"THIS_IS_NOT_A_VALID_MP4_HEADER_GARBAGE_BYTES_123456789")
    res2 = verify_rendered_video(corrupted)
    assert res2["status"] == "NEEDS_REVIEW"
    assert res2["reason"] == "VERIFICATION_OR_CODEC_ERROR"
    assert res2["decoded_frames"] == 0

    # 3. Empty file (0 bytes)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    res3 = verify_rendered_video(empty)
    assert res3["status"] == "NEEDS_REVIEW"
    assert res3["reason"] == "VERIFICATION_OR_CODEC_ERROR"
    assert res3["decoded_frames"] == 0


def test_unblurred_face_triggers_needs_review_with_timestamp(tmp_path: Path):
    """
    Assert that verifying a video containing an unblurred visible face:
    1. Returns status 'NEEDS_REVIEW'.
    2. Records the exact frame index and timestamp in flagged_frames.
    3. Classifies the leak as 'UNBLURRED_MISSED_FACE' when no blur box was applied.
    4. Generates a diagnostic snapshot JPEG on disk.
    """
    # Find a real image with a clear detectable face
    test_images = sorted(glob.glob(str(config.IMAGES_100_DIR / "test" / "*.jpg")))
    assert len(test_images) > 0, "No benchmark images found in data/images_100/test/"

    verifier = RetinaFaceVerifier(conf_thresh=0.25)
    selected_img = None
    for p in test_images:
        candidate = cv2.imread(p)
        faces = verifier.detect(candidate, conf_thresh=0.25)
        if len(faces) > 0:
            selected_img = candidate
            break

    assert selected_img is not None, "Could not find benchmark image with detectable face for leak test"

    # Create an unblurred MP4 video from this face image (6 frames at 10 fps)
    h, w = selected_img.shape[:2]
    unblurred_mp4 = tmp_path / "unblurred_face_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    writer = cv2.VideoWriter(str(unblurred_mp4), fourcc, fps, (w, h))
    for _ in range(6):
        writer.write(selected_img)
    writer.release()

    # Empty tracks json (simulating primary detector completely missed it)
    tracks_json = tmp_path / "empty_tracks.json"
    with open(tracks_json, "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "frames": []}, f)

    res = verify_rendered_video(
        rendered_mp4_path=unblurred_mp4,
        tracks_json_path=tracks_json,
        conf_thresh=0.25,
        stride=2,
        job_id="test_unblurred_leak",
        verifier=verifier,
    )

    # 1. Must fail closed
    assert res["status"] == "NEEDS_REVIEW"
    assert len(res["flagged_frames"]) > 0

    flagged = res["flagged_frames"][0]
    # 2. Exact frame index and timestamp recorded
    assert "frame_idx" in flagged
    assert "timestamp_sec" in flagged
    assert flagged["timestamp_sec"] >= 0.0

    # 3. Classified as UNBLURRED_MISSED_FACE
    assert flagged["diagnostic_classification"] == "UNBLURRED_MISSED_FACE"
    assert flagged["max_applied_blur_iou"] < 0.10

    # 4. Diagnostic JPEG snapshot written
    snapshot_path = Path(flagged["snapshot_path"])
    assert snapshot_path.exists(), f"Snapshot JPEG not found: {snapshot_path}"
    assert snapshot_path.stat().st_size > 0


def test_fully_anonymized_or_blank_video_is_safe(tmp_path: Path):
    """Assert that a video with zero visible faces returns status SAFE."""
    w, h = 320, 240
    safe_mp4 = tmp_path / "safe_blank_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(safe_mp4), fourcc, 10.0, (w, h))

    blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(6):
        writer.write(blank_frame)
    writer.release()

    verifier = RetinaFaceVerifier(conf_thresh=0.30)
    res = verify_rendered_video(
        rendered_mp4_path=safe_mp4,
        tracks_json_path=None,
        conf_thresh=0.30,
        stride=2,
        verifier=verifier,
    )

    assert res["status"] == "SAFE"
    assert res["reason"] is None
    assert len(res["flagged_frames"]) == 0
    assert res["decoded_frames"] == 6
