"""Smoke test script for Primary Spatio-Temporal Anonymization Engine."""

import glob
import json
import os
import sys
from pathlib import Path

import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.pipeline.detector import PrimaryFaceDetector
from app.pipeline.redactor import process_video_redaction, redact_single_image


def run_smoke_test():
    print("=" * 70)
    print("RUNNING SMOKE TEST: Primary Spatio-Temporal Anonymization Engine")
    print("=" * 70)

    # 1. Test detector initialization
    print("\n[Step 1] Initializing PrimaryFaceDetector...")
    detector = PrimaryFaceDetector(conf_thresh=0.25)
    print(f"Loaded detector: model={detector.model_name}, weight={detector.loaded_weight_path}")
    assert detector.loaded_weight_path is not None, "Failed to load detector weight path!"

    # 2. Test single image redaction on hand-annotated benchmark sample
    print("\n[Step 2] Testing static image redaction on benchmark dataset...")
    test_images = sorted(glob.glob(str(config.IMAGES_100_DIR / "test" / "*.jpg")))
    if test_images:
        sample_img_path = test_images[0]
        img = cv2.imread(sample_img_path)
        assert img is not None, f"Failed to load sample image: {sample_img_path}"
        redacted_img, detections = redact_single_image(img, detector=detector)

        out_eval_dir = config.EVAL_100_REDACTED_DIR
        out_eval_dir.mkdir(parents=True, exist_ok=True)
        eval_out_path = out_eval_dir / f"smoke_redacted_{Path(sample_img_path).name}"
        cv2.imwrite(str(eval_out_path), redacted_img)

        print(f"  Sample image: {Path(sample_img_path).name}")
        print(f"  Detected faces: {len(detections)}")
        print(f"  Redacted image saved to: {eval_out_path}")
        assert eval_out_path.exists(), "Redacted sample image was not written!"
    else:
        print("  Warning: No test images found in data/images_100/test/")

    # 3. Test video redaction on video snippet
    print("\n[Step 3] Testing video stream redaction + tracker interpolation + encoding...")
    candidates = [
        "data/video_snippets/face from top/Uni+ DVR_ch4_main_20250730080000_20250730090000-00.00.05.355-00.00.08.422-seg1.mp4",
        "data/video_snippets/good lighting  + occlusion/clip_nominal_01.mp4",
    ]
    sample_video_path = None
    for c in candidates:
        if Path(c).exists():
            sample_video_path = c
            break

    if not sample_video_path:
        all_videos = sorted(glob.glob("data/video_snippets/**/*.mp4", recursive=True))
        if all_videos:
            sample_video_path = all_videos[0]

    assert sample_video_path is not None, "No test video snippets found in data/video_snippets!"
    print(f"  Input test video: {sample_video_path}")

    out_video_dir = config.REDACTED_VIDEOS_DIR
    out_video_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_video_dir / "smoke_redacted_test.mp4"
    tracks_json_path = Path(f"{out_video_path}.tracks.json")

    # Clean previous run artifacts if present
    if out_video_path.exists():
        out_video_path.unlink()
    if tracks_json_path.exists():
        tracks_json_path.unlink()

    telemetry = process_video_redaction(
        input_video_path=sample_video_path,
        output_video_path=out_video_path,
        detector=detector,
        conf_thresh=0.25,
        dilation_ratio=0.28,
        interpolation_window=12,
    )

    print("\n[Step 4] Verifying outputs...")
    print("  Telemetry summary:")
    for k, v in telemetry.items():
        print(f"    - {k}: {v}")

    # Assert video exists and is non-empty
    assert out_video_path.exists(), f"Output video file missing: {out_video_path}"
    assert out_video_path.stat().st_size > 1000, f"Output video file is suspiciously small: {out_video_path.stat().st_size} bytes"

    # Assert companion tracks.json exists and is valid
    assert tracks_json_path.exists(), f"Companion tracks JSON missing: {tracks_json_path}"
    with open(tracks_json_path, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)

    assert "metadata" in tracks_data, "tracks.json missing 'metadata' block"
    assert "frames" in tracks_data, "tracks.json missing 'frames' block"
    assert len(tracks_data["frames"]) == telemetry["total_frames"], "Frame count mismatch in tracks.json"

    # Verify video readability with OpenCV
    cap = cv2.VideoCapture(str(out_video_path))
    assert cap.isOpened(), "Encoded video cannot be opened by OpenCV!"
    read_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        read_frames += 1
    cap.release()
    print(f"  Verified encoded video: {read_frames} frames successfully decoded.")
    assert read_frames > 0, "No frames could be decoded from encoded video!"

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED CLEANLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
