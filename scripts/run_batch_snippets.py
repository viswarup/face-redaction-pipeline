"""CLI script to batch-process all body-camera video snippets through the FastAPI pipeline.
Verifies queueing, idempotency deduplication, background redaction, QA verification,
and outputs a comprehensive batch status table and summary JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.db import get_job, list_jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_snippets")


def to_relative_path(path_val: str | Path) -> str:
    """Convert absolute path to project-relative path if within project root."""
    try:
        p = Path(path_val)
        base = config.BASE_DIR.resolve()
        if p.is_absolute():
            resolved = p.resolve()
            if resolved.is_relative_to(base):
                return str(resolved.relative_to(base))
        return str(path_val)
    except Exception:
        return str(path_val)


def run_batch_snippets(
    snippets_dir: Optional[Path] = None,
    poll_interval: float = 2.0,
    max_wait_seconds: float = 400.0,
    limit: Optional[int] = 5,
) -> dict[str, Any]:
    """
    Submits video snippets to the FastAPI pipeline and polls for completion.
    Tests idempotency on the first clip.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # Locate demo clips
    demo_dir = config.VIDEO_SNIPPETS_DIR / "demo_clips"
    if demo_dir.exists() and any(demo_dir.glob("*.mp4")):
        base_snippets = demo_dir
    else:
        base_snippets = snippets_dir or config.VIDEO_SNIPPETS_DIR

    all_videos = sorted(list(base_snippets.glob("*.mp4")))
    # If there are subdirectories, also search them if demo_dir is empty
    if not all_videos:
        all_videos = sorted(list(base_snippets.rglob("*.mp4")))

    if limit and limit > 0:
        all_videos = all_videos[:limit]

    logger.info("Found %d video snippets in %s to process", len(all_videos), base_snippets)
    assert len(all_videos) > 0, f"No MP4 video snippets found in {base_snippets}"

    submitted_jobs: list[dict[str, Any]] = []
    idempotency_test_results: dict[str, Any] = {}

    # 1. Submit snippets to FastAPI POST /jobs
    print("\n" + "=" * 80)
    print("SUBMITTING VIDEO SNIPPETS TO PIPELINE")
    print("=" * 80)

    for idx, vid_path in enumerate(all_videos):
        folder_tag = vid_path.parent.name
        rel_vid_path = to_relative_path(vid_path)
        event_id = f"Video {idx+1}"

        resp = client.post(
            "/jobs",
            data={"event_id": event_id, "file_path": rel_vid_path},
        )
        assert resp.status_code == 201, f"Failed to submit {vid_path.name}: {resp.text}"
        data = resp.json()
        job_id = data["job_id"]

        submitted_jobs.append(
            {
                "job_id": job_id,
                "event_id": event_id,
                "video_name": vid_path.name,
                "folder": folder_tag,
                "path": rel_vid_path,
            }
        )
        logger.info(
            "[%02d/%02d] Submitted %s -> job_id=%s (status=%s)",
            idx + 1,
            len(all_videos),
            vid_path.name,
            job_id,
            data["status"],
        )

        # 2. Test Idempotency: Submit the first snippet a second time with exact same event_id
        if idx == 0:
            logger.info("--> Testing Idempotency: Re-submitting identical event %s...", event_id)
            retry_resp = client.post(
                "/jobs",
                data={"event_id": event_id, "file_path": rel_vid_path},
            )
            assert retry_resp.status_code == 201, f"Retry submission failed: {retry_resp.text}"
            retry_data = retry_resp.json()

            assert retry_data["job_id"] == job_id, "Idempotency failed: generated different job_id!"
            assert retry_data["is_duplicate_event"] is True, "Idempotency failed: is_duplicate_event is not True!"
            assert retry_data["duplicate_request_count"] == 1, "Idempotency failed: duplicate_request_count != 1"

            idempotency_test_results = {
                "tested_job_id": job_id,
                "event_id": event_id,
                "is_duplicate_event": retry_data["is_duplicate_event"],
                "duplicate_request_count": retry_data["duplicate_request_count"],
                "idempotency_verified": True,
            }
            logger.info("--> Idempotency cleanly verified! Duplicate request counter = 1.")

    # 3. Poll for background worker completion
    print("\n" + "=" * 80)
    print("WAITING FOR ASYNCHRONOUS PIPELINE JOBS TO COMPLETE...")
    print("=" * 80)

    start_wait = time.perf_counter()
    all_job_ids = [j["job_id"] for j in submitted_jobs]

    while True:
        completed_count = 0
        current_states = {}
        for jid in all_job_ids:
            row = get_job(jid)
            st = row.get("status") if row else "UNKNOWN"
            current_states[jid] = st
            if st in ("SAFE", "NEEDS_REVIEW"):
                completed_count += 1

        elapsed = time.perf_counter() - start_wait
        logger.info(
            "Progress: %d/%d jobs finished in %.1fs...",
            completed_count,
            len(all_job_ids),
            elapsed,
        )

        if completed_count == len(all_job_ids):
            logger.info("All %d batch snippet jobs completed processing!", completed_count)
            break

        if elapsed > max_wait_seconds:
            logger.warning("Reached maximum wait time (%.1fs). Some jobs still pending.", max_wait_seconds)
            break

        time.sleep(poll_interval)

    # 4. Compile Batch Execution Report
    table_rows: list[list[Any]] = []
    summary_results: list[dict[str, Any]] = []

    safe_count = 0
    needs_review_count = 0

    for item in submitted_jobs:
        jid = item["job_id"]
        row = get_job(jid) or {}
        st = row.get("status", "PENDING")
        if st == "SAFE":
            safe_count += 1
        elif st == "NEEDS_REVIEW":
            needs_review_count += 1

        fps = float(row.get("processing_fps") or 0.0)

        # Parse telemetry
        telemetry_raw = row.get("telemetry_json", "{}")
        try:
            telemetry = json.loads(telemetry_raw) if isinstance(telemetry_raw, str) else telemetry_raw
        except Exception:
            telemetry = {}

        redaction_meta = telemetry.get("redaction", {})
        frames = redaction_meta.get("total_frames", 0)
        time_sec = redaction_meta.get("elapsed_seconds", 0.0)
        resolution = redaction_meta.get("resolution", [0, 0])
        res_str = f"{resolution[0]}x{resolution[1]}" if resolution != [0, 0] else "N/A"
        coasted = redaction_meta.get("total_coasted_boxes", 0)
        interpolated = redaction_meta.get("total_interpolated_boxes", 0)
        video_fps = redaction_meta.get("fps", 0.0)
        duration_sec = round(frames / video_fps, 1) if video_fps > 0 else 0.0
        blind_spot = telemetry.get("blind_spot_warning", False)

        flagged_raw = row.get("flagged_frames_json", "[]")
        try:
            flagged = json.loads(flagged_raw) if isinstance(flagged_raw, str) else flagged_raw
        except Exception:
            flagged = []

        table_rows.append(
            [
                jid,
                item["video_name"][:32],
                res_str,
                f"{duration_sec}s ({frames}f)",
                f"{fps:.1f}",
                st,
                coasted + interpolated,
                len(flagged),
                "YES" if blind_spot else "No",
            ]
        )

        summary_results.append(
            {
                "job_id": jid,
                "video_name": item["video_name"],
                "folder": item["folder"],
                "input_path": to_relative_path(item["path"]),
                "output_path": to_relative_path(row.get("output_path", "")),
                "status": st,
                "resolution": resolution,
                "duration_seconds": duration_sec,
                "total_frames": frames,
                "video_fps": video_fps,
                "processing_fps": fps,
                "elapsed_seconds": time_sec,
                "total_coasted_boxes": coasted,
                "total_interpolated_boxes": interpolated,
                "flagged_leak_count": len(flagged),
                "blind_spot_warning": blind_spot,
                "failure_reason": row.get("failure_reason"),
                "duplicate_request_count": row.get("duplicate_request_count", 0),
            }
        )

    headers = [
        "Job ID",
        "Video Clip",
        "Resolution",
        "Duration",
        "FPS",
        "Status",
        "Coasted/Interp",
        "Leaks Flagged",
        "Blind Spot",
    ]

    print("\n" + "=" * 95)
    print("FINAL 5 DEMO CLIPS PIPELINE EXECUTION SUMMARY")
    print("=" * 95)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))
    print("=" * 95)
    print(f"Total Snippets: {len(submitted_jobs)} | SAFE: {safe_count} | NEEDS_REVIEW: {needs_review_count}")
    print("=" * 95 + "\n")

    # Save summary to both clips_5_summary.json and batch_snippets_summary.json
    out_dir = config.DATA_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "idempotency_verification": idempotency_test_results,
        "total_snippets": len(submitted_jobs),
        "safe_count": safe_count,
        "needs_review_count": needs_review_count,
        "jobs": summary_results,
    }

    clips_5_json = out_dir / "clips_5_summary.json"
    with open(clips_5_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    batch_json = out_dir / "batch_snippets_summary.json"
    with open(batch_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info("Saved clips summary to %s and %s", clips_5_json, batch_json)
    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process video snippets through the anonymizer pipeline.")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of snippets to process (default: 5)")
    parser.add_argument("--wait", type=float, default=600.0, help="Maximum seconds to wait for batch completion")
    args = parser.parse_args()

    run_batch_snippets(limit=args.limit, max_wait_seconds=args.wait)
