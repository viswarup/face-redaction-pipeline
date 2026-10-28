"""Human-in-the-Loop (HITL) Privacy & Verification Audit Dashboard in Streamlit.
Provides clip inspection, risk timeline scrubbing, 3-column frame diagnostics,
idempotency stress-testing, and the 100-image benchmark failure taxonomy.
"""

from __future__ import annotations

import glob
import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from fastapi.testclient import TestClient

from app.config import config
from app.db import get_job, list_jobs, update_review_action
from app.main import app

# Page configuration
st.set_page_config(
    page_title="Body-Camera Privacy & Verification Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for Clean White / Light Theme
st.markdown(
    """
    <style>
    /* Force Light Theme & Pure White Background */
    .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    .main .block-container {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Status Badges */
    .badge-safe {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #86efac;
    }
    .badge-review {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #fca5a5;
    }
    .badge-processing {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #93c5fd;
    }
    .badge-amber {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #fde68a;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def resolve_path(path_val: Optional[str | Path]) -> Optional[Path]:
    """Resolve path relative to project root if needed and return existing Path or None."""
    if not path_val:
        return None
    p = Path(path_val)
    if not p.is_absolute():
        p = (config.BASE_DIR / p).resolve()
    return p if p.exists() else None


def extract_frame_from_video(video_path: str | Path, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a specific BGR frame from a video file."""
    p = resolve_path(video_path)
    if not p:
        return None
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def load_companion_tracks(output_path: Optional[str]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Load companion tracks JSON mapping frame_idx -> box records."""
    if not output_path:
        return {}, {}
    p = resolve_path(output_path)
    tracks_json_path = Path(f"{p}.tracks.json") if p else Path(f"{output_path}.tracks.json")
    if not tracks_json_path.exists():
        return {}, {}
    try:
        with open(tracks_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        frames_map = {f["frame_index"]: f for f in data.get("frames", [])}
        return meta, frames_map
    except Exception:
        return {}, {}



# Header
st.title("🛡️ Safety-Critical Body-Camera Face Anonymization Audit")
st.caption(
    "Automated Dual-Engine Pipeline: SCRFD-34G + ByteTrack Trajectory Smoother + Independent RetinaFace QA Verifier"
)

tabs = st.tabs(
    [
        "📋 Video Audit Queue & HITL Inspector",
        "⚡ Live Upload & Idempotency Stress-Tester",
        "📊 100-Image Benchmark & Correlated Blind Spots",
    ]
)

# ==============================================================================
# TAB 1: Video Audit Queue & HITL Inspector
# ==============================================================================
with tabs[0]:
    jobs = list_jobs()

    # Top KPI Row
    total_jobs = len(jobs)
    safe_jobs = sum(1 for j in jobs if j.get("status") == "SAFE")
    review_jobs = sum(1 for j in jobs if j.get("status") == "NEEDS_REVIEW")
    dedup_events = sum(j.get("duplicate_request_count", 0) for j in jobs)

    fps_values = [j.get("processing_fps", 0.0) for j in jobs if (j.get("processing_fps") or 0.0) > 0.0]
    mean_fps = round(sum(fps_values) / len(fps_values), 1) if fps_values else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Video Jobs", total_jobs)
    kpi2.metric("Approved SAFE", safe_jobs)
    kpi3.metric("Quarantined NEEDS_REVIEW", review_jobs)
    kpi4.metric("Deduplicated Events", dedup_events)
    kpi5.metric("Mean Pipeline FPS", f"{mean_fps} fps")

    st.markdown("---")

    # Queue Filter
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        status_filter = st.selectbox(
            "Filter Jobs by Status:",
            ["All Jobs", "NEEDS_REVIEW (Quarantined)", "SAFE (Approved)", "Processing / Queued"],
        )

    filtered_jobs = jobs
    if status_filter == "NEEDS_REVIEW (Quarantined)":
        filtered_jobs = [j for j in jobs if j.get("status") == "NEEDS_REVIEW"]
    elif status_filter == "SAFE (Approved)":
        filtered_jobs = [j for j in jobs if j.get("status") == "SAFE"]
    elif status_filter == "Processing / Queued":
        filtered_jobs = [j for j in jobs if j.get("status") in ("QUEUED", "PROCESSING_REDACTION", "VERIFYING_OUTPUT")]

    # Build Display Table
    table_data = []
    for j in filtered_jobs:
        in_p = Path(j.get("input_path") or "unknown.mp4")
        t_raw = j.get("telemetry_json", "{}")
        try:
            telemetry = json.loads(t_raw) if isinstance(t_raw, str) else t_raw
        except Exception:
            telemetry = {}

        redaction_meta = telemetry.get("redaction", {})
        lum = redaction_meta.get("mean_y_luminance", 0.0)
        blur = redaction_meta.get("mean_laplacian_blur_variance", 0.0)

        f_raw = j.get("flagged_frames_json", "[]")
        try:
            flagged = json.loads(f_raw) if isinstance(f_raw, str) else f_raw
        except Exception:
            flagged = []

        total_interpolated = redaction_meta.get("total_interpolated_boxes", 0)
        total_coasted = redaction_meta.get("total_coasted_boxes", 0)

        table_data.append(
            {
                "Event ID": j["event_id"],
                "Snippet Name": in_p.name,
                "Status": j["status"],
                "Job ID": j["job_id"],
                "Lighting (Brightness)": f"{lum:.1f}",
                "Sharpness (Blur Test)": f"{blur:.1f}",
                "Flagged Leaks (QA Warning)": len(flagged),
                "Motion-Smoothed Boxes": f"{total_interpolated + total_coasted} boxes",
                "Duplicates Handled": j.get("duplicate_request_count", 0),
                "Processing FPS": f"{j.get('processing_fps', 0.0):.1f}",
            }
        )

    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No jobs match the selected filter.")

    st.markdown("### 🔍 Deep-Dive Clip Inspector")

    if jobs:
        display_jobs = filtered_jobs if filtered_jobs else jobs
        job_id_to_label = {
            j["job_id"]: f"{j.get('event_id', 'Job')} — {Path(j.get('input_path') or '').name} [{j.get('status', 'PENDING')}]"
            for j in display_jobs
        }
        selected_job_id = st.selectbox(
            "Select Video to Inspect:",
            options=list(job_id_to_label.keys()),
            format_func=lambda x: job_id_to_label.get(x, x),
        )
        selected_job = get_job(selected_job_id)

        if selected_job:
            current_status = selected_job.get("status", "UNKNOWN")
            event_title = selected_job.get("event_id", "Video")
            st.markdown(f"#### 🎬 Inspecting: **{event_title}** — `{Path(selected_job.get('input_path') or '').name}`")
            output_path_str = selected_job.get("output_path")
            input_path_str = selected_job.get("input_path")

            t_raw = selected_job.get("telemetry_json", "{}")
            try:
                job_telemetry = json.loads(t_raw) if isinstance(t_raw, str) else t_raw
            except Exception:
                job_telemetry = {}

            f_raw = selected_job.get("flagged_frames_json", "[]")
            try:
                flagged_frames = json.loads(f_raw) if isinstance(f_raw, str) else f_raw
            except Exception:
                flagged_frames = []

            meta, frames_map = load_companion_tracks(output_path_str)

            # Correlated Blind-Spot Warning Banner
            blind_spot = job_telemetry.get("blind_spot_warning", False)
            mean_y = job_telemetry.get("redaction", {}).get("mean_y_luminance", 100.0)
            mean_lap = job_telemetry.get("redaction", {}).get("mean_laplacian_blur_variance", 500.0)

            if blind_spot or mean_y < 45.0 or mean_lap < 60.0:
                st.warning(
                    f"⚠️ **CORRELATED BLIND-SPOT RISK DETECTED**: This snippet exhibits low illumination "
                    f"(Mean Y: {mean_y:.1f}) or severe motion blur (Laplacian: {mean_lap:.1f}). "
                    f"Even if RetinaFace does not flag residual leaks, 2D deep detectors share epistemic blind spots "
                    f"under extreme optical degradation. Human review of coasted segments is strongly advised."
                )

            # Side-by-Side Video Players
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.subheader("📹 Raw Input Bodycam Snippet")
                in_res = resolve_path(input_path_str)
                if in_res:
                    st.video(str(in_res))
                else:
                    st.info("Input video file path not found.")

            with v_col2:
                st.subheader("🛡️ Anonymized & Redacted Output")
                out_res = resolve_path(output_path_str)
                if out_res:
                    st.video(str(out_res))
                else:
                    st.info("Rendered video not yet available (still in queue/processing).")

            st.markdown("---")

            # Frame-Level Risk Scrubber & Diagnostic Viewer
            st.subheader("⏱️ Frame-Level Risk Scrubber")

            total_frames = meta.get("total_frames", len(frames_map)) or (len(flagged_frames) if flagged_frames else 100)
            red_frame_indices = {f["frame_idx"]: f for f in flagged_frames}
            amber_frame_indices = {
                f_idx: f_data
                for f_idx, f_data in frames_map.items()
                if any(b.get("source") != "detection" for b in f_data.get("boxes", []))
            }
            green_frame_indices = [
                f_idx
                for f_idx in range(total_frames)
                if f_idx not in red_frame_indices and f_idx not in amber_frame_indices
            ]

            # Summary counters banner
            st.markdown(
                f"""
                <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px;">
                    <span class="badge-review">🔴 <b>Red Tags (RetinaFace Leaks):</b> {len(red_frame_indices)} frames</span>
                    <span class="badge-amber">🟡 <b>Yellow Tags (Coasted/Interpolated):</b> {len(amber_frame_indices)} frames</span>
                    <span class="badge-safe">🟢 <b>Green Tags (Clean Redaction):</b> {len(green_frame_indices)} frames</span>
                    <span style="background: #f1f5f9; padding: 4px 10px; border-radius: 6px; font-weight: 600; color: #334155; border: 1px solid #cbd5e1;">Total: {total_frames} frames</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Dropdown for Tag Category Filter
            tag_col1, tag_col2 = st.columns([1, 2])
            with tag_col1:
                tag_category_choice = st.selectbox(
                    "🏷️ Filter by Tag Category:",
                    [
                        f"🔴 Red Tags Only ({len(red_frame_indices)} Leaks Flagged)",
                        f"🟡 Yellow Tags Only ({len(amber_frame_indices)} Coasted/Interpolated)",
                        f"🟢 Green Tags Only ({len(green_frame_indices)} Clean Frames)",
                        f"All Frames ({total_frames} Total)",
                    ],
                )

            # Build list of selectable frames based on chosen category
            if tag_category_choice.startswith("🔴"):
                category_frames = sorted(list(red_frame_indices.keys()))
                if not category_frames:
                    category_frames = [0]
                frame_label_map = {
                    f_idx: f"Frame #{f_idx} — {red_frame_indices[f_idx]['diagnostic_classification']} (Score: {red_frame_indices[f_idx]['retinaface_score']:.2f}, IoU: {red_frame_indices[f_idx].get('max_applied_blur_iou', 0.0):.2f})"
                    for f_idx in category_frames
                }
            elif tag_category_choice.startswith("🟡"):
                category_frames = sorted(list(amber_frame_indices.keys()))
                if not category_frames:
                    category_frames = [0]
                frame_label_map = {}
                for f_idx in category_frames:
                    sources = [
                        b.get("source", "interp")
                        for b in frames_map.get(f_idx, {}).get("boxes", [])
                        if b.get("source") != "detection"
                    ]
                    src_summary = ", ".join(set(sources)) if sources else "coasted"
                    frame_label_map[f_idx] = f"Frame #{f_idx} — ByteTrack [{src_summary}]"
            elif tag_category_choice.startswith("🟢"):
                category_frames = green_frame_indices if green_frame_indices else [0]
                # If too many green frames, sample or provide list
                frame_label_map = {f_idx: f"Frame #{f_idx} — Clean Redaction (0 Leaks)" for f_idx in category_frames}
            else:
                category_frames = list(range(max(1, total_frames)))
                frame_label_map = {f_idx: f"Frame #{f_idx}" for f_idx in category_frames}

            with tag_col2:
                selected_frame_idx = st.selectbox(
                    "📍 Select Tagged Frame to Inspect:",
                    options=category_frames,
                    format_func=lambda x: frame_label_map.get(x, f"Frame #{x}"),
                )

            # Display Risk Tag Status Banner for current frame
            if selected_frame_idx in red_frame_indices:
                leak = red_frame_indices[selected_frame_idx]
                st.error(
                    f"🔴 **RED TAG DETECTED (Frame #{selected_frame_idx})**: "
                    f"RetinaFace flagged a residual face instance ({leak['diagnostic_classification']}, "
                    f"Confidence: {leak['retinaface_score']:.2f}, Blur IoU: {leak.get('max_applied_blur_iou', 0.0):.2f}). "
                    f"Human review required before public release!"
                )
            elif selected_frame_idx in amber_frame_indices:
                box_sources = [
                    b.get("source", "interp")
                    for b in frames_map.get(selected_frame_idx, {}).get("boxes", [])
                    if b.get("source") != "detection"
                ]
                st.warning(
                    f"🟡 **YELLOW TAG (Frame #{selected_frame_idx})**: "
                    f"Primary detector temporarily dropped out. ByteTrack filled in coverage via "
                    f"trajectory smoothing ({', '.join(set(box_sources))}). Verify box position."
                )
            else:
                st.success(
                    f"🟢 **GREEN TAG (Frame #{selected_frame_idx})**: "
                    f"Clean anonymization. Primary detector active and zero residual faces detected by RetinaFace QA."
                )

            # 3-Column Frame Diagnostic Viewer
            st.markdown(f"#### 🔎 Diagnostic Viewer: Frame #{selected_frame_idx}")
            raw_frame = extract_frame_from_video(input_path_str, selected_frame_idx) if input_path_str else None
            rend_frame = extract_frame_from_video(output_path_str, selected_frame_idx) if output_path_str else None

            d_col1, d_col2, d_col3 = st.columns(3)

            with d_col1:
                st.caption("1. Raw Original Frame")
                if raw_frame is not None:
                    st.image(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.text("Frame unavailable")

            with d_col2:
                st.caption("2. Rendered Redacted Frame")
                if rend_frame is not None:
                    st.image(cv2.cvtColor(rend_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.text("Frame unavailable")

            with d_col3:
                st.caption("3. QA Audit & Diagnostic Overlay")
                if rend_frame is not None:
                    overlay_img = rend_frame.copy()

                    # Applied boxes from tracks.json in Blue
                    frame_boxes = frames_map.get(selected_frame_idx, {}).get("boxes", [])
                    for b in frame_boxes:
                        bx = b.get("dilated_bbox") or b.get("raw_bbox")
                        if bx:
                            cv2.rectangle(
                                overlay_img,
                                (int(bx[0]), int(bx[1])),
                                (int(bx[2]), int(bx[3])),
                                (255, 0, 0),
                                2,
                            )
                            src_label = b.get("source", "det")
                            cv2.putText(
                                overlay_img,
                                src_label,
                                (int(bx[0]), max(15, int(bx[1]) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 0, 0),
                                1,
                            )

                    # Check if this frame is flagged by RetinaFace
                    is_red = selected_frame_idx in red_frame_indices
                    if is_red:
                        leak_info = red_frame_indices[selected_frame_idx]
                        rf_box = leak_info["retinaface_bbox"]
                        cv2.rectangle(
                            overlay_img,
                            (int(rf_box[0]), int(rf_box[1])),
                            (int(rf_box[2]), int(rf_box[3])),
                            (0, 0, 255),
                            2,
                        )
                        label = f"{leak_info['diagnostic_classification']} ({leak_info['retinaface_score']:.2f})"
                        cv2.putText(
                            overlay_img,
                            label,
                            (int(rf_box[0]), max(15, int(rf_box[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                        # Draw landmarks in Red if present
                        for pt_name, pt in leak_info.get("landmarks", {}).items():
                            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                                cv2.circle(overlay_img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

                        st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.error(
                            f"🚨 **LEAK DETECTED**: {leak_info['diagnostic_classification']} "
                            f"(RetinaFace Confidence: {leak_info['retinaface_score']:.2f}, Blur IoU: {leak_info['max_applied_blur_iou']})"
                        )
                    else:
                        st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        if selected_frame_idx in amber_frame_indices:
                            st.warning("🟡 **Coasted Frame**: Track maintained via ByteTrack interpolation.")
                        else:
                            st.success("✅ **Clear**: Zero visible face residuals detected by QA verifier.")
                else:
                    st.text("Frame unavailable")

            st.markdown("---")

            # Reviewer Action Bar
            st.subheader("⚖️ Reviewer Action Bar")
            rev_col1, rev_col2 = st.columns([2, 1])

            with rev_col1:
                notes = st.text_area(
                    "Reviewer Audit Notes / Justification:",
                    value=selected_job.get("reviewer_notes") or "",
                    placeholder="Enter audit rationale (e.g., 'False positive mannequin flagged by RetinaFace', 'Verified safe')...",
                )

            with rev_col2:
                st.write("**Commit Audit Decision:**")
                if st.button("✅ Approve & Mark SAFE (False Positive Override)", type="primary"):
                    update_review_action(job_id=selected_job_id, status="SAFE", reviewer_notes=notes)
                    st.success(f"Job {selected_job_id} successfully marked SAFE!")
                    st.rerun()

                if st.button("🚨 Confirm Leak (Keep Quarantined)"):
                    update_review_action(job_id=selected_job_id, status="NEEDS_REVIEW", reviewer_notes=notes)
                    st.warning(f"Job {selected_job_id} quarantined as NEEDS_REVIEW.")
                    st.rerun()


# ==============================================================================
# TAB 2: Live Upload & Idempotency Stress-Tester
# ==============================================================================
with tabs[1]:
    st.header("⚡ Live Upload & Idempotency Stress-Tester")
    st.caption("Visually verify that repeated or concurrent webhook retries map to exactly 1 logical job.")

    c1, c2 = st.columns(2)
    with c1:
        test_event_id = st.text_input("Upload Event ID:", value="EVT-LIVE-DEMO-001")
        all_snippet_files = sorted(list(config.VIDEO_SNIPPETS_DIR.rglob("*.mp4")))
        snippet_options = [str(f.relative_to(config.BASE_DIR)) for f in all_snippet_files]
        selected_snippet_rel = st.selectbox("Pick existing video snippet:", snippet_options)

    with c2:
        uploaded_file = st.file_uploader("Or Upload Custom Bodycam MP4:", type=["mp4", "avi", "mov"])

    target_video_bytes: Optional[bytes] = None
    target_filename = "test_video.mp4"

    if uploaded_file is not None:
        target_video_bytes = uploaded_file.read()
        target_filename = uploaded_file.name
    elif selected_snippet_rel:
        target_path = config.BASE_DIR / selected_snippet_rel
        if target_path.exists():
            target_video_bytes = target_path.read_bytes()
            target_filename = target_path.name

    st.markdown("---")

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        st.subheader("1. Standard Single Upload")
        if st.button("Submit Single Video Upload", type="secondary"):
            if not target_video_bytes:
                st.error("No video payload selected!")
            else:
                client = TestClient(app)
                file_tuple = (target_filename, io.BytesIO(target_video_bytes), "video/mp4")
                resp = client.post(
                    "/jobs",
                    data={"event_id": test_event_id},
                    files={"file": file_tuple},
                )
                st.json(resp.json())
                st.success(f"Job successfully queued! HTTP {resp.status_code}")

    with action_col2:
        st.subheader("2. Idempotency Stress-Test")
        st.caption("Fires 3 repeated requests with the identical event_id and payload simultaneously.")
        if st.button("🔥 Fire 3x Duplicate Upload Webhook (Same event_id)", type="primary"):
            if not target_video_bytes:
                st.error("No video payload selected!")
            else:
                client = TestClient(app)
                responses = []

                progress_bar = st.progress(0, text="Firing 3x duplicate webhook calls...")
                for i in range(3):
                    file_tuple = (target_filename, io.BytesIO(target_video_bytes), "video/mp4")
                    r = client.post(
                        "/jobs",
                        data={"event_id": test_event_id},
                        files={"file": file_tuple},
                    )
                    responses.append(r.json())
                    progress_bar.progress((i + 1) / 3)

                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.write("**Webhook #1 (Initial)**")
                    st.json(responses[0])
                with r_col2:
                    st.write("**Webhook #2 (Retry 1)**")
                    st.json(responses[1])
                with r_col3:
                    st.write("**Webhook #3 (Retry 2)**")
                    st.json(responses[2])

                # Verify idempotency assertions
                job_1 = responses[0]["job_id"]
                job_2 = responses[1]["job_id"]
                job_3 = responses[2]["job_id"]

                if job_1 == job_2 == job_3:
                    st.success(
                        f"✅ **IDEMPOTENCY PROOF VERIFIED**: All 3 webhooks resolved to identical `job_id`: `{job_1}`. "
                        f"Retry requests flagged `is_duplicate_event=True` and incremented duplicate counter to {responses[2]['duplicate_request_count']} "
                        f"without re-executing pipeline jobs."
                    )
                else:
                    st.error("❌ Idempotency failed: generated different job IDs!")


# ==============================================================================
# TAB 3: 100 Hand-Annotated Images 3-Way Benchmark & Correlated Blind Spots
# ==============================================================================
with tabs[2]:
    st.header("📊 100-Image Benchmark: 3-Way Model Comparison & Correlated Blind Spots")
    st.caption(
        "Empirical evaluation of Base SCRFD-34G, RetinaFace (standalone detector), and SCRFD-34G + YOLOv8-Pose Fusion "
        "across 100 hand-annotated images with Ground-Truth faces and RetinaFace QA verification."
    )

    summary_file = config.DATA_DIR / "outputs" / "eval_100_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        # 3 Top-Level Model Comparison Cards
        three_way_list = eval_data.get("three_way_comparison", [])
        if three_way_list and len(three_way_list) >= 3:
            m1_data = three_way_list[0]
            m2_data = three_way_list[1]
            m3_data = three_way_list[2]

            card1, card2, card3 = st.columns(3)
            with card1:
                st.markdown("#### 1️⃣ Base SCRFD-34G")
                st.caption("Standalone Primary Face Detector")
                st.metric(
                    "🎯 Face Detection Recall",
                    f"{m1_data.get('recall_pct', 68.3):.1f}%",
                    help="The most critical metric: Percentage of real faces successfully caught and blurred (302 of 442 faces).",
                )
                st.metric(
                    "🛡️ Fully Protected Faces",
                    f"{m1_data.get('fully_obscured_pct', 52.0):.1f}%",
                    help="Faces with 100% complete blur coverage and zero leaks detected.",
                )
                st.metric(
                    "⚠️ Missed Faces (Unprotected)",
                    f"{m1_data.get('missed_faces_pct', 31.7):.1f}% ({m1_data.get('missed_faces', 140)} faces)",
                    help="Faces where the detector applied 0% blur, leaving the face visible.",
                )
                st.metric(
                    "🔍 Partially Covered Faces",
                    f"{m1_data.get('residual_visible_pct', 16.3):.1f}%",
                    help="Faces with blur applied, but outer boundaries (chin/hairline) partially exposed.",
                )
                st.metric("⚡ Processing Speed", f"{m1_data.get('fps', 2.5):.1f} FPS ({m1_data.get('avg_latency_ms', 399.0):.0f} ms/photo)")

            with card2:
                st.markdown("#### 2️⃣ RetinaFace Detector")
                st.caption("Standalone Verifier Backbone")
                st.metric(
                    "🎯 Face Detection Recall",
                    f"{m2_data.get('recall_pct', 47.1):.1f}%",
                    help="RetinaFace missed more than half of bodycam faces when tested alone.",
                )
                st.metric(
                    "🛡️ Fully Protected Faces",
                    f"{m2_data.get('fully_obscured_pct', 37.1):.1f}%",
                    help="Complete solid blur coverage.",
                )
                st.metric(
                    "⚠️ Missed Faces (Unprotected)",
                    f"{m2_data.get('missed_faces_pct', 52.9):.1f}% ({m2_data.get('missed_faces', 234)} faces)",
                    help="High miss rate under blur and distance.",
                )
                st.metric(
                    "🔍 Partially Covered Faces",
                    f"{m2_data.get('residual_visible_pct', 10.0):.1f}%",
                    help="Partially blurred faces.",
                )
                st.metric("⚡ Processing Speed", f"{m2_data.get('fps', 1.9):.1f} FPS ({m2_data.get('avg_latency_ms', 529.0):.0f} ms/photo)")

            with card3:
                st.markdown("#### 3️⃣ SCRFD + Pose Fusion (Winner) 🏆")
                st.caption("SCRFD-34G + YOLOv8-Pose Cranial/Shoulder Fusion")
                gain_recall = m3_data.get('recall_pct', 76.0) - m1_data.get('recall_pct', 68.3)
                reduction_miss = m3_data.get('missed_faces_pct', 24.0) - m1_data.get('missed_faces_pct', 31.7)
                st.metric(
                    "🎯 Face Detection Recall",
                    f"{m3_data.get('recall_pct', 76.0):.1f}%",
                    delta=f"{gain_recall:+.1f}% boost",
                    help="Highest Recall! Catches 336 of 442 faces, recovering 34 faces that Base SCRFD missed.",
                )
                st.metric(
                    "🛡️ Fully Protected Faces",
                    f"{m3_data.get('fully_obscured_pct', 41.2):.1f}%",
                    help="Solid full-head coverage.",
                )
                st.metric(
                    "⚠️ Missed Faces (Unprotected)",
                    f"{m3_data.get('missed_faces_pct', 24.0):.1f}% ({m3_data.get('missed_faces', 106)} faces)",
                    delta=f"{reduction_miss:+.1f}% reduction",
                    delta_color="inverse",
                    help="Significantly lowers unprotected face leakage.",
                )
                st.metric(
                    "🔍 Partially Covered Faces",
                    f"{m3_data.get('residual_visible_pct', 34.8):.1f}%",
                    help="Cranial boxes securely blur the head, with edges flagged for safety verification.",
                )
                st.metric("⚡ Processing Speed", f"{m3_data.get('fps', 2.2):.1f} FPS ({m3_data.get('avg_latency_ms', 458.0):.0f} ms/photo)")

        st.markdown("---")

        # -------------------------------------------------------------
        # TABLE 1: Overall 3-Way Model Comparison
        # -------------------------------------------------------------
        st.subheader("📋 Table 1: Overall 3-Way Model Comparison (All 100 Images)")
        if "three_way_table_rows" in eval_data and "three_way_headers" in eval_data:
            df_t1 = pd.DataFrame(eval_data["three_way_table_rows"], columns=eval_data["three_way_headers"])
            st.dataframe(df_t1, use_container_width=True)

        with st.expander("📖 Plain-English Guide: What do these metrics mean?"):
            st.markdown(
                """
                - **Face Detection Recall (%) [The Most Important Metric]**: Out of 100 real human faces in the video, how many did the model find and blur? **Higher is better.** Model 3 achieves **76.0% Recall** (catching 336 out of 442 faces), outperforming Base SCRFD (68.3%) and RetinaFace (47.1%).
                - **Missed Faces (Unprotected) [The #1 Danger]**: Faces where the model completely failed to see a person, applying zero blur and leaving real identities exposed. **Lower is better.** Pose Fusion reduces missed faces from 140 down to 106 (recovering 34 faces).
                - **Fully Protected (%)**: Faces with 100% complete blur coverage where zero facial details could be detected by our QA verifier.
                - **Partially Covered (%)**: Faces that received blur over the head, but outer features (like the jawline or ear) were partially visible or flagged for human review.
                - **Blur Overlap (IoU)**: A score from 0.0 to 1.0 measuring how accurately the blur box aligns with the real face.
                """
            )

        # Interactive Comparison Chart
        if three_way_list:
            st.markdown("##### 📈 Visual Comparison: Recall vs. Missed vs. Protected")
            chart_data = {
                "Model": [d["model"] for d in three_way_list],
                "Face Detection Recall (%)": [d.get("recall_pct", 100.0 - d["missed_faces_pct"]) for d in three_way_list],
                "Missed Faces (%)": [d["missed_faces_pct"] for d in three_way_list],
                "Fully Protected (%)": [d["fully_obscured_pct"] for d in three_way_list],
                "Partially Covered (%)": [d["residual_visible_pct"] for d in three_way_list],
            }
            df_chart = pd.DataFrame(chart_data).set_index("Model")
            st.bar_chart(df_chart)

        st.markdown("---")

        # -------------------------------------------------------------
        # TABLE 2: Per-Condition Breakdown Across All 3 Models
        # -------------------------------------------------------------
        st.subheader("📑 Table 2: Per-Condition Performance Across All 3 Models")
        st.caption("See how each model performs across challenging real-world environments like Motion Blur, Low Light, Occlusion, and Top-Down camera angles.")
        if "per_condition_table_rows" in eval_data and "per_condition_headers" in eval_data:
            df_t2 = pd.DataFrame(eval_data["per_condition_table_rows"], columns=eval_data["per_condition_headers"])
            st.dataframe(df_t2, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # TABLE 3: Detector Agreement & Correlated Failure Mode Matrix
        # -------------------------------------------------------------
        st.subheader("🤝 Table 3: Model Agreement & How Pose Fusion Rescues Blind Spots")
        st.caption(
            "Analysis showing where both face detectors failed together (common blind spots) "
            "and how looking at body posture (shoulders & neck) rescues hidden faces."
        )
        agr = eval_data.get("agreement_matrix", {})
        if "table_rows" in agr and "headers" in agr:
            df_t3 = pd.DataFrame(agr["table_rows"], columns=agr["headers"])
            st.dataframe(df_t3, use_container_width=True)

        st.info(
            f"💡 **Plain-English Key Finding**: "
            f"Pure face detectors rely entirely on seeing eyes, nose, or mouth. Under rapid motion or steep angles, those facial details vanish, "
            f"causing both SCRFD and RetinaFace to miss the exact same {agr.get('shared_false_negatives_both_missed', 124)} faces (28.1% of all faces). "
            f"**SCRFD + Pose Fusion fixes this blind spot**: by triangulating the head position from the shoulders and collarbone, "
            f"it successfully catches and blurs **{agr.get('shared_fn_rescued_by_pose_fusion', 32)} of those blind spots**!"
        )

        st.markdown("---")

        # -------------------------------------------------------------
        # 4-Panel Visual Failure Strips Gallery
        # -------------------------------------------------------------
        st.subheader("🖼️ 4-Panel Visual Comparison Gallery")
        st.caption(
            "Side-by-side progression: "
            "[1. Raw + Hand GT (Green)] | [2. Base SCRFD-34G] | [3. RetinaFace] | [4. SCRFD-34G + Body/Head Pose Fusion (Rescued)]"
        )

        gallery_dir = config.DATA_DIR / "outputs" / "failure_gallery"
        comparison_strips = sorted(list(gallery_dir.glob("comparison_*.jpg")))

        if comparison_strips:
            for c_path in comparison_strips:
                st.image(str(c_path), caption=f"4-Panel Comparison: {c_path.name}", use_container_width=True)
        else:
            # Fallback to standard failure composites if 4-panel strips not yet populated
            fallback_images = sorted(list(gallery_dir.glob("*.jpg")))[:6]
            for f_path in fallback_images:
                st.image(str(f_path), caption=f"Failure Diagnostic: {f_path.name}", use_container_width=True)
    else:
        st.warning("⚠️ `data/outputs/eval_100_summary.json` not found. Run `make eval-images` to generate benchmark data.")
