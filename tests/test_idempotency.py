"""Idempotency and concurrency test suite for Face Anonymization & Verification API."""

import concurrent.futures
import io
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import config
from app.db import compute_idempotency_key, get_connection, init_db
from app.main import app


@pytest.fixture(autouse=True)
def isolated_db_and_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate DB and test directories to a clean temporary folder for each test."""
    test_db_path = tmp_path / "test_jobs.db"
    test_video_dir = tmp_path / "video_snippets"
    test_video_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "JOBS_DB_PATH", test_db_path)
    monkeypatch.setattr(config, "VIDEO_SNIPPETS_DIR", test_video_dir)
    init_db(test_db_path)
    yield test_db_path


def test_sqlite_wal_mode_enabled(isolated_db_and_dirs: Path):
    """Verify that SQLite runs in WAL mode as required for high-concurrency state storage."""
    with get_connection(isolated_db_and_dirs) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        row = cursor.fetchone()
        journal_mode = row[0].lower()
        assert journal_mode == "wal", f"Expected WAL mode, got {journal_mode}"


def test_repeated_post_jobs_idempotency(isolated_db_and_dirs: Path):
    """
    Assert that sending 3 repeated POST /jobs requests with the exact same
    event_id and video payload:
    1. Returns the exact same job_id for all 3 responses.
    2. Results in exactly 1 row in the database for that idempotency_key.
    3. Results in duplicate_request_count == 2 after the 2 retries.
    """
    client = TestClient(app)

    event_id = "EVT-BODYCAM-2026-1027-ALPHA"
    dummy_video_bytes = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free"

    responses = []
    for i in range(3):
        # Simulate upload
        file_payload = ("test_bodycam_snippet.mp4", io.BytesIO(dummy_video_bytes), "video/mp4")
        resp = client.post(
            "/jobs",
            data={"event_id": event_id},
            files={"file": file_payload},
        )
        assert resp.status_code == 201, f"Request {i+1} failed: {resp.text}"
        responses.append(resp.json())

    # 1. All 3 responses return the exact same job_id
    job_id_1 = responses[0]["job_id"]
    job_id_2 = responses[1]["job_id"]
    job_id_3 = responses[2]["job_id"]

    assert job_id_1 == job_id_2 == job_id_3, (
        f"job_id mismatch across duplicate submissions: {job_id_1}, {job_id_2}, {job_id_3}"
    )

    # Validate duplicate flags and counts per response
    assert responses[0]["is_duplicate_event"] is False
    assert responses[0]["duplicate_request_count"] == 0

    assert responses[1]["is_duplicate_event"] is True
    assert responses[1]["duplicate_request_count"] == 1

    assert responses[2]["is_duplicate_event"] is True
    assert responses[2]["duplicate_request_count"] == 2

    # 2. Database contains exactly 1 row for that idempotency_key
    with get_connection(isolated_db_and_dirs) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs")
        all_jobs = cursor.fetchall()
        assert len(all_jobs) == 1, f"Expected exactly 1 job row, found {len(all_jobs)}"

        row = dict(all_jobs[0])
        assert row["job_id"] == job_id_1
        assert row["event_id"] == event_id
        # 3. duplicate_request_count equals 2 in the database
        assert row["duplicate_request_count"] == 2

        expected_key = compute_idempotency_key(event_id, row["file_sha256"])
        assert row["idempotency_key"] == expected_key


def test_concurrent_post_jobs_idempotency(isolated_db_and_dirs: Path):
    """
    Assert that sending 3 concurrent POST /jobs requests simultaneously
    with the exact same event_id and payload cleanly handles SQLite WAL concurrency,
    returning the same job_id and producing exactly 1 row with duplicate_request_count == 2.
    """
    client = TestClient(app)

    event_id = "EVT-CONCURRENT-BURST-001"
    dummy_video_bytes = b"MP4_SAMPLE_BURST_DATA_CONCURRENT_SAFETY_CRITICAL_2026"

    def submit():
        file_payload = ("burst_bodycam.mp4", io.BytesIO(dummy_video_bytes), "video/mp4")
        return client.post(
            "/jobs",
            data={"event_id": event_id},
            files={"file": file_payload},
        )

    # Launch 3 requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(submit) for _ in range(3)]
        results = [f.result() for f in futures]

    for resp in results:
        assert resp.status_code == 201, f"Concurrent request returned {resp.status_code}: {resp.text}"

    data_list = [r.json() for r in results]
    job_ids = {d["job_id"] for d in data_list}

    # All must agree on the same job_id
    assert len(job_ids) == 1, f"Concurrent requests generated different job_ids: {job_ids}"

    single_job_id = list(job_ids)[0]

    # Inspect final state in DB
    with get_connection(isolated_db_and_dirs) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs")
        rows = cursor.fetchall()
        assert len(rows) == 1, f"Expected 1 database row after concurrency, got {len(rows)}"

        db_job = dict(rows[0])
        assert db_job["job_id"] == single_job_id
        assert db_job["duplicate_request_count"] == 2


def test_different_events_create_separate_jobs(isolated_db_and_dirs: Path):
    """Assert that different event_ids create distinct jobs."""
    client = TestClient(app)
    video_bytes = b"SAMPLE_PAYLOAD_DIFFERENT_EVENTS"

    resp1 = client.post(
        "/jobs",
        data={"event_id": "EVT-A"},
        files={"file": ("v1.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    resp2 = client.post(
        "/jobs",
        data={"event_id": "EVT-B"},
        files={"file": ("v2.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )

    assert resp1.status_code == 201
    assert resp2.status_code == 201

    data1 = resp1.json()
    data2 = resp2.json()

    assert data1["job_id"] != data2["job_id"]
    assert data1["event_id"] == "EVT-A"
    assert data2["event_id"] == "EVT-B"

    # Listing jobs
    list_resp = client.get("/jobs")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 2


def test_review_action_flow(isolated_db_and_dirs: Path):
    """Verify that human reviewer can update status to SAFE override or confirmed NEEDS_REVIEW with notes."""
    client = TestClient(app)
    video_bytes = b"SAMPLE_PAYLOAD_FOR_REVIEW"

    resp = client.post(
        "/jobs",
        data={"event_id": "EVT-REVIEW-TEST"},
        files={"file": ("review.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]

    # Review action: SAFE override
    action_resp = client.post(
        f"/jobs/{job_id}/review_action",
        json={"status": "SAFE", "reviewer_notes": "All faces confirmed fully redacted by operator #42."},
    )
    assert action_resp.status_code == 200
    review_data = action_resp.json()
    assert review_data["status"] == "SAFE"
    assert review_data["reviewer_notes"] == "All faces confirmed fully redacted by operator #42."

    # Fetch by job ID
    get_resp = client.get(f"/jobs/{job_id}")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["status"] == "SAFE"
    assert get_data["reviewer_notes"] == "All faces confirmed fully redacted by operator #42."
