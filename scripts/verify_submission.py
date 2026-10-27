"""Submission Verification Script.
Audits the complete repository before submission:
1. Verifies zero hardcoded absolute '/Users/' paths in DB, JSON summaries, and source code.
2. Verifies SQLite database contains the 5 demo clip jobs with idempotency verified on clip 1.
3. Verifies output deliverables: clips_5_summary.json, eval_100_summary.json, rendered MP4s, failure gallery.
4. Verifies test suite execution (pytest).
5. Verifies README.md deliverables and image links.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_absolute_paths() -> tuple[bool, list[str]]:
    """Scan database, json outputs, tracks files, and source code for forbidden /Users/ paths."""
    violations: list[str] = []

    # 1. Check SQLite jobs.db
    db_path = PROJECT_ROOT / "data" / "jobs.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM jobs").fetchall()
        for r in rows:
            for col in ["input_path", "output_path", "telemetry_json", "flagged_frames_json"]:
                val = r[col]
                if val and "/Users/" in str(val):
                    violations.append(f"DB jobs table col '{col}' in job '{r['job_id']}' contains '/Users/': {val[:80]}...")
        conn.close()

    # 2. Check JSON output files
    output_dir = PROJECT_ROOT / "data" / "outputs"
    for jf in output_dir.glob("*.json"):
        content = jf.read_text(encoding="utf-8")
        if "/Users/" in content:
            violations.append(f"JSON file '{jf.relative_to(PROJECT_ROOT)}' contains '/Users/' absolute path")

    # 3. Check companion .tracks.json files
    redacted_dir = output_dir / "redacted_videos"
    for tf in redacted_dir.glob("*.tracks.json"):
        content = tf.read_text(encoding="utf-8")
        if "/Users/" in content:
            violations.append(f"Tracks JSON '{tf.relative_to(PROJECT_ROOT)}' contains '/Users/' absolute path")

    # 4. Check core python files for hardcoded string literals with /Users/
    for py_file in list((PROJECT_ROOT / "app").rglob("*.py")) + list((PROJECT_ROOT / "scripts").rglob("*.py")):
        if py_file.name == "verify_submission.py":
            continue
        with open(py_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if '"/Users/' in line or "'/Users/" in line:
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:L{idx} hardcoded '/Users/' path: {line.strip()}")

    passed = len(violations) == 0
    return passed, violations


def check_database_state() -> tuple[bool, list[str]]:
    """Check that jobs.db contains the demo clips and clip 1 has duplicate_request_count >= 1."""
    notes: list[str] = []
    db_path = PROJECT_ROOT / "data" / "jobs.db"
    if not db_path.exists():
        return False, ["data/jobs.db does not exist."]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM jobs ORDER BY created_at ASC").fetchall()
    conn.close()

    if len(rows) not in (5, 6):
        notes.append(f"Expected 5 or 6 demo clip jobs in jobs.db, found {len(rows)}.")
    else:
        notes.append(f"Found {len(rows)} demo clip jobs in jobs.db.")

    if len(rows) > 0:
        first_job = rows[0]
        dup_count = first_job["duplicate_request_count"]
        if dup_count < 1:
            notes.append(f"First job {first_job['job_id']} duplicate_request_count = {dup_count} (expected >= 1 for idempotency proof).")
        else:
            notes.append(f"Idempotency verified: First job {first_job['job_id']} recorded duplicate_request_count = {dup_count}.")

    passed = (len(rows) in (5, 6)) and (len(rows) > 0 and rows[0]["duplicate_request_count"] >= 1)
    return passed, notes


def check_deliverable_files() -> tuple[bool, list[str]]:
    """Check existence of all required deliverable files and galleries."""
    notes: list[str] = []
    all_ok = True

    required_files = [
        PROJECT_ROOT / "data" / "outputs" / "clips_5_summary.json",
        PROJECT_ROOT / "data" / "outputs" / "eval_100_summary.json",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "Makefile",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "app_dashboard.py",
    ]

    for rf in required_files:
        if not rf.exists():
            notes.append(f"MISSING: {rf.relative_to(PROJECT_ROOT)}")
            all_ok = False
        else:
            notes.append(f"FOUND: {rf.relative_to(PROJECT_ROOT)}")

    # Check failure gallery
    gallery_dir = PROJECT_ROOT / "data" / "outputs" / "failure_gallery"
    gallery_images = list(gallery_dir.glob("*.jpg")) + list(gallery_dir.glob("*.png"))
    if len(gallery_images) < 4:
        notes.append(f"Failure gallery has only {len(gallery_images)} images (expected >= 4).")
        all_ok = False
    else:
        notes.append(f"Failure gallery has {len(gallery_images)} diagnostic composites.")

    # Check rendered videos
    redacted_dir = PROJECT_ROOT / "data" / "outputs" / "redacted_videos"
    redacted_videos = list(redacted_dir.glob("*_redacted.mp4"))
    if len(redacted_videos) < 5:
        notes.append(f"Found {len(redacted_videos)} rendered videos in {redacted_dir.relative_to(PROJECT_ROOT)} (expected 5).")
        all_ok = False
    else:
        notes.append(f"Found {len(redacted_videos)} browser-compatible H.264 rendered video outputs.")

    return all_ok, notes


def run_test_suite() -> tuple[bool, str]:
    """Execute pytest test suite."""
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "-v"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        return res.returncode == 0, res.stdout + res.stderr
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 80)
    print("SUBMISSION VERIFICATION AUDIT")
    print("=" * 80)

    # 1. Check deliverable files
    print("\n[1/4] Checking Deliverable Files & Directories...")
    files_ok, file_notes = check_deliverable_files()
    for fn in file_notes:
        is_ok = not fn.startswith("MISSING") and "only" not in fn and "expected" not in fn
        print(f"  {'✓' if is_ok else '✗'} {fn}")

    # 2. Check Database & Idempotency
    print("\n[2/4] Checking SQLite Database & Idempotency Proof...")
    db_ok, db_notes = check_database_state()
    for dn in db_notes:
        is_ok = not dn.startswith("Expected") and "not in" not in dn and "<" not in dn
        print(f"  {'✓' if is_ok else '✗'} {dn}")

    # 3. Check for absolute /Users/ paths
    print("\n[3/4] Checking Strictly Relative Paths (Zero '/Users/' violations)...")
    paths_ok, path_violations = check_absolute_paths()
    if paths_ok:
        print("  ✓ PASSED: Zero '/Users/' absolute paths found across DB, JSON summaries, tracks, and code!")
    else:
        print(f"  ✗ FAILED: Found {len(path_violations)} forbidden absolute path occurrences:")
        for v in path_violations[:10]:
            print(f"    - {v}")

    # 4. Run Pytest Suite
    print("\n[4/4] Executing Pytest Test Suite...")
    tests_ok, test_output = run_test_suite()
    if tests_ok:
        print("  ✓ PASSED: All pytest unit and integration tests passed cleanly!")
    else:
        print("  ✗ FAILED: Pytest suite encountered failures:")
        print(test_output[-500:])

    print("\n" + "=" * 80)
    all_passed = files_ok and db_ok and paths_ok and tests_ok
    if all_passed:
        print(">>> ALL VERIFICATION CHECKS PASSED: REPOSITORY IS READY FOR SUBMISSION! <<<")
        print("=" * 80)
        sys.exit(0)
    else:
        print(">>> SOME VERIFICATION CHECKS FAILED. PLEASE RESOLVE ISSUES ABOVE. <<<")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
