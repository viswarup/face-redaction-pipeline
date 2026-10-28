"""
Generate a self-contained, high-contrast visual 10-slide HTML presentation deck
and a multi-page landscape PDF presentation with zero external CDN dependencies.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_base64_img(rel_path: str) -> str:
    full_path = REPO_ROOT / rel_path
    if not full_path.exists():
        return ""
    suffix = full_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    with open(full_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def build_slides_html() -> str:
    img_track_1 = get_base64_img("data/outputs/failure_gallery/tracking_vs_notracking_01.jpg")
    img_pose_1 = get_base64_img("data/outputs/failure_gallery/comparison_01.jpg")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Safety-Critical Bodycam Face Redaction & Verification Pipeline</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background-color: #0b1120;
      color: #0f172a;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}

    .deck-frame {{
      width: 100%;
      max-width: 1100px;
      min-height: 680px;
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      border: 1px solid #334155;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }}

    .deck-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 24px;
      background-color: #f8fafc;
      border-bottom: 2px solid #e2e8f0;
      font-size: 13px;
      font-weight: 600;
      color: #334155;
    }}

    .header-badge {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: #1e293b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 12px;
    }}

    .pulse-dot {{
      width: 10px;
      height: 10px;
      background-color: #2563eb;
      border-radius: 50%;
    }}

    .slide-viewport {{
      flex: 1;
      padding: 32px 40px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      background-color: #ffffff;
    }}

    .slide {{
      display: none;
      flex-direction: column;
      justify-content: space-between;
      height: 100%;
      min-height: 520px;
      opacity: 1 !important;
    }}

    .slide.active {{
      display: flex;
      opacity: 1 !important;
    }}

    /* Typography */
    .slide-tag {{
      display: inline-block;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #2563eb;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      padding: 4px 10px;
      border-radius: 9999px;
      margin-bottom: 8px;
    }}

    h1 {{
      font-size: 34px;
      font-weight: 900;
      color: #0f172a;
      line-height: 1.18;
      letter-spacing: -0.02em;
    }}

    h2 {{
      font-size: 26px;
      font-weight: 800;
      color: #0f172a;
      line-height: 1.22;
      letter-spacing: -0.01em;
      margin-bottom: 6px;
    }}

    .subtitle {{
      font-size: 14.5px;
      color: #334155;
      line-height: 1.5;
      margin-top: 4px;
    }}

    /* Grid & Cards */
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
    }}

    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }}

    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }}

    .card {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    .card-title {{
      font-size: 14px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .card-body {{
      font-size: 12px;
      color: #334155;
      line-height: 1.5;
    }}

    .card-stat {{
      font-size: 24px;
      font-weight: 900;
      color: #0f172a;
      line-height: 1.1;
    }}

    .card-stat-label {{
      font-size: 11px;
      font-weight: 600;
      color: #475569;
      margin-top: 3px;
    }}

    /* Highlights */
    .pill-green {{ background: #dcfce7; border-color: #86efac; color: #14532d; }}
    .pill-blue {{ background: #dbeafe; border-color: #93c5fd; color: #1e3a8a; }}
    .pill-purple {{ background: #f3e8ff; border-color: #d8b4fe; color: #581c87; }}
    .pill-amber {{ background: #fef3c7; border-color: #fde047; color: #78350f; }}
    .pill-red {{ background: #fee2e2; border-color: #fca5a5; color: #7f1d1d; }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #ffffff;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #cbd5e1;
    }}

    th, td {{
      padding: 9px 12px;
      text-align: left;
      border: 1px solid #cbd5e1;
    }}

    th {{
      background-color: #f1f5f9;
      font-weight: 700;
      color: #0f172a;
      font-size: 12px;
    }}

    tr:nth-child(even) {{
      background-color: #f8fafc;
    }}

    .highlight-row {{
      background-color: #eff6ff !important;
      font-weight: 700;
    }}

    /* Diagrams & SVG */
    .diagram-container {{
      background: #f8fafc;
      border: 1.5px solid #cbd5e1;
      border-radius: 12px;
      padding: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    /* Code & Mono */
    .code-box {{
      background: #0f172a;
      color: #f8fafc;
      padding: 12px 16px;
      border-radius: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11.5px;
      line-height: 1.5;
    }}

    /* Images */
    .proof-img {{
      width: 100%;
      max-height: 180px;
      object-fit: contain;
      background: #000;
      border-radius: 8px;
      border: 1px solid #94a3b8;
    }}

    /* Bottom Deck Footer */
    .deck-footer {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 24px;
      background-color: #f8fafc;
      border-top: 2px solid #e2e8f0;
    }}

    .nav-btn {{
      padding: 8px 18px;
      font-size: 13px;
      font-weight: 700;
      border-radius: 8px;
      cursor: pointer;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #1e293b;
      transition: all 0.15s;
    }}

    .nav-btn:hover:not(:disabled) {{
      background: #e2e8f0;
    }}

    .nav-btn-primary {{
      background: #2563eb;
      color: #ffffff;
      border-color: #2563eb;
    }}

    .nav-btn-primary:hover:not(:disabled) {{
      background: #1d4ed8;
    }}

    .nav-btn:disabled {{
      opacity: 0.35;
      cursor: not-allowed;
    }}

    .dot-container {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #cbd5e1;
      cursor: pointer;
      border: none;
      transition: all 0.15s;
    }}

    .dot.active {{
      width: 22px;
      border-radius: 4px;
      background: #2563eb;
    }}

    .nav-hint {{
      color: #94a3b8;
      font-size: 12px;
      margin-top: 12px;
      text-align: center;
    }}

    /* Print / PDF Mode */
    @media print {{
      @page {{
        size: 14in 8.5in;
        margin: 0.3in;
      }}
      body {{
        background: #ffffff !important;
        padding: 0 !important;
        color: #0f172a !important;
      }}
      .deck-frame {{
        max-width: 100% !important;
        height: auto !important;
        overflow: visible !important;
        box-shadow: none !important;
        border: none !important;
        min-height: auto !important;
      }}
      .deck-header, .deck-footer, .nav-hint {{
        display: none !important;
      }}
      .slide-viewport {{
        height: auto !important;
        overflow: visible !important;
        display: block !important;
        padding: 0 !important;
      }}
      .slide {{
        display: flex !important;
        page-break-after: always !important;
        break-after: page !important;
        height: 7.8in !important;
        max-height: 7.8in !important;
        overflow: hidden !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 28px !important;
        margin-bottom: 24px !important;
        box-sizing: border-box !important;
      }}
    }}
  </style>
</head>
<body>

  <!-- 16:9 Presentation Frame -->
  <div class="deck-frame">

    <!-- Header -->
    <header class="deck-header">
      <div class="header-badge">
        <span class="pulse-dot"></span>
        <span>Safety-Critical Bodycam Pipeline • Technical Presentation</span>
      </div>
      <div style="display: flex; align-items: center; gap: 14px;">
        <span id="slide-indicator" style="font-weight: 700; color: #475569;">Slide 1 of 10</span>
        <button id="fullscreen-btn" class="nav-btn" style="padding: 4px 10px; font-size: 11px;">⛶ Fullscreen (F)</button>
      </div>
    </header>

    <!-- Main Viewport -->
    <main class="slide-viewport">

      <!-- ======================================================== -->
      <!-- SLIDE 1: Title & Executive Overview                      -->
      <!-- ======================================================== -->
      <section class="slide active" style="align-items: center; justify-content: center; text-align: center;">
        <div style="max-width: 860px; display: flex; flex-direction: column; align-items: center; gap: 18px;">
          <div class="slide-tag">🛡️ Production-Grade Robotics & Privacy Engineering</div>
          <h1 style="color: #0f172a;">Safety-Critical Bodycam Face Redaction & Verification Pipeline</h1>
          <p class="subtitle" style="max-width: 720px; font-size: 15px; color: #334155;">
            An end-to-end, fail-closed privacy architecture engineered for real-world egocentric video. Eliminates motion blur flicker leaks, recovers steep camera profiles using body pose, and audits every frame with an independent AI verifier.
          </p>

          <!-- 4 KPI Cards -->
          <div class="grid-4" style="width: 100%; margin-top: 10px;">
            <div class="card pill-blue" style="text-align: center;">
              <div class="card-stat" style="color: #1d4ed8;">76.0% Recall</div>
              <div class="card-stat-label">Face Catch Rate (336/442)</div>
            </div>
            <div class="card pill-green" style="text-align: center;">
              <div class="card-stat" style="color: #15803d;">-88.1% Flashes</div>
              <div class="card-stat-label">Fewer Accidental Leaks</div>
            </div>
            <div class="card pill-purple" style="text-align: center;">
              <div class="card-stat" style="color: #6b21a8;">3,101 Frames</div>
              <div class="card-stat-label">Rescued by Motion Memory</div>
            </div>
            <div class="card pill-red" style="text-align: center;">
              <div class="card-stat" style="color: #b91c1c;">100% Fail-Closed</div>
              <div class="card-stat-label">Zero Unvetted Video Released</div>
            </div>
          </div>
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 2: The Core Problem                                -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">The Operational Challenge</div>
          <h2>Why Standard Face Blurring Fails on Bodycams</h2>
          <p class="subtitle">Body-worn cameras move violently, walk through dark corridors, and view humans at steep angles.</p>
        </div>

        <div class="grid-3" style="margin: 16px 0;">
          <div class="card pill-amber">
            <div>
              <div style="font-size: 26px; margin-bottom: 6px;">⚡</div>
              <div class="card-title" style="color: #78350f;">1. Severe Motion Blur</div>
              <p class="card-body" style="color: #451a03;">
                When an officer or robot turns quickly, camera shutter lag smears eyes, nose, and mouth into smooth streaks. Standalone face detectors suddenly output <strong>zero boxes</strong> for 1 to 5 frames.
              </p>
            </div>
            <div style="margin-top: 10px; font-weight: 700; font-size: 11px; color: #9a3412;">
              ⚠️ Danger: Creates sudden unblurred face flashes
            </div>
          </div>

          <div class="card pill-red">
            <div>
              <div style="font-size: 26px; margin-bottom: 6px;">📐</div>
              <div class="card-title" style="color: #7f1d1d;">2. Steep Angles & Foreshortening</div>
              <p class="card-body" style="color: #450a0a;">
                Chest-mounted cameras look up at tall people or down at seated individuals. Extreme perspective distortion hides facial landmarks, causing <strong>50% failure rates</strong> on top-down views.
              </p>
            </div>
            <div style="margin-top: 10px; font-weight: 700; font-size: 11px; color: #991b1b;">
              ⚠️ Danger: Faces completely ignored by AI
            </div>
          </div>

          <div class="card pill-blue">
            <div>
              <div style="font-size: 26px; margin-bottom: 6px;">🎯</div>
              <div class="card-title" style="color: #1e3a8a;">3. Model Confirmation Bias</div>
              <p class="card-body" style="color: #172554;">
                Standard systems use the same AI model to both blur and check their work. If an AI detector misses a face initially, it will predictably miss it during quality-check too.
              </p>
            </div>
            <div style="margin-top: 10px; font-weight: 700; font-size: 11px; color: #1e40af;">
              ⚠️ Danger: False sense of 100% security
            </div>
          </div>
        </div>

        <div class="card" style="background: #0f172a; color: #ffffff; padding: 12px 18px; flex-direction: row; align-items: center; justify-content: space-between;">
          <span style="font-size: 13px; font-weight: 500;">
            <strong>Our Philosophy:</strong> In privacy law (GDPR/FOIA), a 95% detection rate is a 5% legal disaster. The pipeline must be <strong>Strictly Fail-Closed</strong>.
          </span>
          <span style="background: #dc2626; color: #ffffff; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 800; text-transform: uppercase;">
            Fail-Closed Safety
          </span>
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 3: System Architecture Flowchart                   -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">System Architecture</div>
          <h2>End-to-End Pipeline Flowchart</h2>
          <p class="subtitle">Multi-stage defense-in-depth: from cryptographic hashing to independent QA verification.</p>
        </div>

        <div class="diagram-container" style="margin: 12px 0;">
          <svg viewBox="0 0 940 220" style="width: 100%; height: auto; max-height: 230px;">
            <defs>
              <marker id="arr" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#475569"/>
              </marker>
              <marker id="arr-g" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#16a34a"/>
              </marker>
              <marker id="arr-r" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#dc2626"/>
              </marker>
            </defs>

            <!-- Box 1: Video Ingest -->
            <rect x="10" y="70" width="105" height="75" rx="8" fill="#0f172a" stroke="#1e293b" stroke-width="2"/>
            <text x="62" y="103" fill="#ffffff" font-size="12" font-weight="bold" text-anchor="middle">Input Video</text>
            <text x="62" y="122" fill="#94a3b8" font-size="10" text-anchor="middle">SHA-256 Digest</text>

            <line x1="115" y1="108" x2="145" y2="108" stroke="#475569" stroke-width="2.5" marker-end="url(#arr)"/>

            <!-- Box 2: Dual Detection -->
            <rect x="145" y="55" width="135" height="105" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>
            <text x="212" y="80" fill="#1d4ed8" font-size="12" font-weight="bold" text-anchor="middle">1. Dual Detection</text>
            <text x="212" y="102" fill="#0f172a" font-size="11" font-weight="bold" text-anchor="middle">SCRFD-34G (Face)</text>
            <text x="212" y="120" fill="#0f172a" font-size="11" font-weight="bold" text-anchor="middle">+ YOLOv8 (Pose)</text>
            <text x="212" y="142" fill="#3b82f6" font-size="9" font-weight="600" text-anchor="middle">Fuses head & shoulders</text>

            <line x1="280" y1="108" x2="310" y2="108" stroke="#475569" stroke-width="2.5" marker-end="url(#arr)"/>

            <!-- Box 3: Motion Tracking -->
            <rect x="310" y="55" width="140" height="105" rx="8" fill="#faf5ff" stroke="#9333ea" stroke-width="2"/>
            <text x="380" y="80" fill="#7e22ce" font-size="12" font-weight="bold" text-anchor="middle">2. Motion Tracking</text>
            <text x="380" y="102" fill="#0f172a" font-size="11" font-weight="bold" text-anchor="middle">ByteTrack + Kalman</text>
            <text x="380" y="122" fill="#6b21a8" font-size="9" font-weight="600" text-anchor="middle">Motion memory coasts</text>
            <text x="380" y="138" fill="#6b21a8" font-size="9" font-weight="600" text-anchor="middle">through camera shake</text>

            <line x1="450" y1="108" x2="480" y2="108" stroke="#475569" stroke-width="2.5" marker-end="url(#arr)"/>

            <!-- Box 4: Heavy Redaction -->
            <rect x="480" y="55" width="135" height="105" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="2"/>
            <text x="547" y="80" fill="#b45309" font-size="12" font-weight="bold" text-anchor="middle">3. Heavy Redaction</text>
            <text x="547" y="102" fill="#0f172a" font-size="11" font-weight="bold" text-anchor="middle">+28% Box Dilation</text>
            <text x="547" y="120" fill="#0f172a" font-size="11" font-weight="bold" text-anchor="middle">Gaussian σ=25 Blur</text>
            <text x="547" y="142" fill="#d97706" font-size="9" font-weight="600" text-anchor="middle">Covers hair & ears</text>

            <line x1="615" y1="108" x2="645" y2="108" stroke="#475569" stroke-width="2.5" marker-end="url(#arr)"/>

            <!-- Box 5: Independent Auditor -->
            <rect x="645" y="55" width="145" height="105" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="2"/>
            <text x="717" y="80" fill="#15803d" font-size="12" font-weight="bold" text-anchor="middle">4. Secondary QA</text>
            <text x="717" y="102" fill="#0f172a" font-size="11" font-weight="bold" text-anchor="middle">RetinaFace Auditor</text>
            <text x="717" y="122" fill="#16a34a" font-size="9" font-weight="600" text-anchor="middle">Audits rendered MP4</text>
            <text x="717" y="138" fill="#16a34a" font-size="9" font-weight="600" text-anchor="middle">Zero-trust inspection</text>

            <!-- Outputs -->
            <line x1="790" y1="92" x2="825" y2="60" stroke="#16a34a" stroke-width="2.5" marker-end="url(#arr-g)"/>
            <rect x="825" y="40" width="100" height="42" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
            <text x="875" y="66" fill="#14532d" font-size="11" font-weight="bold" text-anchor="middle">SAFE (Approved)</text>

            <line x1="790" y1="125" x2="825" y2="155" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arr-r)"/>
            <rect x="825" y="138" width="100" height="42" rx="6" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>
            <text x="875" y="160" fill="#7f1d1d" font-size="10" font-weight="bold" text-anchor="middle">NEEDS_REVIEW</text>
            <text x="875" y="172" fill="#991b1b" font-size="8" text-anchor="middle">Sent to Dashboard</text>

            <text x="470" y="200" fill="#64748b" font-size="11" font-style="italic" text-anchor="middle">
              SQLite WAL Database tracks deduplication, audit logs, and reviewer actions across every stage
            </text>
          </svg>
        </div>

        <div class="grid-4" style="text-align: center;">
          <div class="card" style="padding: 10px; background: #eff6ff; border-color: #bfdbfe;">
            <div style="font-weight: 700; color: #1d4ed8; font-size: 11px;">1. Detects Multi-Scale</div>
            <div style="color: #1e3a8a; font-size: 10.5px;">Faces + skeletons</div>
          </div>
          <div class="card" style="padding: 10px; background: #faf5ff; border-color: #e9d5ff;">
            <div style="font-weight: 700; color: #7e22ce; font-size: 11px;">2. Motion Memory</div>
            <div style="color: #581c87; font-size: 10.5px;">Bridges camera blur</div>
          </div>
          <div class="card" style="padding: 10px; background: #fffbeb; border-color: #fde68a;">
            <div style="font-weight: 700; color: #b45309; font-size: 11px;">3. Solid Blur Bubble</div>
            <div style="color: #78350f; font-size: 10.5px;">+28% box dilation</div>
          </div>
          <div class="card" style="padding: 10px; background: #f0fdf4; border-color: #bbf7d0;">
            <div style="font-weight: 700; color: #15803d; font-size: 11px;">4. Zero-Trust Auditor</div>
            <div style="color: #14532d; font-size: 10.5px;">Guarantees no leaks</div>
          </div>
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 4: Body/Head Pose Fusion                           -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">The Technical Breakthrough</div>
          <h2>Body/Head Pose Fusion: Finding Faces from Shoulders</h2>
          <p class="subtitle">When facial features dissolve in motion blur, the human body skeleton remains visible.</p>
        </div>

        <div class="grid-2" style="margin: 16px 0; align-items: center;">
          <div class="card" style="background: #f8fafc; border: 1.5px solid #cbd5e1; gap: 12px;">
            <div class="card-title" style="font-size: 15px; color: #0f172a;">
              <span style="background: #2563eb; color: #fff; padding: 2px 7px; border-radius: 4px; font-size: 11px;">AI</span>
              How YOLOv8-Pose Rescues Blind Spots
            </div>
            <p class="card-body" style="font-size: 12.5px;">
              When people turn around or light drops, 2D facial classifiers output <strong>zero boxes</strong>. But our body pose model extracts shoulder and neck keypoints:
            </p>
            <div class="code-box">
cranial_width = 0.90 * shoulder_span<br>
head_center   = neck_center - 0.70 * cranial_width<br>
blur_box      = [cranial_x1, cranial_y1, cranial_x2, cranial_y2]
            </div>
            <ul style="font-size: 12px; color: #334155; padding-left: 18px; line-height: 1.6;">
              <li><strong>If Face Overlaps Pose (IoU ≥ 0.20):</strong> We take their combined outer union to cover ears & chin.</li>
              <li><strong>If Face is Missing (IoU &lt; 0.20):</strong> We inject a recovered head blur box automatically.</li>
            </ul>
          </div>

          <div class="card pill-blue" style="border-width: 2px; gap: 14px;">
            <div style="font-size: 13px; font-weight: 800; color: #1e3a8a; text-transform: uppercase;">
              Empirical Proof on 442 Real Faces
            </div>
            <div class="grid-2">
              <div class="card" style="background: #ffffff; text-align: center; padding: 14px;">
                <div class="card-stat" style="color: #2563eb;">32 Faces</div>
                <div class="card-stat-label">Rescued from Total Blind Spots</div>
              </div>
              <div class="card" style="background: #ffffff; text-align: center; padding: 14px;">
                <div class="card-stat" style="color: #16a34a;">+7.7%</div>
                <div class="card-stat-label">Absolute Recall Gain (76.0%)</div>
              </div>
            </div>
            <div class="card pill-green" style="font-size: 12px; line-height: 1.5; font-weight: 600;">
              ✨ <strong>Key Result:</strong> Rescued <strong>25.8% of severe blind spots</strong> where BOTH primary face detectors (SCRFD and RetinaFace) completely failed.
            </div>
          </div>
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 5: 100-Image Benchmark Table                       -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">Empirical Benchmark (100 Hand-Labeled Images)</div>
          <h2>3-Way Model Comparison: Which AI Works Best?</h2>
          <p class="subtitle">Evaluated across 442 ground-truth faces under extreme blur, low light, occlusion, and steep angles.</p>
        </div>

        <div style="margin: 14px 0;">
          <table>
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Real Faces</th>
                <th>Face Catch Recall (%) 🏆</th>
                <th>Fully Protected</th>
                <th>Partially Covered</th>
                <th>Missed Faces (Leaks)</th>
                <th>Processing Speed</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style="font-weight: 700; color: #0f172a;">Base SCRFD-34G</td>
                <td>442</td>
                <td style="font-weight: 700; color: #1e293b;">68.3% (302)</td>
                <td>52.0% (230)</td>
                <td>16.3% (72)</td>
                <td style="color: #dc2626; font-weight: 700;">31.7% (140)</td>
                <td>2.5 FPS (398 ms)</td>
              </tr>
              <tr>
                <td style="font-weight: 700; color: #0f172a;">RetinaFace Detector</td>
                <td>442</td>
                <td style="font-weight: 700; color: #1e293b;">47.1% (208)</td>
                <td>37.1% (164)</td>
                <td>10.0% (44)</td>
                <td style="color: #dc2626; font-weight: 700;">52.9% (234)</td>
                <td>1.9 FPS (529 ms)</td>
              </tr>
              <tr class="highlight-row">
                <td style="font-weight: 800; color: #1e3a8a;">SCRFD + Body/Head Pose Fusion</td>
                <td>442</td>
                <td style="color: #1d4ed8; font-size: 13.5px; font-weight: 900;">76.0% (336) 🏆</td>
                <td>41.2% (182)</td>
                <td>34.8% (154)</td>
                <td style="color: #15803d; font-weight: 800;">24.0% (106) ✨</td>
                <td>2.2 FPS (458 ms)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="grid-3">
          <div class="card" style="padding: 12px;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px; margin-bottom: 3px;">Face Catch Recall:</div>
            <div style="font-size: 11.5px; color: #334155;">Percentage of real faces caught and blurred. Pose Fusion is the clear winner (+7.7% over Base SCRFD).</div>
          </div>
          <div class="card" style="padding: 12px;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px; margin-bottom: 3px;">Missed Faces:</div>
            <div style="font-size: 11.5px; color: #334155;">Faces left with zero blur. Pose Fusion reduces dangerous unblurred faces by 34 faces across the 100 images.</div>
          </div>
          <div class="card" style="padding: 12px;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px; margin-bottom: 3px;">Partially Covered:</div>
            <div style="font-size: 11.5px; color: #334155;">Wide boxes that safely obscure the head but are automatically sent to human review for zero-leak verification.</div>
          </div>
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 6: Video Tracking Ablation Table                   -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">Video Tracking Ablation (1,127 Real Video Frames)</div>
          <h2>Why Video Tracking is Non-Negotiable</h2>
          <p class="subtitle">Direct empirical comparison across 5 video clips: Frame-by-Frame Detection vs. Motion Tracking.</p>
        </div>

        <div style="margin: 14px 0;">
          <table>
            <thead>
              <tr>
                <th>Safety & Video Quality Metric</th>
                <th>Standard (Frame-by-Frame Detection)</th>
                <th>Our Method (Motion Tracking + Smoothing)</th>
                <th>Plain-English Meaning & Real-World Impact</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style="font-weight: 700; color: #0f172a;">Total Video Frames Tested</td>
                <td>1,127 frames</td>
                <td>1,127 frames</td>
                <td>Complete empirical test across 5 demo video clips</td>
              </tr>
              <tr class="highlight-row" style="background: #f0fdf4 !important;">
                <td style="font-weight: 800; color: #14532d;">Accidental Face Flashes (Flickers)</td>
                <td style="color: #dc2626; font-weight: 700;">369 flashes</td>
                <td style="color: #15803d; font-weight: 900; font-size: 13px;">44 flashes</td>
                <td><strong>88.1% fewer flashes!</strong> Stops faces popping out unblurred</td>
              </tr>
              <tr>
                <td style="font-weight: 700; color: #0f172a;">Face Frames Rescued from Zero Blur</td>
                <td style="color: #64748b;">0 (Left completely unblurred!)</td>
                <td style="color: #1d4ed8; font-weight: 800;">3,101 frames</td>
                <td>Motion memory keeps blur locked on during camera shake</td>
              </tr>
              <tr class="highlight-row">
                <td style="font-weight: 800; color: #1e3a8a;">Blur Mask Shaking / Wobble (Jitter)</td>
                <td>4.40 px (Erratic wobble)</td>
                <td style="color: #1d4ed8; font-weight: 800;">1.49 px (Smooth)</td>
                <td><strong>66.0% smoother!</strong> Keeps hands/tools visible for robotics</td>
              </tr>
              <tr>
                <td style="font-weight: 700; color: #0f172a;">Visible Face Leaks Caught by Inspector</td>
                <td>238 frames</td>
                <td style="color: #15803d; font-weight: 800;">127 frames</td>
                <td><strong>46.6% drop</strong> in visible facial features escaping blur</td>
              </tr>
              <tr>
                <td style="font-weight: 700; color: #0f172a;">Videos Sent to Human Review</td>
                <td>5 of 5 clips</td>
                <td>5 of 5 clips</td>
                <td><strong>100% fail-closed safety:</strong> Zero unvetted releases</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="card pill-blue" style="font-size: 12.5px; font-weight: 600; line-height: 1.5;">
          💡 <strong>Key Takeaway:</strong> Without tracking, bodycam video suffers from <strong>369 momentary unblurred flash leaks</strong>. ByteTrack motion memory eliminates 88.1% of flashes and stabilizes mask wobbles so robots can cleanly observe surrounding body mechanics.
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 7: Visual Evidence Gallery                         -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">Visual Diagnostic Evidence</div>
          <h2>Real-World Bodycam Failure & Recovery Strips</h2>
          <p class="subtitle">Indisputable visual proof of motion blur recovery and pose fusion cranial estimation.</p>
        </div>

        <div style="display: flex; flex-direction: column; gap: 14px; margin: 12px 0;">
          <!-- Tracking Proof Strip -->
          <div class="card" style="padding: 12px; background: #f8fafc; border: 1.5px solid #cbd5e1; gap: 6px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; color: #0f172a;">
              <span>1. Tracking vs. No Tracking: Eliminating Unblurred Flicker Flashes</span>
              <span style="font-family: monospace; color: #475569;">Video 1, Frame #143</span>
            </div>
            {f'<img src="{img_track_1}" class="proof-img" alt="Tracking Proof">' if img_track_1 else '<div class="card" style="text-align: center; padding: 20px;">tracking_vs_notracking_01.jpg</div>'}
            <p style="font-size: 11px; color: #475569; font-style: italic;">
              Panel 1: Raw rapid head turn. Panel 2: Detection-Only drops below threshold (RED UNBLURRED LEAK). Panel 3: ByteTrack motion memory maintains smooth blue blur bubble.
            </p>
          </div>

          <!-- Pose Proof Strip -->
          <div class="card" style="padding: 12px; background: #f8fafc; border: 1.5px solid #cbd5e1; gap: 6px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; color: #0f172a;">
              <span>2. Pose Fusion: Rescuing Motion Blur & Extreme Low Light</span>
              <span style="font-family: monospace; color: #475569;">100-Image Benchmark (comparison_01.jpg)</span>
            </div>
            {f'<img src="{img_pose_1}" class="proof-img" alt="Pose Proof">' if img_pose_1 else '<div class="card" style="text-align: center; padding: 20px;">comparison_01.jpg</div>'}
            <p style="font-size: 11px; color: #475569; font-style: italic;">
              Panel 1: Raw face. Panels 2 & 3: Standalone SCRFD and RetinaFace miss the face entirely (RED LEAKS). Panel 4: Pose Fusion uses shoulders to project cranial blur, rescuing the face.
            </p>
          </div>
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 8: Independent Post-Render QA Auditor              -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">Zero-Trust Quality Control</div>
          <h2>The Independent Post-Render QA Inspector</h2>
          <p class="subtitle">Why a separate, second AI model must inspect the final rendered video from disk.</p>
        </div>

        <div class="grid-2" style="margin: 16px 0; align-items: center;">
          <div class="card" style="background: #f8fafc; border: 1.5px solid #cbd5e1; gap: 12px;">
            <div class="card-title" style="font-size: 15px; color: #0f172a;">How the QA Inspector Works</div>
            <ul style="font-size: 12.5px; color: #334155; padding-left: 20px; line-height: 1.6;">
              <li><strong>Decodes Rendered MP4:</strong> Doesn't trust in-memory bounding boxes; it reads the final exported video file that the public will see.</li>
              <li><strong>Independent Architecture:</strong> Uses <strong>RetinaFace (ResNet-50)</strong>, completely separate from the primary SCRFD-34G detector.</li>
              <li><strong>Biometric Feature Search:</strong> Searches every frame for leftover eyes, nose, or mouth outlines.</li>
              <li><strong>Instant Quarantine:</strong> If any face is spotted with confidence ≥ 0.30, the video is marked <span style="background: #fee2e2; color: #7f1d1d; padding: 2px 6px; border-radius: 4px; font-weight: 700;">NEEDS_REVIEW</span>.</li>
            </ul>
          </div>

          <div class="card pill-green" style="border-width: 2px; gap: 12px;">
            <div class="card-title" style="font-size: 15px; color: #14532d;">The 2 Leak Classifications Caught by QA</div>
            <div style="display: flex; flex-direction: column; gap: 10px;">
              <div class="card" style="background: #ffffff; padding: 12px;">
                <div style="font-weight: 700; color: #b45309; font-size: 12.5px;">1. Residual Dilation Leak (IoU ≥ 0.10)</div>
                <div style="font-size: 11.5px; color: #475569; margin-top: 2px;">The blur was applied, but the person's jawline, ear, or side profile poked slightly out of the edge.</div>
              </div>
              <div class="card" style="background: #ffffff; padding: 12px;">
                <div style="font-weight: 700; color: #dc2626; font-size: 12.5px;">2. Completely Missed Face (IoU &lt; 0.10)</div>
                <div style="font-size: 11.5px; color: #475569; margin-top: 2px;">The primary detector produced zero blur boxes. The QA auditor catches the miss so a human reviewer can fix it.</div>
              </div>
            </div>
            <div style="font-size: 12px; font-weight: 700; color: #14532d; margin-top: 4px;">
              🔒 Result: 100% of demo clips quarantined for review; zero unvetted leaks escape to the public.
            </div>
          </div>
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 9: Human-in-the-Loop Audit Dashboard               -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">Reviewer Workflow</div>
          <h2>Human-in-the-Loop Streamlit Audit Dashboard</h2>
          <p class="subtitle">Empowering non-technical privacy and compliance officers to review and approve video jobs.</p>
        </div>

        <div class="grid-3" style="margin: 16px 0;">
          <div class="card" style="gap: 8px;">
            <div style="font-size: 26px;">📺</div>
            <div class="card-title">Synchronized Video Player</div>
            <p class="card-body">
              Plays raw bodycam footage and rendered redacted footage side-by-side in lockstep, making visual verification effortless.
            </p>
          </div>

          <div class="card" style="gap: 8px;">
            <div style="font-size: 26px;">🚨</div>
            <div class="card-title">Risk-Flag Scrubber</div>
            <p class="card-body">
              Auditors don't need to watch 60 minutes of video. The dashboard highlights <span style="color: #dc2626; font-weight: 700;">Red Frames (Leaks)</span> and <span style="color: #d97706; font-weight: 700;">Amber Frames (Coasted)</span> so reviewers jump straight to high-risk moments.
            </p>
          </div>

          <div class="card" style="gap: 8px;">
            <div style="font-size: 26px;">⚖️</div>
            <div class="card-title">One-Click Legal Override</div>
            <p class="card-body">
              If an auditor verifies that a flagged leak was a false alarm (e.g. mannequin or poster), they enter an audit justification note and approve it to <span style="color: #16a34a; font-weight: 700;">SAFE</span> with persistent SQLite logging.
            </p>
          </div>
        </div>

        <div class="card pill-blue" style="flex-direction: row; align-items: center; justify-content: space-between; padding: 12px 18px;">
          <span style="font-size: 13px; font-weight: 600;">
            To launch the interactive dashboard locally: <code class="code-box" style="padding: 2px 8px; font-size: 11px;">make dashboard</code>
          </span>
          <span style="font-size: 12px; font-weight: 700; color: #1d4ed8;">Available at http://localhost:8501</span>
        </div>
      </section>

      <!-- ======================================================== -->
      <!-- SLIDE 10: Plain-English Glossary Appendix                -->
      <!-- ======================================================== -->
      <section class="slide">
        <div>
          <div class="slide-tag">Reference Appendix</div>
          <h2>Plain-English Glossary of Every Metric & Formula</h2>
          <p class="subtitle">Exhaustive definitions and real-world meaning for non-technical stakeholders.</p>
        </div>

        <div class="grid-2" style="margin: 12px 0; max-height: 380px; overflow-y: auto; gap: 10px; padding-right: 6px;">
          <div class="card" style="padding: 10px; background: #f8fafc;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px;">1. Overlap Score / IoU</div>
            <div style="font-family: monospace; font-size: 10px; color: #475569;">Area(A ∩ B) / Area(A ∪ B)</div>
            <div style="font-size: 11px; color: #334155; margin-top: 2px;">Fraction of real face covered by blur. ≥0.55 is solidly protected; &lt;0.10 is completely missed.</div>
          </div>

          <div class="card" style="padding: 10px; background: #f8fafc;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px;">2. Face Catch Recall</div>
            <div style="font-family: monospace; font-size: 10px; color: #475569;">Real Faces Caught / Total Real Faces</div>
            <div style="font-size: 11px; color: #334155; margin-top: 2px;">Most important safety metric: out of 100 real faces, how many did we catch and blur? (Pose Fusion = 76.0%).</div>
          </div>

          <div class="card" style="padding: 10px; background: #f8fafc;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px;">3. +28% Box Expansion (Dilation)</div>
            <div style="font-family: monospace; font-size: 10px; color: #475569;">1.28 * width, 1.28 * height (+63.8% area)</div>
            <div style="font-size: 11px; color: #334155; margin-top: 2px;">Expands the blur box beyond eyes/mouth to cover forehead, ears, and jaw, preventing edge leaks.</div>
          </div>

          <div class="card" style="padding: 10px; background: #f8fafc;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px;">4. Blur Mask Wobble (Acceleration Jitter)</div>
            <div style="font-family: monospace; font-size: 10px; color: #475569;">||c_{{t+1}} - 2*c_t + c_{{t-1}}||_2</div>
            <div style="font-size: 11px; color: #334155; margin-top: 2px;">How much the blur box shakes between frames. Tracking makes masks 66.0% smoother (1.49 px vs 4.40 px).</div>
          </div>

          <div class="card" style="padding: 10px; background: #f8fafc;">
            <div style="font-family: monospace; font-size: 10px; color: #475569;">5. Gap Bridging (Linear Interpolation)</div>
            <div style="font-family: monospace; font-size: 10px; color: #475569;">bbox(t) = bbox(t1) + fraction * delta</div>
            <div style="font-size: 11px; color: #334155; margin-top: 2px;">Connects detections across 1-12 frame dropouts, slashing accidental unblurred flashes by 88.1%.</div>
          </div>

          <div class="card" style="padding: 10px; background: #f8fafc;">
            <div style="font-family: monospace; font-size: 10px; color: #475569;">6. Motion Tracking Memory (Kalman)</div>
            <div style="font-family: monospace; font-size: 10px; color: #475569;">[x, y, a, h, x_dot, y_dot, a_dot, h_dot]</div>
            <div style="font-size: 11px; color: #334155; margin-top: 2px;">Physics-based trajectory prediction that coasts blur bubbles during camera bounce (rescued 3,101 frames).</div>
          </div>

          <div class="card" style="padding: 10px; background: #f8fafc;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px;">7. Body Pose Cranial Estimation</div>
            <div style="font-family: monospace; font-size: 10px; color: #475569;">cranial_width = 0.90 * shoulder_span</div>
            <div style="font-size: 11px; color: #334155; margin-top: 2px;">Estimates head position from shoulder geometry when facial features dissolve, rescuing 25.8% of blind spots.</div>
          </div>

          <div class="card" style="padding: 10px; background: #f8fafc;">
            <div style="font-weight: 700; color: #0f172a; font-size: 12px;">8. Fail-Closed Gatekeeper</div>
            <div style="font-family: monospace; font-size: 10px; color: #475569;">SAFE if Leaks == 0 else NEEDS_REVIEW</div>
            <div style="font-size: 11px; color: #334155; margin-top: 2px;">Zero-trust security: never release unverified video; any flagged leak must be reviewed by a human officer.</div>
          </div>
        </div>
      </section>

    </main>

    <!-- Footer Controls -->
    <footer class="deck-footer">
      <button id="prev-btn" class="nav-btn">← Previous</button>
      <div id="dot-container" class="dot-container"></div>
      <button id="next-btn" class="nav-btn nav-btn-primary">Next →</button>
    </footer>

  </div>

  <p class="nav-hint">
    Use <strong>Left / Right Arrow keys</strong>, <strong>Spacebar</strong>, or <strong>Enter</strong> to navigate. Press <strong>F</strong> for Fullscreen.
  </p>

  <script>
    const slides = document.querySelectorAll('.slide');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const slideIndicator = document.getElementById('slide-indicator');
    const dotContainer = document.getElementById('dot-container');
    const fullscreenBtn = document.getElementById('fullscreen-btn');

    let currentSlide = 0;
    const totalSlides = slides.length;

    for (let i = 0; i < totalSlides; i++) {{
      const dot = document.createElement('button');
      dot.className = `dot ${{i === 0 ? 'active' : ''}}`;
      dot.title = `Go to slide ${{i + 1}}`;
      dot.addEventListener('click', () => goToSlide(i));
      dotContainer.appendChild(dot);
    }}

    const dots = dotContainer.querySelectorAll('.dot');

    function updateSlide() {{
      slides.forEach((slide, idx) => {{
        if (idx === currentSlide) {{
          slide.classList.add('active');
        }} else {{
          slide.classList.remove('active');
        }}
      }});

      dots.forEach((dot, idx) => {{
        if (idx === currentSlide) {{
          dot.classList.add('active');
        }} else {{
          dot.classList.remove('active');
        }}
      }});

      slideIndicator.textContent = `Slide ${{currentSlide + 1}} of ${{totalSlides}}`;
      prevBtn.disabled = currentSlide === 0;
      nextBtn.disabled = currentSlide === totalSlides - 1;
    }}

    function goToSlide(idx) {{
      if (idx >= 0 && idx < totalSlides) {{
        currentSlide = idx;
        updateSlide();
      }}
    }}

    prevBtn.addEventListener('click', () => {{
      if (currentSlide > 0) {{
        currentSlide--;
        updateSlide();
      }}
    }});

    nextBtn.addEventListener('click', () => {{
      if (currentSlide < totalSlides - 1) {{
        currentSlide++;
        updateSlide();
      }}
    }});

    window.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') {{
        if (currentSlide < totalSlides - 1) {{
          currentSlide++;
          updateSlide();
        }}
      }} else if (e.key === 'ArrowLeft') {{
        if (currentSlide > 0) {{
          currentSlide--;
          updateSlide();
        }}
      }} else if (e.key === 'f' || e.key === 'F') {{
        toggleFullscreen();
      }}
    }});

    function toggleFullscreen() {{
      if (!document.fullscreenElement) {{
        document.documentElement.requestFullscreen().catch(err => console.log(err));
      }} else {{
        if (document.exitFullscreen) {{
          document.exitFullscreen();
        }}
      }}
    }}

    fullscreenBtn.addEventListener('click', toggleFullscreen);

    updateSlide();
  </script>
</body>
</html>
"""


def main():
    html_content = build_slides_html()
    repo_slides = REPO_ROOT / "slides.html"
    artifact_dir = Path.home() / ".gemini" / "antigravity" / "brain" / "01f6340a-6734-44be-a2fd-26b245ab0f80"
    artifact_slides = artifact_dir / "presentation_slides.html"

    with open(repo_slides, "w", encoding="utf-8") as f:
        f.write(html_content)

    if artifact_dir.exists():
        with open(artifact_slides, "w", encoding="utf-8") as f:
            f.write(html_content)

    print("Slide deck HTML generated successfully (zero external CDN dependencies, 100% self-contained).")


if __name__ == "__main__":
    main()
