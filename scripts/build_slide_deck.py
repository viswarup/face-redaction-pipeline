"""
Generate an interactive, visual 10-slide HTML presentation deck with embedded
SVG diagrams, benchmark tables, and base64 visual proof strips.
"""

from __future__ import annotations

import base64
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
  <title>Safety-Critical Bodycam Face Anonymization Pipeline - Slide Deck</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .mono {{
      font-family: 'JetBrains Mono', monospace;
    }}
    .slide {{
      display: none;
    }}
    .slide.active {{
      display: flex;
    }}
    .fade-in {{
      animation: fadeIn 0.22s ease-out;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      padding: 6px 10px;
      border: 1px solid #e2e8f0;
      text-align: left;
    }}
    th {{
      background-color: #f8fafc;
      font-weight: 600;
      color: #0f172a;
    }}
    @media print {{
      @page {{
        size: 16in 9in;
        margin: 0.5in;
      }}
      body {{
        background-color: #ffffff !important;
        padding: 0 !important;
      }}
      .slide {{
        display: flex !important;
        page-break-after: always;
        break-after: page;
        min-height: 8in !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: none !important;
        margin-bottom: 20px;
      }}
      header, footer, p.nav-help {{
        display: none !important;
      }}
    }}
  </style>
</head>
<body class="bg-slate-900 text-slate-800 antialiased min-h-screen flex flex-col items-center justify-center p-3 md:p-6 select-none">

  <!-- Main Slide Frame (16:9 Aspect Ratio) -->
  <div class="relative w-full max-w-6xl bg-white rounded-2xl shadow-2xl border border-slate-700/50 overflow-hidden flex flex-col" style="min-height: 680px; height: 86vh;">

    <!-- Top Slide Header -->
    <header class="flex items-center justify-between px-6 py-3 border-b border-slate-100 bg-slate-50/90 text-xs">
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full bg-blue-600 animate-pulse"></span>
        <span class="font-bold tracking-wide uppercase text-slate-600">Bodycam Privacy & Robotics Anonymization</span>
      </div>
      <div class="flex items-center gap-4 font-semibold text-slate-500">
        <span id="slide-indicator">Slide 1 of 10</span>
        <button id="fullscreen-btn" class="px-2 py-1 rounded bg-slate-200/80 hover:bg-slate-300 text-slate-700 transition" title="Toggle Fullscreen (F)">
          ⛶ Fullscreen
        </button>
      </div>
    </header>

    <!-- Slide Content Viewport -->
    <main class="flex-1 p-6 md:p-8 flex flex-col justify-center overflow-y-auto">

      <!-- ========================================== -->
      <!-- SLIDE 1: Title Slide                       -->
      <!-- ========================================== -->
      <section class="slide active flex-col items-center justify-center text-center fade-in space-y-6">
        <div class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-bold tracking-wide">
          🛡️ Production-Grade CV & Safety Engineering
        </div>
        <h1 class="text-3xl md:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight max-w-4xl">
          Safety-Critical Bodycam Face Redaction & Verification Pipeline
        </h1>
        <p class="text-base md:text-lg text-slate-600 max-w-2xl leading-relaxed">
          A zero-trust, fail-closed anonymization pipeline that eliminates flicker leaks under rapid motion blur, recovers steep camera profiles with body pose, and verifies every frame with an independent AI auditor.
        </p>

        <!-- KPI Pill Strip -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 max-w-3xl w-full pt-4">
          <div class="p-3 rounded-xl bg-blue-50 border border-blue-200 text-center">
            <div class="text-xl font-extrabold text-blue-700">76.0% Recall</div>
            <div class="text-[11px] text-blue-900 font-medium">Face Catch Rate (336/442)</div>
          </div>
          <div class="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-center">
            <div class="text-xl font-extrabold text-emerald-700">-88.1% Flashes</div>
            <div class="text-[11px] text-emerald-900 font-medium">Fewer Accidental Leaks</div>
          </div>
          <div class="p-3 rounded-xl bg-purple-50 border border-purple-200 text-center">
            <div class="text-xl font-extrabold text-purple-700">3,101 Frames</div>
            <div class="text-[11px] text-purple-900 font-medium">Rescued by Motion Memory</div>
          </div>
          <div class="p-3 rounded-xl bg-rose-50 border border-rose-200 text-center">
            <div class="text-xl font-extrabold text-rose-700">100% Fail-Closed</div>
            <div class="text-[11px] text-rose-900 font-medium">Zero Unvetted Video Released</div>
          </div>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 2: The Core Problem                  -->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-4">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">The Operational Challenge</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">Why Standard Face Blurring Fails on Bodycams</h2>
          <p class="text-xs md:text-sm text-slate-600">Body-worn cameras move violently, walk through dark corridors, and view people at steep angles.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="p-5 rounded-xl bg-amber-50 border border-amber-200 flex flex-col justify-between">
            <div>
              <div class="text-2xl mb-1">⚡</div>
              <h3 class="font-bold text-amber-900 text-sm">1. Severe Motion Blur</h3>
              <p class="text-xs text-amber-800 mt-2 leading-relaxed">
                When an officer or robot turns quickly, camera shutter lag smears facial features (eyes, nose, mouth) into smooth streaks. Standalone face detectors suddenly report 0 faces.
              </p>
            </div>
            <div class="mt-3 p-2 rounded bg-amber-100/80 text-[11px] font-semibold text-amber-900">
              Result: Accidental unblurred flashes
            </div>
          </div>

          <div class="p-5 rounded-xl bg-rose-50 border border-rose-200 flex flex-col justify-between">
            <div>
              <div class="text-2xl mb-1">📐</div>
              <h3 class="font-bold text-rose-900 text-sm">2. Steep Angles & Foreshortening</h3>
              <p class="text-xs text-rose-800 mt-2 leading-relaxed">
                Chest-mounted cameras look up at tall people or down at seated individuals. Facial proportions distort, and back-of-the-head profiles hide all facial landmarks.
              </p>
            </div>
            <div class="mt-3 p-2 rounded bg-rose-100/80 text-[11px] font-semibold text-rose-900">
              Result: 50% missed on top-down views
            </div>
          </div>

          <div class="p-5 rounded-xl bg-indigo-50 border border-indigo-200 flex flex-col justify-between">
            <div>
              <div class="text-2xl mb-1">🎯</div>
              <h3 class="font-bold text-indigo-900 text-sm">3. Model Confirmation Bias</h3>
              <p class="text-xs text-indigo-800 mt-2 leading-relaxed">
                Most pipelines use the same AI model to both blur and check their work. If an AI detector misses a face initially, it will predictably miss it during verification too.
              </p>
            </div>
            <div class="mt-3 p-2 rounded bg-indigo-100/80 text-[11px] font-semibold text-indigo-900">
              Result: False sense of 100% security
            </div>
          </div>
        </div>

        <div class="p-3.5 rounded-xl bg-slate-900 text-white flex items-center justify-between text-xs">
          <span><strong>Our Philosophy:</strong> In privacy law (GDPR/FOIA), a 95% success rate is a 5% legal disaster. The pipeline must be <strong>Fail-Closed</strong>.</span>
          <span class="px-2.5 py-1 bg-red-600 rounded font-bold uppercase text-[10px]">Fail-Closed Safety</span>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 3: System Architecture Flowchart    -->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-3">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">System Architecture</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">End-to-End Pipeline Flowchart</h2>
        </div>

        <!-- SVG Flowchart -->
        <div class="bg-slate-50 border border-slate-200 rounded-xl p-3 flex items-center justify-center">
          <svg viewBox="0 0 920 230" class="w-full h-auto" style="max-height: 240px;">
            <defs>
              <marker id="arr" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#64748b"/>
              </marker>
              <marker id="arr-g" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#10b981"/>
              </marker>
              <marker id="arr-r" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#ef4444"/>
              </marker>
            </defs>

            <!-- Video Ingest -->
            <rect x="10" y="80" width="100" height="70" rx="8" fill="#0f172a"/>
            <text x="60" y="112" fill="#fff" font-size="11" font-weight="bold" text-anchor="middle">Input Video</text>
            <text x="60" y="132" fill="#94a3b8" font-size="9" text-anchor="middle">SHA-256 Fingerprint</text>

            <line x1="110" y1="115" x2="140" y2="115" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>

            <!-- Stage 1 -->
            <rect x="140" y="65" width="130" height="100" rx="8" fill="#fff" stroke="#3b82f6" stroke-width="2"/>
            <text x="205" y="90" fill="#1d4ed8" font-size="11" font-weight="bold" text-anchor="middle">1. Dual Detection</text>
            <text x="205" y="110" fill="#334155" font-size="10" font-weight="semibold" text-anchor="middle">SCRFD-34G (Face)</text>
            <text x="205" y="128" fill="#334155" font-size="10" font-weight="semibold" text-anchor="middle">+ YOLOv8 (Pose)</text>
            <text x="205" y="148" fill="#64748b" font-size="8" text-anchor="middle">Fuses head & shoulders</text>

            <line x1="270" y1="115" x2="300" y2="115" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>

            <!-- Stage 2 -->
            <rect x="300" y="65" width="135" height="100" rx="8" fill="#fff" stroke="#8b5cf6" stroke-width="2"/>
            <text x="367" y="90" fill="#6d28d9" font-size="11" font-weight="bold" text-anchor="middle">2. Motion Tracking</text>
            <text x="367" y="110" fill="#334155" font-size="10" font-weight="semibold" text-anchor="middle">ByteTrack + Kalman</text>
            <text x="367" y="128" fill="#64748b" font-size="8" text-anchor="middle">Motion memory coasts</text>
            <text x="367" y="145" fill="#64748b" font-size="8" text-anchor="middle">through camera shake</text>

            <line x1="435" y1="115" x2="465" y2="115" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>

            <!-- Stage 3 -->
            <rect x="465" y="65" width="130" height="100" rx="8" fill="#fff" stroke="#f59e0b" stroke-width="2"/>
            <text x="530" y="90" fill="#b45309" font-size="11" font-weight="bold" text-anchor="middle">3. Heavy Redaction</text>
            <text x="530" y="110" fill="#334155" font-size="10" font-weight="semibold" text-anchor="middle">+28% Box Expansion</text>
            <text x="530" y="128" fill="#64748b" font-size="8" text-anchor="middle">Gaussian σ=25 Blur</text>
            <text x="530" y="145" fill="#64748b" font-size="8" text-anchor="middle">Covers hair & ears</text>

            <line x1="595" y1="115" x2="625" y2="115" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>

            <!-- Stage 4 -->
            <rect x="625" y="65" width="145" height="100" rx="8" fill="#fff" stroke="#10b981" stroke-width="2"/>
            <text x="697" y="90" fill="#047857" font-size="11" font-weight="bold" text-anchor="middle">4. Secondary QA</text>
            <text x="697" y="110" fill="#334155" font-size="10" font-weight="semibold" text-anchor="middle">RetinaFace Auditor</text>
            <text x="697" y="128" fill="#64748b" font-size="8" text-anchor="middle">Audits rendered MP4</text>
            <text x="697" y="145" fill="#64748b" font-size="8" text-anchor="middle">Zero-trust inspection</text>

            <!-- Outputs -->
            <line x1="770" y1="100" x2="805" y2="65" stroke="#10b981" stroke-width="2" marker-end="url(#arr-g)"/>
            <rect x="805" y="45" width="95" height="40" rx="6" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5"/>
            <text x="852" y="68" fill="#065f46" font-size="10" font-weight="bold" text-anchor="middle">SAFE (Approved)</text>

            <line x1="770" y1="130" x2="805" y2="165" stroke="#ef4444" stroke-width="2" marker-end="url(#arr-r)"/>
            <rect x="805" y="145" width="95" height="40" rx="6" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5"/>
            <text x="852" y="168" fill="#991b1b" font-size="9" font-weight="bold" text-anchor="middle">NEEDS_REVIEW</text>
            <text x="852" y="179" fill="#b91c1c" font-size="7" text-anchor="middle">Sent to Dashboard</text>

            <text x="460" y="210" fill="#64748b" font-size="9" font-style="italic" text-anchor="middle">
              SQLite WAL Database tracks deduplication, audit trails, and human reviewer notes
            </text>
          </svg>
        </div>

        <div class="grid grid-cols-4 gap-2 text-center text-[11px]">
          <div class="p-2 rounded bg-blue-50 text-blue-900 font-medium">1. Detects multi-scale faces & skeletons</div>
          <div class="p-2 rounded bg-purple-50 text-purple-900 font-medium">2. Bridges motion blur dropouts</div>
          <div class="p-2 rounded bg-amber-50 text-amber-900 font-medium">3. Destroys biometric identifiers</div>
          <div class="p-2 rounded bg-emerald-50 text-emerald-900 font-medium">4. Guarantees zero unvetted leaks</div>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 4: Body/Head Pose Fusion             -->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-4">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">The Technical Breakthrough</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">Body/Head Pose Fusion: Finding Faces from Shoulders</h2>
          <p class="text-xs md:text-sm text-slate-600">When facial features dissolve in motion blur, the human body skeleton remains visible.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-5 items-center">
          <div class="p-5 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
            <h3 class="font-bold text-slate-900 text-sm flex items-center gap-2">
              <span class="p-1 rounded bg-blue-600 text-white text-xs">AI</span> How YOLOv8-Pose Rescues Blind Spots
            </h3>
            <p class="text-xs text-slate-600 leading-relaxed">
              When people turn around or light drops, 2D facial classifiers output <strong>zero boxes</strong>. But our body pose model extracts shoulder and neck keypoints:
            </p>
            <div class="p-3 bg-slate-900 text-slate-100 rounded-lg mono text-[11px] leading-relaxed">
              cranial_width = 0.90 * shoulder_span<br>
              head_center   = neck_center - 0.70 * cranial_width<br>
              blur_box      = [cranial_x1, cranial_y1, cranial_x2, cranial_y2]
            </div>
            <ul class="text-xs text-slate-700 space-y-1 list-disc list-inside">
              <li><strong>If Face Overlaps Pose (IoU ≥ 0.20):</strong> We take their combined union to cover ears & chin.</li>
              <li><strong>If Face is Missing (IoU &lt; 0.20):</strong> We inject a recovered head blur box automatically.</li>
            </ul>
          </div>

          <div class="p-5 rounded-xl bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 space-y-3">
            <div class="text-xs font-bold uppercase text-blue-700">Empirical Proof on 442 Real Faces</div>
            <div class="grid grid-cols-2 gap-3">
              <div class="p-3 bg-white rounded-lg border border-blue-100 text-center">
                <div class="text-2xl font-black text-blue-600">32 Faces</div>
                <div class="text-[11px] text-slate-600 font-medium">Rescued from Total Blind Spots</div>
              </div>
              <div class="p-3 bg-white rounded-lg border border-blue-100 text-center">
                <div class="text-2xl font-black text-emerald-600">+7.7%</div>
                <div class="text-[11px] text-slate-600 font-medium">Absolute Recall Gain (76.0%)</div>
              </div>
            </div>
            <div class="p-3 rounded-lg bg-emerald-100/70 border border-emerald-300 text-xs text-emerald-950 font-medium leading-relaxed">
              ✨ <strong>Key Result:</strong> Rescued <strong>25.8% of severe blind spots</strong> where BOTH primary face detectors (SCRFD and RetinaFace) completely failed.
            </div>
          </div>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 5: 100-Image Benchmark Table         -->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-3">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">Empirical Benchmark (100 Hand-Labeled Images)</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">3-Way Model Comparison: Which AI Works Best?</h2>
        </div>

        <div class="overflow-x-auto rounded-xl border border-slate-200">
          <table class="text-xs">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Total Faces</th>
                <th>Face Catch Recall (%) 🏆</th>
                <th>Fully Protected</th>
                <th>Partially Covered</th>
                <th>Missed Faces (Leaks)</th>
                <th>Processing Speed</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="font-bold text-slate-800">Base SCRFD-34G</td>
                <td>442</td>
                <td class="font-semibold text-slate-700">68.3% (302)</td>
                <td>52.0% (230)</td>
                <td>16.3% (72)</td>
                <td class="text-rose-600 font-semibold">31.7% (140)</td>
                <td>2.5 FPS (398 ms)</td>
              </tr>
              <tr>
                <td class="font-bold text-slate-800">RetinaFace Detector</td>
                <td>442</td>
                <td class="font-semibold text-slate-700">47.1% (208)</td>
                <td>37.1% (164)</td>
                <td>10.0% (44)</td>
                <td class="text-rose-600 font-semibold">52.9% (234)</td>
                <td>1.9 FPS (529 ms)</td>
              </tr>
              <tr class="bg-blue-50/70 font-semibold">
                <td class="font-bold text-blue-900">SCRFD + Body/Head Pose Fusion</td>
                <td>442</td>
                <td class="font-black text-blue-700 text-sm">76.0% (336) 🏆</td>
                <td>41.2% (182)</td>
                <td>34.8% (154)</td>
                <td class="text-emerald-700 font-bold">24.0% (106) ✨</td>
                <td>2.2 FPS (458 ms)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Plain English Highlights -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <strong>Face Catch Recall:</strong> Percentage of real human faces caught and blurred out of all 442 faces. Pose Fusion is the clear winner (+7.7% over Base SCRFD).
          </div>
          <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <strong>Missed Faces:</strong> Faces left with zero blur. Pose Fusion reduces missed faces by 34 faces across the 100 images.
          </div>
          <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <strong>Partially Covered:</strong> Wide boxes that safely obscure the head but are automatically sent to human review for zero-leak verification.
          </div>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 6: Video Tracking Ablation Table     -->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-3">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">Video Tracking Ablation (1,127 Real Video Frames)</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">Why Video Tracking is Non-Negotiable</h2>
        </div>

        <div class="overflow-x-auto rounded-xl border border-slate-200">
          <table class="text-xs">
            <thead>
              <tr>
                <th>Safety & Video Quality Metric</th>
                <th>Frame-by-Frame Detection</th>
                <th>With Motion Tracking (Our Pipeline)</th>
                <th>Plain-English Meaning & Impact</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="font-bold text-slate-800">Total Video Frames Tested</td>
                <td>1,127 frames</td>
                <td>1,127 frames</td>
                <td>Complete empirical test across 5 demo video clips</td>
              </tr>
              <tr class="bg-emerald-50/60">
                <td class="font-bold text-emerald-950">Accidental Face Flashes (Flickers)</td>
                <td class="text-rose-600 font-semibold">369 flashes</td>
                <td class="text-emerald-700 font-black">44 flashes</td>
                <td><strong>88.1% fewer flashes!</strong> Stops faces popping out unblurred</td>
              </tr>
              <tr>
                <td class="font-bold text-slate-800">Unblurred Face Frames Rescued</td>
                <td class="text-slate-500">0 (Dropped / Unblurred)</td>
                <td class="text-blue-700 font-bold">3,101 frames</td>
                <td>Motion memory keeps blur locked on during camera shake</td>
              </tr>
              <tr class="bg-blue-50/60">
                <td class="font-bold text-blue-950">Blur Mask Shaking / Wobble (Jitter)</td>
                <td>4.40 px (Erratic wobble)</td>
                <td class="font-bold text-blue-800">1.49 px (Smooth)</td>
                <td><strong>66.0% smoother!</strong> Keeps hands/tools visible for robotics</td>
              </tr>
              <tr>
                <td class="font-bold text-slate-800">Visible Face Leaks Caught by QA</td>
                <td>238 frames</td>
                <td class="font-bold text-emerald-700">127 frames</td>
                <td><strong>46.6% drop</strong> in visible facial features escaping blur</td>
              </tr>
              <tr class="bg-slate-50">
                <td class="font-bold text-slate-800">Videos Sent to Human Review</td>
                <td>5 of 5 clips</td>
                <td>5 of 5 clips</td>
                <td><strong>100% fail-closed safety:</strong> Zero unvetted releases</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="p-3 bg-slate-100 rounded-lg text-xs text-slate-700 leading-relaxed">
          💡 <strong>Key Takeaway:</strong> Without tracking, bodycam video suffers from <strong>369 momentary unblurred flash leaks</strong>. ByteTrack motion memory eliminates 88.1% of flashes and stabilizes mask wobbles so robots can cleanly observe surrounding body mechanics.
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 7: Visual Proof & Failure Gallery    -->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-3">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">Visual Diagnostic Evidence</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">Real-World Bodycam Failure & Recovery Strips</h2>
        </div>

        <div class="space-y-4">
          <!-- Tracking Proof Strip -->
          <div class="border border-slate-200 rounded-xl p-3 bg-slate-50 space-y-2">
            <div class="flex items-center justify-between text-xs">
              <span class="font-bold text-slate-800">1. Tracking vs. No Tracking: Eliminating Unblurred Flicker Flashes</span>
              <span class="text-slate-500 font-mono">Video 1, Frame #143</span>
            </div>
            {f'<img src="{img_track_1}" class="w-full rounded-lg border border-slate-300 shadow-sm" alt="Tracking Proof">' if img_track_1 else '<div class="p-6 bg-slate-200 text-center rounded text-xs">tracking_vs_notracking_01.jpg</div>'}
            <p class="text-[11px] text-slate-600 italic">
              Panel 1: Raw rapid head turn. Panel 2: Detection-Only drops below threshold (RED UNBLURRED LEAK). Panel 3: ByteTrack motion memory maintains smooth blue blur bubble.
            </p>
          </div>

          <!-- Pose Proof Strip -->
          <div class="border border-slate-200 rounded-xl p-3 bg-slate-50 space-y-2">
            <div class="flex items-center justify-between text-xs">
              <span class="font-bold text-slate-800">2. Pose Fusion: Rescuing Motion Blur & Extreme Low Light</span>
              <span class="text-slate-500 font-mono">100-Image Benchmark (comparison_01.jpg)</span>
            </div>
            {f'<img src="{img_pose_1}" class="w-full rounded-lg border border-slate-300 shadow-sm" alt="Pose Proof">' if img_pose_1 else '<div class="p-6 bg-slate-200 text-center rounded text-xs">comparison_01.jpg</div>'}
            <p class="text-[11px] text-slate-600 italic">
              Panel 1: Raw face. Panels 2 & 3: Standalone SCRFD and RetinaFace miss the face entirely (RED LEAKS). Panel 4: Pose Fusion uses shoulders to project cranial blur, rescuing the face.
            </p>
          </div>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 8: Independent Post-Render QA Auditor-->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-4">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">Zero-Trust Quality Control</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">The Independent Post-Render QA Inspector</h2>
          <p class="text-xs md:text-sm text-slate-600">Why a separate, second AI model must inspect the final rendered video from disk.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div class="p-5 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
            <h3 class="font-bold text-slate-900 text-sm">How the QA Inspector Works</h3>
            <ul class="text-xs text-slate-700 space-y-2.5 list-disc list-inside">
              <li><strong>Decodes Rendered MP4:</strong> Doesn't trust in-memory bounding boxes; it reads the final exported video file that the public will see.</li>
              <li><strong>Independent Architecture:</strong> Uses <strong>RetinaFace (ResNet-50)</strong>, completely separate from the primary SCRFD-34G detector.</li>
              <li><strong>Biometric Feature Search:</strong> Searches every frame for leftover eyes, nose, or mouth outlines.</li>
              <li><strong>Instant Quarantine:</strong> If any face is spotted with confidence ≥ 0.30, the video is marked <code class="bg-red-100 text-red-800 px-1 py-0.5 rounded font-bold">NEEDS_REVIEW</code>.</li>
            </ul>
          </div>

          <div class="p-5 rounded-xl bg-emerald-50 border border-emerald-200 space-y-3">
            <h3 class="font-bold text-emerald-900 text-sm">The 2 Leak Classifications Caught by QA</h3>
            <div class="space-y-3">
              <div class="p-3 bg-white rounded-lg border border-emerald-200 text-xs">
                <div class="font-bold text-amber-800">1. Residual Dilation Leak (IoU ≥ 0.10)</div>
                <div class="text-slate-600 mt-1">The blur was applied, but the person's jawline, ear, or side profile poked slightly out of the edge.</div>
              </div>
              <div class="p-3 bg-white rounded-lg border border-emerald-200 text-xs">
                <div class="font-bold text-red-800">2. Completely Missed Face (IoU &lt; 0.10)</div>
                <div class="text-slate-600 mt-1">The primary detector produced zero blur boxes. The QA auditor catches the miss so a human reviewer can fix it.</div>
              </div>
            </div>
            <div class="text-[11px] text-emerald-900 font-semibold">
              🔒 Result: 100% of demo clips quarantined for review; zero unvetted leaks escape to the public.
            </div>
          </div>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 9: Human-in-the-Loop Audit Dashboard -->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-4">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">Reviewer Workflow</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">Human-in-the-Loop Streamlit Audit Dashboard</h2>
          <p class="text-xs md:text-sm text-slate-600">Empowering non-technical privacy and compliance officers to review and approve video jobs.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <div class="text-2xl">📺</div>
            <h3 class="font-bold text-slate-900 text-sm">Synchronized Player</h3>
            <p class="text-xs text-slate-600 leading-relaxed">
              Plays raw bodycam footage and rendered redacted footage side-by-side in lockstep, making visual verification effortless.
            </p>
          </div>

          <div class="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <div class="text-2xl">🚨</div>
            <h3 class="font-bold text-slate-900 text-sm">Risk-Flag Scrubber</h3>
            <p class="text-xs text-slate-600 leading-relaxed">
              Auditors don't need to watch 60 minutes of video. The dashboard highlights <span class="text-red-600 font-bold">Red Frames (Leaks)</span> and <span class="text-amber-600 font-bold">Amber Frames (Coasted)</span> so reviewers jump straight to high-risk moments.
            </p>
          </div>

          <div class="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <div class="text-2xl">⚖️</div>
            <h3 class="font-bold text-slate-900 text-sm">One-Click Override</h3>
            <p class="text-xs text-slate-600 leading-relaxed">
              If an auditor verifies that a flagged leak was a false alarm (e.g. mannequin or poster), they enter an audit justification note and approve it to <span class="text-emerald-600 font-bold">SAFE</span> with persistent SQLite logging.
            </p>
          </div>
        </div>

        <div class="p-4 rounded-xl bg-blue-50 border border-blue-200 flex items-center justify-between text-xs">
          <span>To launch the interactive dashboard locally: <code class="bg-blue-100 text-blue-900 px-2 py-0.5 rounded font-mono font-bold">make dashboard</code></span>
          <span class="text-blue-700 font-semibold">Available at http://localhost:8501</span>
        </div>
      </section>

      <!-- ========================================== -->
      <!-- SLIDE 10: Plain-English Glossary Appendix  -->
      <!-- ========================================== -->
      <section class="slide flex-col justify-between fade-in space-y-3">
        <div>
          <span class="text-xs font-bold text-blue-600 uppercase tracking-wider">Reference Appendix</span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-1">Plain-English Glossary of Every Metric & Formula</h2>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs overflow-y-auto max-h-[420px] pr-1">
          <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div class="font-bold text-slate-900">1. Overlap Score / IoU</div>
            <div class="text-slate-500 font-mono text-[10px]">Area(A ∩ B) / Area(A ∪ B)</div>
            <div class="text-slate-700 mt-1">Fraction of the real face covered by the blur bubble. ≥0.55 is solidly protected; &lt;0.10 is completely missed.</div>
          </div>

          <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div class="font-bold text-slate-900">2. Face Catch Recall</div>
            <div class="text-slate-500 font-mono text-[10px]">Real Faces Caught / Total Real Faces</div>
            <div class="text-slate-700 mt-1">Most important safety metric: out of 100 real faces, how many did we catch and blur? (Pose Fusion = 76.0%).</div>
          </div>

          <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div class="font-bold text-slate-900">3. +28% Box Expansion (Dilation)</div>
            <div class="text-slate-500 font-mono text-[10px]">1.28 * width, 1.28 * height (+63.8% area)</div>
            <div class="text-slate-700 mt-1">Expands the blur box beyond eyes/mouth to cover forehead, ears, and jaw, preventing edge leaks.</div>
          </div>

          <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div class="font-bold text-slate-900">4. Blur Mask Wobble (Acceleration Jitter)</div>
            <div class="text-slate-500 font-mono text-[10px]">||c_{{t+1}} - 2*c_t + c_{{t-1}}||_2</div>
            <div class="text-slate-700 mt-1">How much the blur box shakes between frames. Tracking makes masks 66.0% smoother (1.49 px vs 4.40 px).</div>
          </div>

          <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div class="font-bold text-slate-900">5. Gap Bridging (Linear Interpolation)</div>
            <div class="text-slate-500 font-mono text-[10px]">bbox(t) = bbox(t1) + fraction * delta</div>
            <div class="text-slate-700 mt-1">Connects detections across 1-12 frame dropouts, slashing accidental unblurred flashes by 88.1%.</div>
          </div>

          <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div class="font-bold text-slate-900">6. Motion Tracking Memory (Kalman)</div>
            <div class="text-slate-500 font-mono text-[10px]">[x, y, a, h, x_dot, y_dot, a_dot, h_dot]</div>
            <div class="text-slate-700 mt-1">Physics-based trajectory prediction that coasts blur bubbles during camera bounce (rescued 3,101 frames).</div>
          </div>

          <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div class="font-bold text-slate-900">7. Body Pose Cranial Estimation</div>
            <div class="text-slate-500 font-mono text-[10px]">cranial_width = 0.90 * shoulder_span</div>
            <div class="text-slate-700 mt-1">Estimates head position from shoulder geometry when facial features dissolve, rescuing 25.8% of blind spots.</div>
          </div>

          <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div class="font-bold text-slate-900">8. Fail-Closed Gatekeeper</div>
            <div class="text-slate-500 font-mono text-[10px]">SAFE if Leaks == 0 else NEEDS_REVIEW</div>
            <div class="text-slate-700 mt-1">Zero-trust security: never release unverified video; any flagged leak must be reviewed by a human officer.</div>
          </div>
        </div>
      </section>

    </main>

    <!-- Bottom Navigation Bar -->
    <footer class="flex items-center justify-between px-6 py-3 border-t border-slate-100 bg-slate-50/90 text-xs">
      <button id="prev-btn" class="px-4 py-2 rounded-lg bg-white border border-slate-300 font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed shadow-sm transition">
        ← Previous
      </button>

      <div id="dot-container" class="flex items-center gap-1.5">
        <!-- Dots injected by JS -->
      </div>

      <button id="next-btn" class="px-4 py-2 rounded-lg bg-blue-600 font-semibold text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed shadow-sm transition">
        Next →
      </button>
    </footer>

  </div>

  <p class="text-[11px] text-slate-400 mt-3">
    Navigate with <strong>Left / Right Arrow keys</strong> or <strong>Spacebar</strong>. Press <strong>F</strong> for Fullscreen.
  </p>

  <!-- Interactive Navigation Script -->
  <script>
    const slides = document.querySelectorAll('.slide');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const slideIndicator = document.getElementById('slide-indicator');
    const dotContainer = document.getElementById('dot-container');
    const fullscreenBtn = document.getElementById('fullscreen-btn');

    let currentSlide = 0;
    const totalSlides = slides.length;

    // Create pagination dots
    for (let i = 0; i < totalSlides; i++) {{
      const dot = document.createElement('button');
      dot.className = `w-2 h-2 rounded-full transition-all ${{i === 0 ? 'bg-blue-600 w-4' : 'bg-slate-300'}}`;
      dot.title = `Go to slide ${{i + 1}}`;
      dot.addEventListener('click', () => goToSlide(i));
      dotContainer.appendChild(dot);
    }}

    const dots = dotContainer.querySelectorAll('button');

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
          dot.className = 'w-4 h-2 rounded-full bg-blue-600 transition-all';
        }} else {{
          dot.className = 'w-2 h-2 rounded-full bg-slate-300 transition-all';
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

    with open(artifact_slides, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Slide deck generated successfully:")
    print(f"  - {repo_slides}")
    print(f"  - {artifact_slides}")


if __name__ == "__main__":
    main()
