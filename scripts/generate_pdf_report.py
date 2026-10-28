"""
Generate an executive-grade PDF version of REPORT.md using Pandoc + Headless Chrome.
Embeds all images as base64 data URIs for self-contained, publication-ready rendering.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_base64_image(image_path: Path) -> str:
    """Reads an image and returns a base64 data URI."""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def generate_pdf():
    report_md = REPO_ROOT / "REPORT.md"
    temp_html = REPO_ROOT / "data" / "outputs" / "report_temp.html"
    output_pdf = REPO_ROOT / "data" / "outputs" / "Safety_Critical_Face_Redaction_Pipeline_Report.pdf"
    root_pdf = REPO_ROOT / "Safety_Critical_Face_Redaction_Pipeline_Report.pdf"

    if not report_md.exists():
        print(f"Error: {report_md} not found!")
        sys.exit(1)

    print("Step 1: Converting REPORT.md to HTML via Pandoc...")
    pandoc_bin = shutil.which("pandoc") or "/opt/homebrew/bin/pandoc"
    cmd_pandoc = [pandoc_bin, str(report_md), "-o", str(temp_html), "--standalone"]
    res = subprocess.run(cmd_pandoc, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Pandoc error: {res.stderr}")
        sys.exit(res.returncode)

    print("Step 2: Injecting executive styling & embedding base64 images...")
    with open(temp_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Replace local image paths with base64 data URIs
    def replace_img(match):
        img_src = match.group(1)
        full_path = (REPO_ROOT / img_src).resolve()
        if full_path.exists():
            return f'src="{get_base64_image(full_path)}"'
        return match.group(0)

    html_content = re.sub(r'src="([^"]+\.(?:jpg|jpeg|png))"', replace_img, html_content)

    custom_css = """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
      
      @page {
        size: letter;
        margin: 16mm 14mm 16mm 14mm;
        @bottom-right {
          content: "Page " counter(page) " of " counter(pages);
          font-family: 'Inter', sans-serif;
          font-size: 8pt;
          color: #718096;
        }
      }

      body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1a202c;
        line-height: 1.55;
        font-size: 9.5pt;
        background-color: #ffffff;
      }

      h1 {
        font-size: 20pt;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
        border-bottom: 2.5px solid #2563eb;
        padding-bottom: 8px;
      }

      h2 {
        font-size: 13.5pt;
        font-weight: 700;
        color: #1e293b;
        margin-top: 20px;
        margin-bottom: 8px;
        letter-spacing: -0.01em;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
        page-break-after: avoid;
      }

      h3 {
        font-size: 11pt;
        font-weight: 600;
        color: #334155;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
      }

      h4 {
        font-size: 10pt;
        font-weight: 600;
        color: #475569;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
      }

      p {
        margin-top: 0;
        margin-bottom: 8px;
      }

      ul, ol {
        margin-top: 0;
        margin-bottom: 8px;
        padding-left: 20px;
      }

      li {
        margin-bottom: 3px;
      }

      hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 16px 0;
      }

      /* Tables */
      table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0 14px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
      }

      th, td {
        padding: 5px 8px;
        text-align: left;
        border: 1px solid #cbd5e1;
      }

      th {
        background-color: #f1f5f9;
        color: #0f172a;
        font-weight: 600;
      }

      tr:nth-child(even) {
        background-color: #f8fafc;
      }

      /* Code & Diagrams */
      pre {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 10px 14px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 7.5pt;
        line-height: 1.35;
        overflow-x: auto;
        page-break-inside: avoid;
        margin: 8px 0 12px 0;
      }

      code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 8.5pt;
        background-color: #f1f5f9;
        color: #b91c1c;
        padding: 1px 4px;
        border-radius: 3px;
      }

      pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
      }

      /* Images & Figures */
      img {
        max-width: 100%;
        height: auto;
        border-radius: 4px;
        border: 1px solid #cbd5e1;
        margin-top: 4px;
        margin-bottom: 2px;
        page-break-inside: avoid;
      }

      em {
        color: #64748b;
        font-size: 8pt;
        display: block;
        margin-bottom: 8px;
      }

      .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 8pt;
        font-weight: 600;
        border-radius: 12px;
        background-color: #dbeafe;
        color: #1e40af;
      }
    </style>
    """

    # Insert CSS before </head>
    html_content = html_content.replace("</head>", f"{custom_css}\n</head>")

    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Step 3: Rendering PDF via Headless Chrome...")
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    chrome_tmp = Path("/tmp/chrome_pdf_tmp")
    chrome_tmp.mkdir(parents=True, exist_ok=True)

    cmd_chrome = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--user-data-dir={chrome_tmp}",
        f"--print-to-pdf={output_pdf}",
        str(temp_html),
    ]

    res_chrome = subprocess.run(cmd_chrome, capture_output=True, text=True)
    if res_chrome.returncode != 0:
        print(f"Chrome error: {res_chrome.stderr}")
        sys.exit(res_chrome.returncode)

    if output_pdf.exists():
        shutil.copy2(output_pdf, root_pdf)
        pdf_size_mb = output_pdf.stat().st_size / (1024 * 1024)
        print(f"\n[SUCCESS] PDF Report generated successfully ({pdf_size_mb:.2f} MB):")
        print(f"  - {output_pdf}")
        print(f"  - {root_pdf}")
    else:
        print("Error: Output PDF was not created.")
        sys.exit(1)


if __name__ == "__main__":
    generate_pdf()
