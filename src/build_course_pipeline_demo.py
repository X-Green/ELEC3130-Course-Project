"""Build the static course-facing PCBA pipeline demo."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


STAGES = [
    {
        "id": "inputs",
        "title": "1. Inputs",
        "function": "run_pipeline() -> load_image(), load_roi_json()",
        "input": [
            "Golden reference render",
            "Synthetic test image",
            "KiCad-derived component and solder ROI JSON",
        ],
        "methods": [
            "Reference-based inspection",
            "Golden-coordinate ROI mapping",
            "Ground truth reserved for evaluation only",
        ],
        "output": [
            "BGR image arrays",
            "Component and solder ROI map",
            "Case metadata for reporting",
        ],
        "images": ["input_golden.png", "input_test.png"],
    },
    {
        "id": "registration",
        "title": "2. Board Registration",
        "function": "register_board()",
        "input": ["Golden image", "Raw test image"],
        "methods": [
            "Board mask extraction",
            "ORB keypoints and descriptor matching",
            "RANSAC homography",
            "Phase-correlation fallback",
            "Edge IoU and Chamfer diagnostics",
        ],
        "output": [
            "Aligned test image in golden coordinates",
            "Transform metadata",
            "Match, checkerboard, edge-overlay, and heatmap diagnostics",
        ],
        "images": [
            "aligned_test.png",
            "alignment_matches_inliers.png",
            "alignment_checkerboard.png",
            "alignment_edge_overlay_red_test_green_golden.png",
        ],
    },
    {
        "id": "enhancement",
        "title": "3. Spatial Enhancement",
        "function": "enhance_and_segment()",
        "input": ["Golden image", "Aligned test image"],
        "methods": [
            "LAB luminance CLAHE",
            "Bilateral edge-preserving denoising",
            "Per-channel histogram matching",
            "Brightness and color normalization",
        ],
        "output": [
            "Enhanced golden image",
            "Normalized test image",
            "Stable appearance for thresholding and ROI comparison",
        ],
        "images": ["aligned_test.png", "alignment_difference_heatmap.png", "board_mask.png"],
    },
    {
        "id": "frequency",
        "title": "4. Frequency-Domain Enhancement",
        "function": "suppress_reference_periodic_noise(), log_magnitude_spectrum(), apply_frequency_filter(), frequency_difference_image()",
        "input": ["Enhanced grayscale golden image", "Enhanced grayscale test image"],
        "methods": [
            "2D FFT",
            "Reference-guided notch filtering for periodic noise",
            "Log-magnitude spectrum",
            "High-pass filtering",
            "Band-pass filtering",
            "Golden/test band-pass difference",
        ],
        "output": [
            "Frequency-denoised test image used by downstream inspection",
            "Notch mask showing suppressed frequency peaks",
            "FFT spectrum",
            "High-frequency detail image",
            "Band-pass detail image",
            "Local structural difference map",
        ],
        "images": [
            "frequency_denoised_test.png",
            "frequency_notch_mask.png",
            "frequency_spectrum.png",
            "frequency_highpass.png",
            "frequency_bandpass.png",
            "frequency_bandpass_difference.png",
        ],
    },
    {
        "id": "segmentation",
        "title": "5. Segmentation And Morphology",
        "function": "enhance_and_segment() mask builders",
        "input": ["Normalized test image", "ROI map"],
        "methods": [
            "Otsu thresholding",
            "Canny edge detection",
            "HSV/Lab solder-color segmentation",
            "Morphological opening and closing",
            "ROI-constrained masks",
        ],
        "output": ["Board mask", "Component mask", "Solder mask"],
        "images": ["board_mask.png", "component_mask.png", "solder_mask.png"],
    },
    {
        "id": "component",
        "title": "6. Component Inspection",
        "function": "inspect_components()",
        "input": ["Golden image", "Aligned test image", "Component ROIs"],
        "methods": [
            "Template matching",
            "ROI absolute difference",
            "Edge density",
            "Contour orientation",
            "Rotated-template correlation",
            "Rule-based labels",
        ],
        "output": ["Missing, shifted, rotated, or visual-mismatch component candidates"],
        "images": ["component_mask.png", "defect_overlay.png"],
    },
    {
        "id": "solder",
        "title": "7. Solder Inspection",
        "function": "inspect_solder_joints()",
        "input": ["Enhanced golden/test images", "Solder ROIs", "Solder mask"],
        "methods": [
            "HSV/Lab metallic masks",
            "Adaptive local thresholding",
            "Area and width features",
            "Connected components",
            "Pad-pair corridor bridge evidence",
            "Rule-based solder labels",
        ],
        "output": ["Bridge, insufficient, excessive, or missing solder candidates"],
        "images": ["solder_mask.png", "defect_overlay.png"],
    },
    {
        "id": "fusion",
        "title": "8. Fusion And Report",
        "function": "fuse_and_classify_defects(), write_visual_outputs(), evaluate_detections()",
        "input": ["Component candidates", "Solder candidates", "Segmentation result"],
        "methods": [
            "IoU duplicate suppression",
            "Final rule-based label retention",
            "Overlay rendering",
            "Precision, recall, F1, and false-positive accounting",
        ],
        "output": ["Final defect list", "Overlay image", "report.json", "evaluation.json"],
        "images": ["defect_overlay.png"],
    },
]

CASE_IMAGE_FILES = sorted(
    {
        image_name
        for stage in STAGES
        for image_name in stage["images"]
        if image_name not in {"input_golden.png", "input_test.png"}
    }
)

ROBUSTNESS_ORDER = [
    "registration_background",
    "enhancement_stress",
    "missing_component",
    "shifted_component",
    "rotated_component",
    "solder_bridge",
    "insufficient_solder",
    "excessive_solder",
    "missing_solder",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {row["case_id"]: row for row in rows}


def _resolve_path(raw_path: str | Path, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    manifest_candidate = manifest_path.parent / path
    if manifest_candidate.exists():
        return manifest_candidate
    return cwd_path


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format(value: Any, digits: int = 3) -> str:
    return f"{_as_float(value):.{digits}f}"


def _successful(row: dict[str, str]) -> bool:
    if row.get("error"):
        return False
    if _as_int(row.get("false_positive")) or _as_int(row.get("false_negative")):
        return False
    if _as_int(row.get("truth_count")) == 0:
        return _as_int(row.get("prediction_count")) == 0
    return _as_float(row.get("f1_score")) >= 0.999


def _choose_cases(manifest: dict[str, Any], rows_by_id: dict[str, dict[str, str]], limit: int) -> list[dict[str, Any]]:
    cases = manifest.get("cases", [])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for variant_type in ROBUSTNESS_ORDER:
        candidates = sorted(
            [case for case in cases if case.get("variant_type") == variant_type],
            key=lambda item: (
                0 if _successful(rows_by_id.get(item["case_id"], {})) else 1,
                item.get("board_id", ""),
                item["case_id"],
            ),
        )
        for case in candidates:
            row = rows_by_id.get(case["case_id"])
            if row is None or case["case_id"] in seen:
                continue
            selected.append(case)
            seen.add(case["case_id"])
            break

    for case in sorted(cases, key=lambda item: item["case_id"]):
        if len(selected) >= limit:
            break
        if case["case_id"] in seen:
            continue
        row = rows_by_id.get(case["case_id"])
        if row is None or not _successful(row):
            continue
        selected.append(case)
        seen.add(case["case_id"])

    return selected[:limit]


def _case_metrics(row: dict[str, str], report: dict[str, Any]) -> dict[str, Any]:
    alignment = report.get("alignment", {})
    segmentation = report.get("segmentation", {})
    component_defects = report.get("component_defects", [])
    solder_defects = report.get("solder_defects", [])
    final_defects = report.get("defects", [])
    top_component = component_defects[0] if component_defects else {}
    top_solder = solder_defects[0] if solder_defects else {}
    return {
        "inputs": [
            ["Truth defects", row.get("truth_count", "0")],
            ["Predictions", row.get("prediction_count", "0")],
            ["Variant", row.get("variant_type", "")],
        ],
        "registration": [
            ["Method", alignment.get("method", "")],
            ["Inlier matches", str(alignment.get("inlier_matches", 0))],
            ["Edge IoU", _format(alignment.get("edge_iou_dilated"))],
            ["Chamfer mean px", _format(alignment.get("edge_chamfer_mean_px"))],
        ],
        "enhancement": [
            ["Status", segmentation.get("status", "")],
            ["Mean board abs diff", _format(alignment.get("mean_abs_difference_on_board"))],
            ["Has ROI map", str(segmentation.get("has_roi_map", ""))],
        ],
        "frequency": [
            ["Notch active", str(report.get("segmentation", {}).get("frequency_domain", {}).get("notch_filter", {}).get("active", ""))],
            ["Notch peaks", str(report.get("segmentation", {}).get("frequency_domain", {}).get("notch_filter", {}).get("selected_peak_count", ""))],
            ["Band-pass mean diff", _format(row.get("frequency_bandpass_mean"))],
            ["GT ROI mean diff", _format(row.get("frequency_gt_mean"))],
            ["High-pass mean", _format(row.get("frequency_highpass_mean"))],
        ],
        "segmentation": [
            ["Board mask", "board_mask.png"],
            ["Component mask", "component_mask.png"],
            ["Solder mask", "solder_mask.png"],
        ],
        "component": [
            ["Candidates", str(len(component_defects))],
            ["Top label", str(top_component.get("defect_type", "None"))],
            ["Top score", _format(top_component.get("score")) if top_component else "n/a"],
        ],
        "solder": [
            ["Candidates", str(len(solder_defects))],
            ["Top label", str(top_solder.get("defect_type", "None"))],
            ["Top score", _format(top_solder.get("score")) if top_solder else "n/a"],
        ],
        "fusion": [
            ["Final defects", str(len(final_defects))],
            ["TP / FP / FN", f"{row.get('true_positive', '0')} / {row.get('false_positive', '0')} / {row.get('false_negative', '0')}"],
            ["F1", _format(row.get("f1_score"))],
        ],
    }


def _build_case_payload(
    case: dict[str, Any],
    row: dict[str, str],
    manifest_path: Path,
    eval_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    case_id = case["case_id"]
    source_case_dir = eval_dir / "cases" / case_id
    destination_case_dir = output_root / "cases" / case_id
    destination_case_dir.mkdir(parents=True, exist_ok=True)

    golden_dest = destination_case_dir / "input_golden.png"
    test_dest = destination_case_dir / "input_test.png"
    _copy_file(_resolve_path(case["golden"], manifest_path), golden_dest)
    _copy_file(_resolve_path(case["test"], manifest_path), test_dest)

    image_paths = {
        "input_golden.png": _relative(golden_dest, output_root),
        "input_test.png": _relative(test_dest, output_root),
    }
    for file_name in CASE_IMAGE_FILES:
        source = source_case_dir / file_name
        if source.exists():
            destination = destination_case_dir / file_name
            _copy_file(source, destination)
            image_paths[file_name] = _relative(destination, output_root)

    report_source = source_case_dir / "report.json"
    evaluation_source = source_case_dir / "evaluation.json"
    report_dest = destination_case_dir / "report.json"
    evaluation_dest = destination_case_dir / "evaluation.json"
    _copy_file(report_source, report_dest)
    _copy_file(evaluation_source, evaluation_dest)
    report = _load_json(report_source)
    evaluation = _load_json(evaluation_source)

    stage_metrics = _case_metrics(row, report)
    stages = []
    for stage in STAGES:
        stages.append(
            {
                **stage,
                "metrics": stage_metrics.get(stage["id"], []),
                "image_items": [
                    {
                        "label": image_name.replace("_", " ").replace(".png", "").title(),
                        "src": image_paths[image_name],
                    }
                    for image_name in stage["images"]
                    if image_name in image_paths
                ],
            }
        )

    return {
        "case_id": case_id,
        "board_id": case.get("board_id", ""),
        "variant_type": case.get("variant_type", ""),
        "metadata": case.get("metadata", {}),
        "success": _successful(row),
        "truth_count": _as_int(row.get("truth_count")),
        "prediction_count": _as_int(row.get("prediction_count")),
        "true_positive": _as_int(row.get("true_positive")),
        "false_positive": _as_int(row.get("false_positive")),
        "false_negative": _as_int(row.get("false_negative")),
        "precision": _as_float(row.get("precision")),
        "recall": _as_float(row.get("recall")),
        "f1_score": _as_float(row.get("f1_score")),
        "stages": stages,
        "image_paths": image_paths,
        "report_path": _relative(report_dest, output_root),
        "evaluation_path": _relative(evaluation_dest, output_root),
        "final_defects": report.get("defects", []),
        "ground_truth": evaluation.get("ground_truth", []),
    }


def _robustness_gallery(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    gallery = []
    for case in cases:
        image = case["image_paths"].get("defect_overlay.png") or case["image_paths"].get("input_test.png")
        if not image:
            continue
        label = case["variant_type"]
        metadata = case.get("metadata", {})
        if metadata.get("stress_name"):
            label = f"{label}: {metadata['stress_name']}"
        gallery.append(
            {
                "case_id": case["case_id"],
                "variant_type": label,
                "success": "yes" if case["success"] else "limited",
                "src": image,
            }
        )
    return gallery


def _html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PCBA Classical Pipeline Course Demo</title>
  <style>
    :root {{
      --bg: #f4f6f4;
      --panel: #ffffff;
      --ink: #202421;
      --muted: #5f6862;
      --line: #d9ded8;
      --accent: #0f766e;
      --accent2: #355f8c;
      --good: #23713a;
      --warn: #9b5b15;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }}
    header {{
      padding: 18px 22px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 3;
    }}
    h1 {{ margin: 0 0 6px; font-size: 25px; line-height: 1.2; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; }}
    p {{ margin: 0 0 10px; color: var(--muted); line-height: 1.45; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: #fbfcfb;
      padding: 8px 10px;
      min-height: 58px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    main {{
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 16px;
      padding: 16px;
    }}
    aside {{
      position: sticky;
      top: 112px;
      align-self: start;
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px;
    }}
    label {{ display: block; color: var(--muted); font-size: 12px; margin: 10px 0 5px; }}
    select {{
      width: 100%;
      padding: 8px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      font-size: 13px;
    }}
    .stage-nav {{ display: grid; gap: 6px; margin-top: 12px; }}
    button {{
      text-align: left;
      border: 1px solid var(--line);
      background: #fbfcfb;
      color: var(--ink);
      padding: 8px;
      cursor: pointer;
      font-size: 13px;
    }}
    button.active {{ border-color: var(--accent); background: #e8f4f1; color: var(--accent); }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 14px;
      margin-bottom: 16px;
    }}
    .case-card {{
      border-top: 1px solid var(--line);
      margin-top: 12px;
      padding-top: 12px;
      font-size: 13px;
    }}
    .pill {{
      display: inline-block;
      padding: 3px 7px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      margin-top: 5px;
    }}
    .pill.good {{ color: var(--good); border-color: #a9d2b7; background: #edf7ef; }}
    .stage-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) 280px;
      gap: 12px;
    }}
    .explain {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .box {{
      border: 1px solid var(--line);
      background: #fbfcfb;
      padding: 10px;
    }}
    .box.full {{ grid-column: 1 / -1; }}
    ul {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.45; }}
    code {{
      background: #eef1ed;
      padding: 2px 5px;
      border: 1px solid var(--line);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 400; width: 45%; }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      background: #eef0ee;
    }}
    figcaption {{
      padding: 7px 8px;
      background: #fbfcfb;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
    }}
    img {{
      display: block;
      width: 100%;
      height: 250px;
      object-fit: contain;
      background: #eef0ee;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .gallery img {{ height: 150px; }}
    pre {{
      margin: 0;
      max-height: 310px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f7f8f6;
      border: 1px solid var(--line);
      padding: 10px;
      font-size: 12px;
    }}
    @media (max-width: 1000px) {{
      header {{ position: static; }}
      main {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .stage-grid, .explain {{ grid-template-columns: 1fr; }}
      .image-grid, .gallery {{ grid-template-columns: 1fr; }}
      img, .gallery img {{ height: auto; max-height: 280px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>PCBA Classical Image Processing Pipeline</h1>
    <p>Static course demo for the full reference-based inspection workflow in <code>src/pipeline.py</code>.</p>
    <div class="metrics" id="metrics"></div>
  </header>
  <main>
    <aside>
      <label for="caseSelect">Case</label>
      <select id="caseSelect"></select>
      <label for="stageSelect">Pipeline Stage</label>
      <select id="stageSelect"></select>
      <div class="stage-nav" id="stageNav"></div>
      <div class="case-card" id="caseCard"></div>
    </aside>
    <div>
      <section id="stagePanel"></section>
      <section>
        <h2>Robustness Gallery</h2>
        <p>One page view of the dataset elements: camera/background variation, enhancement stress cases, component defects, and solder defects.</p>
        <div class="gallery" id="gallery"></div>
      </section>
      <section>
        <h2>Final Report JSON</h2>
        <pre id="finalJson"></pre>
      </section>
    </div>
  </main>
  <script>
    const DATA = {data_json};
    const fmt = (v, digits = 3) => {{
      const n = Number(v);
      return Number.isFinite(n) ? n.toFixed(digits) : '0.000';
    }};
    const metrics = document.getElementById('metrics');
    const caseSelect = document.getElementById('caseSelect');
    const stageSelect = document.getElementById('stageSelect');
    const stageNav = document.getElementById('stageNav');
    const caseCard = document.getElementById('caseCard');
    const stagePanel = document.getElementById('stagePanel');
    const gallery = document.getElementById('gallery');
    const finalJson = document.getElementById('finalJson');

    function currentCase() {{
      return DATA.cases.find((item) => item.case_id === caseSelect.value) || DATA.cases[0];
    }}
    function currentStage(item) {{
      return item.stages.find((stage) => stage.id === stageSelect.value) || item.stages[0];
    }}
    function list(items) {{
      return `<ul>${{items.map((item) => `<li>${{item}}</li>`).join('')}}</ul>`;
    }}
    function metricRows(rows) {{
      return `<table><tbody>${{rows.map(([label, value]) => `<tr><th>${{label}}</th><td>${{value}}</td></tr>`).join('')}}</tbody></table>`;
    }}
    function renderMetrics() {{
      const overall = DATA.metrics.overall || {{}};
      const cards = [
        ['Cases', DATA.metrics.case_count],
        ['Demo cases', DATA.cases.length],
        ['Precision', fmt(overall.precision)],
        ['Recall', fmt(overall.recall)],
        ['F1', fmt(overall.f1_score)],
        ['FP rate', fmt(DATA.metrics.defect_free_false_positive_rate)],
      ];
      metrics.innerHTML = cards.map(([label, value]) => `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`).join('');
    }}
    function fillControls() {{
      DATA.cases.forEach((item) => {{
        const option = document.createElement('option');
        option.value = item.case_id;
        option.textContent = `${{item.case_id}} · ${{item.variant_type}}`;
        caseSelect.appendChild(option);
      }});
      DATA.stages.forEach((stage) => {{
        const option = document.createElement('option');
        option.value = stage.id;
        option.textContent = stage.title;
        stageSelect.appendChild(option);
      }});
    }}
    function renderNav(item) {{
      stageNav.innerHTML = item.stages.map((stage) => `<button class="${{stage.id === stageSelect.value ? 'active' : ''}}" data-stage="${{stage.id}}">${{stage.title}}</button>`).join('');
      stageNav.querySelectorAll('button').forEach((button) => button.addEventListener('click', () => {{
        stageSelect.value = button.dataset.stage;
        render();
      }}));
    }}
    function renderCaseCard(item) {{
      const badge = item.success ? '<span class="pill good">clean evaluation</span>' : '<span class="pill">limited case</span>';
      caseCard.innerHTML = `
        <p><strong>${{item.case_id}}</strong></p>
        <p>${{item.board_id}} · ${{item.variant_type}}</p>
        <p>Truth / prediction: ${{item.truth_count}} / ${{item.prediction_count}}</p>
        <p>TP / FP / FN: ${{item.true_positive}} / ${{item.false_positive}} / ${{item.false_negative}}</p>
        <p>Precision / Recall / F1: ${{fmt(item.precision)}} / ${{fmt(item.recall)}} / ${{fmt(item.f1_score)}}</p>
        ${{badge}}
      `;
    }}
    function renderImages(stage) {{
      if (!stage.image_items.length) return '<p>No image artifact available for this stage.</p>';
      return `<div class="image-grid">${{stage.image_items.map((image) => `<figure><figcaption>${{image.label}}</figcaption><img src="${{image.src}}" alt="${{image.label}}"></figure>`).join('')}}</div>`;
    }}
    function renderStage(item, stage) {{
      stagePanel.innerHTML = `
        <h2>${{stage.title}}</h2>
        <div class="stage-grid">
          <div class="explain">
            <div class="box"><h3>Input</h3>${{list(stage.input)}}</div>
            <div class="box"><h3>Function</h3><p><code>${{stage.function}}</code></p></div>
            <div class="box"><h3>DIP Methods</h3>${{list(stage.methods)}}</div>
            <div class="box"><h3>Output</h3>${{list(stage.output)}}</div>
            <div class="box full"><h3>Evidence</h3>${{renderImages(stage)}}</div>
          </div>
          <div class="box"><h3>Stage Metrics</h3>${{metricRows(stage.metrics || [])}}</div>
        </div>
      `;
      finalJson.textContent = JSON.stringify({{
        report: item.report_path,
        evaluation: item.evaluation_path,
        ground_truth: item.ground_truth,
        final_defects: item.final_defects,
      }}, null, 2);
    }}
    function renderGallery() {{
      gallery.innerHTML = DATA.robustness_gallery.map((item) => `
        <figure>
          <figcaption>${{item.variant_type}} · ${{item.case_id}}</figcaption>
          <img src="${{item.src}}" alt="${{item.case_id}}">
        </figure>
      `).join('');
    }}
    function render() {{
      const item = currentCase();
      if (!item.stages.some((stage) => stage.id === stageSelect.value)) stageSelect.value = item.stages[0].id;
      renderNav(item);
      renderCaseCard(item);
      renderStage(item, currentStage(item));
    }}
    renderMetrics();
    fillControls();
    renderGallery();
    render();
    caseSelect.addEventListener('change', render);
    stageSelect.addEventListener('change', render);
  </script>
</body>
</html>
"""


def build_demo(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.manifest)
    eval_dir = Path(args.eval_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(manifest_path)
    rows_by_id = _read_summary(eval_dir / "summary.csv")
    metrics = _load_json(eval_dir / "metrics.json")

    selected_cases = _choose_cases(manifest, rows_by_id, args.limit)
    if not selected_cases:
        raise RuntimeError("No cases available for the course pipeline demo.")

    demo_cases = [
        _build_case_payload(
            case,
            rows_by_id[case["case_id"]],
            manifest_path,
            eval_dir,
            output_root,
        )
        for case in selected_cases
    ]

    data = {
        "manifest": str(manifest_path),
        "eval_dir": str(eval_dir),
        "metrics": metrics,
        "stages": STAGES,
        "cases": demo_cases,
        "robustness_gallery": _robustness_gallery(demo_cases),
    }
    _write_json(output_root / "demo_data.json", data)
    html_path = output_root / "index.html"
    html_path.write_text(_html(data), encoding="utf-8")

    print(f"Wrote {html_path}")
    for case in demo_cases:
        print(f"- {case['case_id']} ({case['variant_type']})")
    return html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the static course PCBA pipeline demo.")
    parser.add_argument("--manifest", default="data/kicad_synth/manifest.json", help="Dataset manifest JSON.")
    parser.add_argument("--eval-dir", default="outputs/kicad_synth_eval", help="Evaluation output directory.")
    parser.add_argument("--output-dir", default="outputs/course_pipeline_demo", help="Demo output directory.")
    parser.add_argument("--limit", type=int, default=14, help="Maximum number of demo cases to include.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_demo(args)


if __name__ == "__main__":
    main()
