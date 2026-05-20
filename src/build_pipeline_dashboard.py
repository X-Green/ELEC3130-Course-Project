"""Build a static HTML dashboard for pipeline visualization and ablation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2

try:
    from .alignment import register_board
    from .component_inspection import inspect_components
    from .detection import evaluate_detections, fuse_and_classify_defects
    from .io_utils import load_image, load_roi_json
    from .models import AlignmentResult, PipelineConfig, PipelineResult
    from .pipeline import candidate_to_dict, run_pipeline, write_visual_outputs
    from .preprocessing import enhance_and_segment
    from .run_dataset_eval import (
        _filter_predictions_for_evaluation,
        _load_ground_truth,
        _normalize_prediction_labels_for_case,
        _resolve_path,
    )
    from .solder_inspection import inspect_solder_joints
except ImportError:  # pragma: no cover - supports direct script execution
    from alignment import register_board
    from component_inspection import inspect_components
    from detection import evaluate_detections, fuse_and_classify_defects
    from io_utils import load_image, load_roi_json
    from models import AlignmentResult, PipelineConfig, PipelineResult
    from pipeline import candidate_to_dict, run_pipeline, write_visual_outputs
    from preprocessing import enhance_and_segment
    from run_dataset_eval import (
        _filter_predictions_for_evaluation,
        _load_ground_truth,
        _normalize_prediction_labels_for_case,
        _resolve_path,
    )
    from solder_inspection import inspect_solder_joints


DEFAULT_CASE_IDS = [
    "board01_registration_rotated_5deg_blue_mat",
    "board01_enhance_periodic_noise",
    "board01_missing_D11",
    "board01_shifted_D11",
    "board02_rotated_C14",
    "board02_solder_bridge_C14",
    "board02_insufficient_solder_C14",
    "board04_solder_bridge_L5",
]

ABLATIONS = [
    {
        "id": "full",
        "label": "Full Pipeline",
        "disabled": [],
        "summary": "All stages enabled.",
    },
    {
        "id": "no_alignment",
        "label": "No Alignment",
        "disabled": ["registration"],
        "summary": "Uses the raw test image in golden coordinates without homography warping.",
    },
    {
        "id": "no_roi_segmentation",
        "label": "No ROI Masking",
        "disabled": ["roi_constraints"],
        "summary": "Runs segmentation masks globally, then still inspects known component and solder ROIs.",
    },
    {
        "id": "no_component",
        "label": "No Component Inspector",
        "disabled": ["component_inspection"],
        "summary": "Keeps solder inspection and fusion, removes component defect candidates.",
    },
    {
        "id": "no_solder",
        "label": "No Solder Inspector",
        "disabled": ["solder_inspection"],
        "summary": "Keeps component inspection and fusion, removes solder defect candidates.",
    },
    {
        "id": "no_frequency_debug",
        "label": "No FFT Outputs",
        "disabled": ["frequency_debug"],
        "summary": "Disables frequency-domain debug images while keeping spatial processing.",
    },
    {
        "id": "no_fusion",
        "label": "No Fusion",
        "disabled": ["fusion"],
        "summary": "Shows raw component and solder candidates before final deduplication/fusion.",
    },
]

STAGE_LABELS = [
    ("registration", "Registration"),
    ("enhancement", "Enhancement"),
    ("frequency_debug", "FFT"),
    ("segmentation", "Segmentation"),
    ("roi_constraints", "ROI"),
    ("component_inspection", "Component"),
    ("solder_inspection", "Solder"),
    ("fusion", "Fusion"),
]

IMAGE_SLOTS = [
    ("golden", "Golden"),
    ("test", "Test"),
    ("aligned_test.png", "Aligned"),
    ("alignment_matches_inliers.png", "Inlier Matches"),
    ("alignment_checkerboard.png", "Checkerboard"),
    ("alignment_edge_overlay_red_test_green_golden.png", "Edge Overlay"),
    ("alignment_difference_heatmap.png", "Difference Heatmap"),
    ("component_mask.png", "Component Mask"),
    ("solder_mask.png", "Solder Mask"),
    ("frequency_spectrum.png", "FFT Spectrum"),
    ("frequency_bandpass_difference.png", "FFT Difference"),
    ("defect_overlay.png", "Defect Overlay"),
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _relative_to_dashboard(path: Path, dashboard_root: Path) -> str:
    return path.resolve().relative_to(dashboard_root.resolve()).as_posix()


def _identity_alignment(golden_image, test_image) -> AlignmentResult:
    test = test_image
    if golden_image.shape[:2] != test_image.shape[:2]:
        test = cv2.resize(
            test_image,
            (golden_image.shape[1], golden_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return AlignmentResult(
        aligned_test_image=test,
        transform_matrix=None,
        board_mask=None,
        debug_images={},
        metadata={
            "status": "completed",
            "method": "identity_ablation_no_registration",
        },
    )


def _run_ablation(
    golden_path: Path,
    test_path: Path,
    roi_path: Path,
    output_dir: Path,
    variant_id: str,
) -> PipelineResult:
    config = PipelineConfig(debug=True)
    if variant_id == "no_frequency_debug":
        config.enable_frequency_debug = False

    if variant_id == "full":
        result = run_pipeline(golden_path, test_path, roi_path, config)
        write_visual_outputs(result, output_dir)
        return result

    golden_image = load_image(golden_path)
    test_image = load_image(test_path)
    roi_map = load_roi_json(roi_path)

    if variant_id == "no_alignment":
        alignment = _identity_alignment(golden_image, test_image)
    else:
        alignment = register_board(golden_image, test_image, config)

    segmentation_roi_map = None if variant_id == "no_roi_segmentation" else roi_map
    segmentation = enhance_and_segment(
        golden_image,
        alignment.aligned_test_image,
        segmentation_roi_map,
        config,
    )

    component_defects = []
    solder_defects = []
    if variant_id != "no_component":
        component_defects = inspect_components(
            golden_image,
            alignment.aligned_test_image,
            segmentation,
            roi_map,
            config,
        )
    if variant_id != "no_solder":
        solder_defects = inspect_solder_joints(
            golden_image,
            alignment.aligned_test_image,
            segmentation,
            roi_map,
            config,
        )

    if (
        alignment.metadata.get("mean_abs_difference_on_board", 0.0)
        > config.solder_global_difference_skip_threshold
        and not component_defects
    ):
        for defect in solder_defects:
            defect.metadata["suppressed_reason"] = "global_appearance_difference"
        solder_defects = []

    if variant_id == "no_fusion":
        final_defects = [*component_defects, *solder_defects]
        for index, defect in enumerate(final_defects, start=1):
            defect.metadata["raw_rank_without_fusion"] = index
    else:
        final_defects = fuse_and_classify_defects(
            component_defects,
            solder_defects,
            segmentation,
            roi_map,
            config,
        )

    result = PipelineResult(
        roi_map=roi_map,
        alignment=alignment,
        segmentation=segmentation,
        component_defects=component_defects,
        solder_defects=solder_defects,
        final_defects=final_defects,
        metadata={
            "golden_image_path": str(golden_path),
            "test_image_path": str(test_path),
            "roi_json_path": str(roi_path),
            "ablation_variant": variant_id,
            "status": "completed",
        },
    )
    write_visual_outputs(result, output_dir)
    return result


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summarize_result(
    result: PipelineResult,
    case: dict[str, Any],
    ground_truth: list,
    output_dir: Path,
    dashboard_root: Path,
) -> dict[str, Any]:
    predictions = _filter_predictions_for_evaluation(result.final_defects, case)
    evaluated_predictions = _normalize_prediction_labels_for_case(predictions, case)
    metrics = evaluate_detections(
        evaluated_predictions,
        ground_truth,
        iou_threshold=PipelineConfig().evaluation_iou_threshold,
    )

    alignment = result.alignment.metadata
    report = {
        "metrics": metrics,
        "alignment": alignment,
        "component_count": len(result.component_defects),
        "solder_count": len(result.solder_defects),
        "final_count": len(result.final_defects),
        "defects": [candidate_to_dict(candidate) for candidate in result.final_defects],
        "component_defects": [candidate_to_dict(candidate) for candidate in result.component_defects],
        "solder_defects": [candidate_to_dict(candidate) for candidate in result.solder_defects],
    }
    _write_json(output_dir / "ablation_report.json", report)

    images: dict[str, str] = {}
    for image_name, _ in IMAGE_SLOTS:
        if image_name in {"golden", "test"}:
            continue
        image_path = output_dir / image_name
        if image_path.exists():
            images[image_name] = _relative_to_dashboard(image_path, dashboard_root)

    return {
        "id": output_dir.name,
        "output_dir": _relative_to_dashboard(output_dir, dashboard_root),
        "prediction_count": len(evaluated_predictions),
        "truth_count": len(ground_truth),
        "true_positive": int(metrics["true_positive"]),
        "false_positive": int(metrics["false_positive"]),
        "false_negative": int(metrics["false_negative"]),
        "precision": _metric_value(metrics, "precision"),
        "recall": _metric_value(metrics, "recall"),
        "f1": _metric_value(metrics, "f1_score"),
        "component_count": len(result.component_defects),
        "solder_count": len(result.solder_defects),
        "final_count": len(result.final_defects),
        "alignment": {
            "method": alignment.get("method", ""),
            "edge_iou": alignment.get("edge_iou_dilated", 0.0),
            "edge_chamfer_mean_px": alignment.get("edge_chamfer_mean_px", 0.0),
            "edge_chamfer_p90_px": alignment.get("edge_chamfer_p90_px", 0.0),
            "reprojection_p90_px": alignment.get("reprojection_error_p90_px", 0.0),
            "mean_abs_difference": alignment.get("mean_abs_difference_on_board", 0.0),
        },
        "defects": [candidate_to_dict(candidate) for candidate in result.final_defects],
        "images": images,
    }


def _select_cases(manifest: dict[str, Any], requested_ids: list[str] | None) -> list[dict[str, Any]]:
    cases = manifest.get("cases", [])
    by_id = {case["case_id"]: case for case in cases}
    ids = requested_ids or DEFAULT_CASE_IDS
    selected = []
    for case_id in ids:
        if case_id in by_id:
            selected.append(by_id[case_id])
    if selected:
        return selected
    return cases[:6]


def _html_template(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PCBA Pipeline Ablation Dashboard</title>
  <style>
    :root {{
      --bg: #f5f6f2;
      --panel: #ffffff;
      --ink: #20231f;
      --muted: #62685f;
      --line: #d9ddd3;
      --accent: #167a68;
      --warn: #b65d23;
      --bad: #b33a3a;
      --good: #2d7a34;
      --shadow: 0 1px 2px rgba(30, 35, 28, 0.08);
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
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 22px;
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 700;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 8px;
      max-width: 1180px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fbfcf8;
      min-height: 58px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .metric strong {{
      font-size: 19px;
      line-height: 1;
    }}
    main {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
    }}
    aside {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: var(--shadow);
      padding: 12px;
      align-self: start;
      position: sticky;
      top: 110px;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin: 10px 0 5px;
    }}
    select {{
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
      padding: 5px 8px;
      font-size: 14px;
    }}
    .stage-strip {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 6px;
      margin-top: 12px;
    }}
    .stage {{
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 6px 7px;
      font-size: 12px;
      background: #f8faf5;
    }}
    .stage.off {{
      border-color: #e0bca5;
      background: #fff4ec;
      color: var(--warn);
    }}
    .content {{
      min-width: 0;
      display: grid;
      gap: 14px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: var(--shadow);
      padding: 12px;
      min-width: 0;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 17px;
      line-height: 1.2;
    }}
    .matrix {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .matrix th,
    .matrix td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      vertical-align: top;
    }}
    .matrix th {{
      color: var(--muted);
      font-weight: 600;
      background: #fafbf7;
    }}
    .good {{ color: var(--good); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 10px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcf8;
      overflow: hidden;
    }}
    figcaption {{
      padding: 7px 9px;
      color: var(--muted);
      font-size: 12px;
      border-bottom: 1px solid var(--line);
    }}
    figure img {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      display: block;
      background: #eef0eb;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f7f8f4;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 10px;
      font-size: 12px;
      max-height: 360px;
      overflow: auto;
    }}
    @media (max-width: 900px) {{
      header {{ position: static; }}
      main {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
      .summary {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>PCBA Pipeline Ablation Dashboard</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <aside>
      <label for="caseSelect">Case</label>
      <select id="caseSelect"></select>
      <label for="variantSelect">Pipeline Variant</label>
      <select id="variantSelect"></select>
      <div class="stage-strip" id="stageStrip"></div>
    </aside>
    <div class="content">
      <section>
        <h2>Case Metrics</h2>
        <div id="caseMetrics"></div>
      </section>
      <section>
        <h2>Ablation Matrix</h2>
        <div id="matrix"></div>
      </section>
      <section>
        <h2>Stage Images</h2>
        <div class="image-grid" id="images"></div>
      </section>
      <section>
        <h2>Final Defects</h2>
        <pre id="defects"></pre>
      </section>
    </div>
  </main>
  <script>
    const DATA = {data_json};
    const caseSelect = document.getElementById('caseSelect');
    const variantSelect = document.getElementById('variantSelect');
    const summary = document.getElementById('summary');
    const stageStrip = document.getElementById('stageStrip');
    const caseMetrics = document.getElementById('caseMetrics');
    const matrix = document.getElementById('matrix');
    const images = document.getElementById('images');
    const defects = document.getElementById('defects');

    const fmt = (value, digits = 3) => {{
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits) : '0.000';
    }};

    function fillSelects() {{
      DATA.cases.forEach((item, index) => {{
        const option = document.createElement('option');
        option.value = item.case_id;
        option.textContent = `${{item.case_id}} · ${{item.variant_type}}`;
        caseSelect.appendChild(option);
        if (index === 0) option.selected = true;
      }});
      DATA.ablations.forEach((item, index) => {{
        const option = document.createElement('option');
        option.value = item.id;
        option.textContent = item.label;
        variantSelect.appendChild(option);
        if (index === 0) option.selected = true;
      }});
    }}

    function selectedCase() {{
      return DATA.cases.find((item) => item.case_id === caseSelect.value) || DATA.cases[0];
    }}

    function selectedVariant(caseData) {{
      return caseData.variants[variantSelect.value] || caseData.variants.full;
    }}

    function renderSummary() {{
      const items = [
        ['Cases', DATA.cases.length],
        ['Ablations', DATA.ablations.length],
        ['Dataset F1', fmt(DATA.dataset_metrics.overall.f1_score)],
        ['Dataset Precision', fmt(DATA.dataset_metrics.overall.precision)],
        ['Dataset Recall', fmt(DATA.dataset_metrics.overall.recall)],
      ];
      summary.innerHTML = items.map(([label, value]) => `
        <div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>
      `).join('');
    }}

    function renderStages(ablation) {{
      const disabled = new Set(ablation.disabled || []);
      stageStrip.innerHTML = DATA.stage_labels.map(([id, label]) => {{
        const off = disabled.has(id);
        return `<div class="stage ${{off ? 'off' : ''}}">${{label}}: ${{off ? 'off' : 'on'}}</div>`;
      }}).join('');
    }}

    function renderMetrics(caseData, variantData) {{
      const a = variantData.alignment || {{}};
      const rows = [
        ['Case', caseData.case_id],
        ['Variant Type', caseData.variant_type],
        ['Truth / Prediction', `${{variantData.truth_count}} / ${{variantData.prediction_count}}`],
        ['TP / FP / FN', `${{variantData.true_positive}} / ${{variantData.false_positive}} / ${{variantData.false_negative}}`],
        ['Precision / Recall / F1', `${{fmt(variantData.precision)}} / ${{fmt(variantData.recall)}} / ${{fmt(variantData.f1)}}`],
        ['Component / Solder / Final Candidates', `${{variantData.component_count}} / ${{variantData.solder_count}} / ${{variantData.final_count}}`],
        ['Alignment Method', a.method || ''],
        ['Edge IoU / Chamfer Mean', `${{fmt(a.edge_iou)}} / ${{fmt(a.edge_chamfer_mean_px)}} px`],
        ['Reprojection p90 / Mean Abs Diff', `${{fmt(a.reprojection_p90_px)}} px / ${{fmt(a.mean_abs_difference)}}`],
      ];
      caseMetrics.innerHTML = `<table class="matrix"><tbody>${{rows.map(([k, v]) => `<tr><th>${{k}}</th><td>${{v}}</td></tr>`).join('')}}</tbody></table>`;
    }}

    function renderMatrix(caseData) {{
      const rows = DATA.ablations.map((ablation) => {{
        const item = caseData.variants[ablation.id];
        const cls = item.false_positive || item.false_negative ? 'bad' : 'good';
        return `<tr>
          <td>${{ablation.label}}</td>
          <td>${{item.prediction_count}}</td>
          <td class="${{cls}}">${{item.true_positive}} / ${{item.false_positive}} / ${{item.false_negative}}</td>
          <td>${{fmt(item.f1)}}</td>
          <td>${{fmt(item.alignment?.edge_iou)}}</td>
          <td>${{fmt(item.alignment?.edge_chamfer_mean_px)}}</td>
        </tr>`;
      }}).join('');
      matrix.innerHTML = `<table class="matrix">
        <thead><tr><th>Variant</th><th>Pred</th><th>TP / FP / FN</th><th>F1</th><th>Edge IoU</th><th>Chamfer px</th></tr></thead>
        <tbody>${{rows}}</tbody>
      </table>`;
    }}

    function renderImages(caseData, variantData) {{
      const slots = DATA.image_slots;
      images.innerHTML = slots.map(([key, label]) => {{
        let src = '';
        if (key === 'golden') src = caseData.golden_image;
        else if (key === 'test') src = caseData.test_image;
        else src = variantData.images[key] || '';
        if (!src) {{
          return `<figure><figcaption>${{label}}</figcaption><div class="muted" style="padding:18px">not generated</div></figure>`;
        }}
        return `<figure><figcaption>${{label}}</figcaption><img src="${{src}}" alt="${{label}}"></figure>`;
      }}).join('');
    }}

    function renderDefects(variantData) {{
      defects.textContent = JSON.stringify(variantData.defects || [], null, 2);
    }}

    function render() {{
      const caseData = selectedCase();
      const ablation = DATA.ablations.find((item) => item.id === variantSelect.value) || DATA.ablations[0];
      const variantData = selectedVariant(caseData);
      renderSummary();
      renderStages(ablation);
      renderMetrics(caseData, variantData);
      renderMatrix(caseData);
      renderImages(caseData, variantData);
      renderDefects(variantData);
    }}

    fillSelects();
    render();
    caseSelect.addEventListener('change', render);
    variantSelect.addEventListener('change', render);
  </script>
</body>
</html>
"""


def build_dashboard(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics)
    dataset_metrics = _load_json(metrics_path) if metrics_path.exists() else {
        "overall": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    }

    selected_cases = _select_cases(manifest, args.case_id)
    dashboard_cases: list[dict[str, Any]] = []

    for case in selected_cases:
        case_id = case["case_id"]
        case_root = output_root / "cases" / case_id
        case_root.mkdir(parents=True, exist_ok=True)

        golden_path = _resolve_path(case["golden"], manifest_path)
        test_path = _resolve_path(case["test"], manifest_path)
        roi_path = _resolve_path(case["roi"], manifest_path)
        ground_truth_path = _resolve_path(case["ground_truth"], manifest_path)
        ground_truth = _load_ground_truth(ground_truth_path)

        golden_copy = case_root / "input_golden.png"
        test_copy = case_root / "input_test.png"
        _copy_image(golden_path, golden_copy)
        _copy_image(test_path, test_copy)

        variants: dict[str, Any] = {}
        for ablation in ABLATIONS:
            variant_id = ablation["id"]
            variant_output = case_root / variant_id
            result = _run_ablation(
                golden_path,
                test_path,
                roi_path,
                variant_output,
                variant_id,
            )
            variants[variant_id] = _summarize_result(
                result,
                case,
                ground_truth,
                variant_output,
                output_root,
            )

        dashboard_cases.append(
            {
                "case_id": case_id,
                "board_id": case.get("board_id", ""),
                "variant_type": case.get("variant_type", ""),
                "golden_image": _relative_to_dashboard(golden_copy, output_root),
                "test_image": _relative_to_dashboard(test_copy, output_root),
                "variants": variants,
            }
        )

    data = {
        "manifest": str(manifest_path),
        "dataset_metrics": dataset_metrics,
        "ablations": ABLATIONS,
        "stage_labels": STAGE_LABELS,
        "image_slots": IMAGE_SLOTS,
        "cases": dashboard_cases,
    }
    _write_json(output_root / "dashboard_data.json", data)
    html_path = output_root / "index.html"
    html_path.write_text(_html_template(data), encoding="utf-8")
    print(f"Wrote {html_path}")
    return html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a static HTML pipeline ablation dashboard.")
    parser.add_argument("--manifest", default="data/kicad_synth/manifest.json", help="Dataset manifest JSON.")
    parser.add_argument("--metrics", default="outputs/kicad_synth_eval/metrics.json", help="Dataset metrics JSON.")
    parser.add_argument("--output-dir", default="outputs/pipeline_dashboard", help="Dashboard output directory.")
    parser.add_argument("--case-id", nargs="*", default=None, help="Specific case ids to include.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_dashboard(args)


if __name__ == "__main__":
    main()
