"""Build a static HTML explainer for the PCBA inspection pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_VARIANT_ORDER = [
    "registration_background",
    "enhancement_stress",
    "missing_component",
    "shifted_component",
    "excessive_solder",
    "missing_solder",
    "solder_bridge",
]

PREFERRED_CASE_IDS = {
    "registration_background": [
        "board01_registration_rotated_5deg_blue_mat",
        "board01_registration_shift_rotate_blue_mat",
        "board02_registration_rotated_5deg_blue_mat",
    ],
    "enhancement_stress": [
        "board01_enhance_periodic_noise",
        "board02_enhance_periodic_noise",
        "board01_enhance_illumination_gradient",
    ],
    "missing_component": [
        "board01_missing_D11",
        "board01_missing_D12",
        "board03_missing_F1",
    ],
    "shifted_component": [
        "board01_shifted_D11",
        "board01_shifted_D12",
        "board03_shifted_F1",
    ],
    "excessive_solder": [
        "board01_excessive_solder_D11",
        "board02_excessive_solder_C14",
        "board03_excessive_solder_F1",
    ],
    "missing_solder": [
        "board01_missing_solder_D11",
        "board02_missing_solder_C14",
        "board03_missing_solder_F1",
    ],
    "solder_bridge": [
        "board02_solder_bridge_C14",
        "board03_solder_bridge_F1",
        "board04_solder_bridge_L6",
    ],
}

CASE_IMAGE_FILES = [
    "aligned_test.png",
    "alignment_matches_inliers.png",
    "alignment_checkerboard.png",
    "alignment_edge_overlay_red_test_green_golden.png",
    "alignment_difference_heatmap.png",
    "board_mask.png",
    "component_mask.png",
    "solder_mask.png",
    "frequency_spectrum.png",
    "frequency_highpass.png",
    "frequency_bandpass.png",
    "frequency_bandpass_difference.png",
    "defect_overlay.png",
]

REQUIRED_CASE_FILES = [
    "report.json",
    "evaluation.json",
    *CASE_IMAGE_FILES,
]

STAGE_DEFINITIONS = [
    {
        "id": "dataset",
        "title": "1. Dataset Input",
        "function": "run_pipeline() loads golden, test, and ROI files through load_image() and load_roi_json().",
        "input": [
            "Golden reference PCBA image in board pixel coordinates.",
            "Test PCBA image from a generated KiCad case.",
            "Component and solder ROI JSON in golden-image coordinates.",
            "Ground-truth JSON for quantitative evaluation.",
        ],
        "methods": [
            "Controlled KiCad rendering",
            "Pixel-coordinate ROI annotation",
            "Reference-based inspection setup",
        ],
        "output": [
            "Golden/test image pair ready for registration.",
            "ROI map containing expected component and solder-joint boxes.",
            "Ground truth used only for evaluation, not for detection.",
        ],
        "role": "Defines the measurable inspection problem: every later result is expressed in golden-image coordinates.",
        "images": ["golden", "test"],
    },
    {
        "id": "registration",
        "title": "2. Board Registration",
        "function": "run_pipeline() -> register_board()",
        "input": [
            "Golden image.",
            "Raw test image.",
        ],
        "methods": [
            "Board mask and edge extraction",
            "ORB keypoints and descriptor matching",
            "RANSAC homography",
            "Phase-correlation fallback",
            "Edge IoU and Chamfer alignment diagnostics",
        ],
        "output": [
            "Aligned test image in golden coordinates.",
            "Homography or translation transform metadata.",
            "Keypoint, match, checkerboard, edge-overlay, and difference visualizations.",
        ],
        "role": "Registration removes camera shift, scale, and rotation so ROI-level subtraction and template matching are meaningful.",
        "images": [
            "aligned_test.png",
            "alignment_matches_inliers.png",
            "alignment_checkerboard.png",
            "alignment_edge_overlay_red_test_green_golden.png",
        ],
    },
    {
        "id": "enhancement",
        "title": "3. Image Enhancement",
        "function": "enhance_and_segment()",
        "input": [
            "Golden image.",
            "Aligned test image.",
        ],
        "methods": [
            "LAB luminance CLAHE",
            "Bilateral filtering",
            "Color histogram matching",
            "Brightness and contrast normalization",
        ],
        "output": [
            "Enhanced golden/test arrays used internally by later stages.",
            "Normalized test image for mask generation and solder inspection.",
            "More stable appearance under lighting, contrast, and color changes.",
        ],
        "role": "Enhancement makes classical thresholds and local comparisons less sensitive to lighting and color variation.",
        "images": [
            "aligned_test.png",
            "alignment_difference_heatmap.png",
            "board_mask.png",
        ],
    },
    {
        "id": "frequency",
        "title": "4. Frequency-Domain Enhancement",
        "function": "log_magnitude_spectrum(), apply_frequency_filter(), frequency_difference_image()",
        "input": [
            "Enhanced golden grayscale image.",
            "Enhanced and normalized test grayscale image.",
        ],
        "methods": [
            "2D FFT",
            "Log-magnitude spectrum visualization",
            "High-pass filtering",
            "Band-pass filtering",
            "Golden/test band-pass difference map",
        ],
        "output": [
            "FFT spectrum image.",
            "High-pass edge/detail image.",
            "Band-pass component and solder detail image.",
            "Band-pass difference map highlighting local structural changes.",
        ],
        "role": "Shows how low-frequency illumination can be separated from mid/high-frequency component, pad, and solder detail.",
        "images": [
            "frequency_spectrum.png",
            "frequency_highpass.png",
            "frequency_bandpass.png",
            "frequency_bandpass_difference.png",
        ],
    },
    {
        "id": "segmentation",
        "title": "5. Segmentation and ROI Masking",
        "function": "enhance_and_segment() mask builders plus ROI constraints",
        "input": [
            "Enhanced and normalized test image.",
            "ROI map from the KiCad dataset.",
        ],
        "methods": [
            "Otsu thresholding",
            "Canny edge detection",
            "HSV/Lab solder-color segmentation",
            "Morphological opening and closing",
            "ROI-constrained masking",
        ],
        "output": [
            "Board mask.",
            "Component edge/detail mask.",
            "Solder or metallic-region mask.",
            "Masks clipped to known inspection ROIs.",
        ],
        "role": "Segmentation provides clean classical evidence for component and solder inspection without using deep learning.",
        "images": [
            "board_mask.png",
            "component_mask.png",
            "solder_mask.png",
        ],
    },
    {
        "id": "component",
        "title": "6. Component Inspection",
        "function": "inspect_components()",
        "input": [
            "Golden image.",
            "Aligned test image.",
            "Component ROIs.",
            "Segmentation result metadata.",
        ],
        "methods": [
            "Template matching",
            "Normalized correlation score",
            "Local absolute difference",
            "Canny edge density",
            "Contour orientation and displacement estimation",
            "Rule-based component labels",
        ],
        "output": [
            "Missing, shifted, rotated, or visual-mismatch component candidates.",
            "Per-candidate bbox, score, and interpretable features.",
        ],
        "role": "Detects assembly defects by comparing each expected component ROI against the aligned test image.",
        "images": [
            "component_mask.png",
            "defect_overlay.png",
        ],
    },
    {
        "id": "solder",
        "title": "7. Solder Inspection",
        "function": "inspect_solder_joints()",
        "input": [
            "Enhanced golden/test images.",
            "Solder-joint ROIs.",
            "Global solder mask.",
        ],
        "methods": [
            "HSV and Lab metallic/bright-region masks",
            "Adaptive local thresholding",
            "Morphological cleanup",
            "Connected components",
            "Area, width, shape, and bridge-connectivity features",
            "Rule-based solder labels",
        ],
        "output": [
            "Bridge, insufficient, excessive, missing, or abnormal solder candidates.",
            "Per-joint area and connectivity measurements.",
        ],
        "role": "Turns local solder appearance into interpretable defect candidates using color, morphology, and shape features.",
        "images": [
            "solder_mask.png",
            "defect_overlay.png",
        ],
    },
    {
        "id": "fusion",
        "title": "8. Fusion, Report, Evaluation",
        "function": "fuse_and_classify_defects(), evaluate_detections(), write_visual_outputs()",
        "input": [
            "Component candidates.",
            "Solder candidates.",
            "Segmentation masks.",
            "Ground truth for evaluation only.",
        ],
        "methods": [
            "Rule-based label retention",
            "Duplicate suppression by IoU",
            "Overlay rendering",
            "Precision, recall, F1, and false-positive accounting",
        ],
        "output": [
            "Final defect list.",
            "Defect overlay image.",
            "report.json and evaluation.json.",
            "Dataset summary metrics.",
        ],
        "role": "Produces the final AOI-style report and connects the pipeline to measurable course-project results.",
        "images": [
            "defect_overlay.png",
        ],
    },
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

    manifest_path_candidate = manifest_path.parent / path
    if manifest_path_candidate.exists():
        return manifest_path_candidate

    return cwd_path


def _relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


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


def _format_float(value: Any, digits: int = 3) -> str:
    return f"{_as_float(value):.{digits}f}"


def _is_successful_case(row: dict[str, str]) -> bool:
    if row.get("error"):
        return False

    truth_count = _as_int(row.get("truth_count"))
    prediction_count = _as_int(row.get("prediction_count"))
    false_positive = _as_int(row.get("false_positive"))
    false_negative = _as_int(row.get("false_negative"))
    f1_score = _as_float(row.get("f1_score"))

    if false_positive != 0 or false_negative != 0:
        return False

    if truth_count == 0:
        return prediction_count == 0

    return f1_score >= 0.999


def _case_files_exist(
    case: dict[str, Any],
    row: dict[str, str],
    manifest_path: Path,
    eval_dir: Path,
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    case_dir = eval_dir / "cases" / case["case_id"]

    for file_name in REQUIRED_CASE_FILES:
        if not (case_dir / file_name).exists():
            missing.append(str(case_dir / file_name))

    for input_key in ("golden", "test", "roi", "ground_truth"):
        if input_key not in case:
            missing.append(f"manifest:{case['case_id']}:{input_key}")
            continue
        resolved = _resolve_path(case[input_key], manifest_path)
        if not resolved.exists():
            missing.append(str(resolved))

    _ = row
    return not missing, missing


def _status_reason(
    case: dict[str, Any],
    row: dict[str, str] | None,
    manifest_path: Path,
    eval_dir: Path,
) -> str:
    if row is None:
        return "missing summary.csv row"
    if not _is_successful_case(row):
        return (
            "failed evaluation "
            f"(truth={row.get('truth_count')}, pred={row.get('prediction_count')}, "
            f"fp={row.get('false_positive')}, fn={row.get('false_negative')}, "
            f"f1={row.get('f1_score')}, error={row.get('error')})"
        )
    ok, missing = _case_files_exist(case, row, manifest_path, eval_dir)
    if not ok:
        return "missing required files: " + ", ".join(missing[:4])
    return ""


def _select_default_cases(
    cases_by_id: dict[str, dict[str, Any]],
    rows_by_id: dict[str, dict[str, str]],
    manifest_path: Path,
    eval_dir: Path,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for variant_type in DEFAULT_VARIANT_ORDER:
        candidates = [
            case
            for case in cases_by_id.values()
            if case.get("variant_type") == variant_type
        ]
        by_id = {case["case_id"]: case for case in candidates}
        ordered: list[dict[str, Any]] = []
        for case_id in PREFERRED_CASE_IDS.get(variant_type, []):
            if case_id in by_id:
                ordered.append(by_id[case_id])
        ordered.extend(
            sorted(
                [
                    case
                    for case in candidates
                    if case["case_id"] not in {item["case_id"] for item in ordered}
                ],
                key=lambda item: item["case_id"],
            )
        )

        for case in ordered:
            row = rows_by_id.get(case["case_id"])
            if row is None:
                continue
            if not _is_successful_case(row):
                continue
            ok, _ = _case_files_exist(case, row, manifest_path, eval_dir)
            if not ok:
                continue
            if case["case_id"] in selected_ids:
                continue
            selected.append(case)
            selected_ids.add(case["case_id"])
            break

    return selected


def _select_requested_cases(
    requested_ids: list[str],
    cases_by_id: dict[str, dict[str, Any]],
    rows_by_id: dict[str, dict[str, str]],
    manifest_path: Path,
    eval_dir: Path,
    include_failures: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for case_id in requested_ids:
        case = cases_by_id.get(case_id)
        if case is None:
            print(f"Skipping {case_id}: not found in manifest.")
            continue

        row = rows_by_id.get(case_id)
        reason = _status_reason(case, row, manifest_path, eval_dir)
        if reason and not include_failures:
            print(f"Skipping {case_id}: {reason}.")
            continue
        if row is None:
            print(f"Skipping {case_id}: missing summary.csv row.")
            continue

        ok, missing = _case_files_exist(case, row, manifest_path, eval_dir)
        if not ok:
            print(f"Skipping {case_id}: missing required files: {', '.join(missing[:4])}.")
            continue

        selected.append(case)
    return selected


def _image_label(file_name: str) -> str:
    labels = {
        "golden": "Golden Reference",
        "test": "Test Image",
        "aligned_test.png": "Aligned Test",
        "alignment_matches_inliers.png": "ORB Inlier Matches",
        "alignment_checkerboard.png": "Registration Checkerboard",
        "alignment_edge_overlay_red_test_green_golden.png": "Edge Overlay",
        "alignment_difference_heatmap.png": "Difference Heatmap",
        "board_mask.png": "Board Mask",
        "component_mask.png": "Component Mask",
        "solder_mask.png": "Solder Mask",
        "frequency_spectrum.png": "FFT Spectrum",
        "frequency_highpass.png": "FFT High-Pass",
        "frequency_bandpass.png": "FFT Band-Pass",
        "frequency_bandpass_difference.png": "FFT Difference",
        "defect_overlay.png": "Final Defect Overlay",
    }
    return labels.get(file_name, file_name)


def _image_caption(file_name: str) -> str:
    captions = {
        "golden": "Correct reference board image.",
        "test": "Generated test image for the selected case.",
        "aligned_test.png": "Test image warped into golden-image coordinates.",
        "alignment_matches_inliers.png": "Feature correspondences kept by RANSAC.",
        "alignment_checkerboard.png": "Alternating golden/test tiles reveal registration quality.",
        "alignment_edge_overlay_red_test_green_golden.png": "Red/green edge overlay; overlap indicates good alignment.",
        "alignment_difference_heatmap.png": "Pixel difference after registration.",
        "board_mask.png": "Localized board region after thresholding and morphology.",
        "component_mask.png": "Component/edge mask constrained by ROIs.",
        "solder_mask.png": "Bright low-saturation solder evidence.",
        "frequency_spectrum.png": "Log-magnitude FFT spectrum.",
        "frequency_highpass.png": "Fine edges and high-frequency detail.",
        "frequency_bandpass.png": "Mid-frequency component and solder structures.",
        "frequency_bandpass_difference.png": "Local detail changes between golden and test images.",
        "defect_overlay.png": "Final rule-based defect boxes and labels.",
    }
    return captions.get(file_name, "")


def _stage_image_items(
    stage: dict[str, Any],
    image_paths: dict[str, str],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for image_key in stage["images"]:
        path = image_paths.get(image_key)
        if not path:
            continue
        items.append(
            {
                "label": _image_label(image_key),
                "caption": _image_caption(image_key),
                "src": path,
            }
        )
    return items


def _top_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True)[0]


def _candidate_summary(candidate: dict[str, Any] | None) -> str:
    if candidate is None:
        return "None"
    score = candidate.get("score")
    score_text = "n/a" if score is None else _format_float(score)
    return f"{candidate.get('defect_type', 'unknown')} from {candidate.get('source', 'unknown')} (score {score_text})"


def _candidate_features(candidate: dict[str, Any] | None, keys: list[str]) -> list[dict[str, str]]:
    if candidate is None:
        return []
    features = candidate.get("features", {})
    rows: list[dict[str, str]] = []
    for key in keys:
        value = features.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            display = _format_float(value)
        else:
            display = str(value)
        rows.append({"label": key, "value": display})
    return rows


def _stage_metrics(
    stage_id: str,
    report: dict[str, Any],
    evaluation: dict[str, Any],
    summary_row: dict[str, str],
) -> list[dict[str, str]]:
    alignment = report.get("alignment", {})
    segmentation = report.get("segmentation", {})
    component_defects = report.get("component_defects", [])
    solder_defects = report.get("solder_defects", [])
    final_defects = report.get("defects", [])
    metrics = evaluation.get("metrics", {})
    truth = evaluation.get("ground_truth", [])
    evaluated_predictions = evaluation.get("evaluated_predictions", [])

    if stage_id == "dataset":
        return [
            {"label": "Case type", "value": evaluation.get("case", {}).get("variant_type", "")},
            {"label": "Ground-truth defects", "value": str(len(truth))},
            {"label": "Evaluation predictions", "value": str(len(evaluated_predictions))},
            {"label": "Board", "value": evaluation.get("case", {}).get("board_id", "")},
        ]

    if stage_id == "registration":
        return [
            {"label": "Method", "value": str(alignment.get("method", ""))},
            {"label": "Raw / inlier matches", "value": f"{alignment.get('raw_matches', 0)} / {alignment.get('inlier_matches', 0)}"},
            {"label": "Inlier ratio", "value": _format_float(alignment.get("inlier_ratio"))},
            {"label": "Reprojection p90", "value": f"{_format_float(alignment.get('reprojection_error_p90_px'))} px"},
            {"label": "Edge IoU", "value": _format_float(alignment.get("edge_iou_dilated"))},
            {"label": "Chamfer mean", "value": f"{_format_float(alignment.get('edge_chamfer_mean_px'))} px"},
        ]

    if stage_id == "enhancement":
        return [
            {"label": "Enhancement status", "value": str(segmentation.get("status", ""))},
            {"label": "Has ROI map", "value": str(segmentation.get("has_roi_map", ""))},
            {"label": "Mean board abs diff", "value": _format_float(alignment.get("mean_abs_difference_on_board"))},
            {"label": "Median board abs diff", "value": _format_float(alignment.get("median_abs_difference_on_board"))},
        ]

    if stage_id == "frequency":
        frequency = segmentation.get("frequency_domain", {})
        return [
            {"label": "Frequency enabled", "value": str(frequency.get("enabled", ""))},
            {"label": "High-pass cutoff", "value": str(frequency.get("highpass_cutoff", ""))},
            {"label": "Band-pass cutoff", "value": str(frequency.get("bandpass_cutoff", ""))},
            {"label": "Band-pass mean diff", "value": _format_float(summary_row.get("frequency_bandpass_mean"))},
            {"label": "GT ROI mean diff", "value": _format_float(summary_row.get("frequency_gt_mean"))},
        ]

    if stage_id == "segmentation":
        methods = segmentation.get("methods", [])
        return [
            {"label": "Mask source", "value": "ROI-constrained preprocessing masks"},
            {"label": "Method count", "value": str(len(methods))},
            {"label": "Board mask", "value": "board_mask.png"},
            {"label": "Component mask", "value": "component_mask.png"},
            {"label": "Solder mask", "value": "solder_mask.png"},
        ]

    if stage_id == "component":
        top = _top_candidate(component_defects)
        rows = [
            {"label": "Component candidates", "value": str(len(component_defects))},
            {"label": "Top candidate", "value": _candidate_summary(top)},
        ]
        rows.extend(
            _candidate_features(
                top,
                [
                    "component_id",
                    "template_score",
                    "direct_difference_score",
                    "displacement_px",
                    "edge_density_ratio",
                    "difference_fraction",
                ],
            )
        )
        return rows

    if stage_id == "solder":
        top = _top_candidate(solder_defects)
        rows = [
            {"label": "Solder candidates", "value": str(len(solder_defects))},
            {"label": "Top candidate", "value": _candidate_summary(top)},
        ]
        if top is not None:
            features = top.get("features", {})
            if "solder_joint_id" in features:
                rows.append({"label": "solder_joint_id", "value": str(features["solder_joint_id"])})
            if "component_id" in features:
                rows.append({"label": "component_id", "value": str(features["component_id"])})
            for key in ("area_delta_ratio", "local_mean_difference", "local_changed_fraction", "corridor_extra_area"):
                if key in features:
                    rows.append({"label": key, "value": _format_float(features[key])})
        return rows

    if stage_id == "fusion":
        return [
            {"label": "Final defects", "value": str(len(final_defects))},
            {"label": "TP / FP / FN", "value": f"{metrics.get('true_positive', 0)} / {metrics.get('false_positive', 0)} / {metrics.get('false_negative', 0)}"},
            {"label": "Precision", "value": _format_float(metrics.get("precision"))},
            {"label": "Recall", "value": _format_float(metrics.get("recall"))},
            {"label": "F1", "value": _format_float(metrics.get("f1_score"))},
        ]

    return []


def _compact_defect(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "defect_type": candidate.get("defect_type"),
        "bbox": candidate.get("bbox"),
        "source": candidate.get("source"),
        "score": candidate.get("score"),
        "features": candidate.get("features", {}),
    }


def _build_case_payload(
    case: dict[str, Any],
    summary_row: dict[str, str],
    manifest_path: Path,
    eval_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    case_id = case["case_id"]
    source_case_dir = eval_dir / "cases" / case_id
    destination_case_dir = output_root / "cases" / case_id
    destination_case_dir.mkdir(parents=True, exist_ok=True)

    golden_source = _resolve_path(case["golden"], manifest_path)
    test_source = _resolve_path(case["test"], manifest_path)
    golden_dest = destination_case_dir / "input_golden.png"
    test_dest = destination_case_dir / "input_test.png"
    _copy_file(golden_source, golden_dest)
    _copy_file(test_source, test_dest)

    image_paths = {
        "golden": _relative_to_root(golden_dest, output_root),
        "test": _relative_to_root(test_dest, output_root),
    }

    for file_name in CASE_IMAGE_FILES:
        source = source_case_dir / file_name
        destination = destination_case_dir / file_name
        _copy_file(source, destination)
        image_paths[file_name] = _relative_to_root(destination, output_root)

    report_source = source_case_dir / "report.json"
    evaluation_source = source_case_dir / "evaluation.json"
    report_dest = destination_case_dir / "report.json"
    evaluation_dest = destination_case_dir / "evaluation.json"
    _copy_file(report_source, report_dest)
    _copy_file(evaluation_source, evaluation_dest)

    report = _load_json(report_source)
    evaluation = _load_json(evaluation_source)

    stages = []
    for stage in STAGE_DEFINITIONS:
        stages.append(
            {
                **stage,
                "metrics": _stage_metrics(stage["id"], report, evaluation, summary_row),
                "image_items": _stage_image_items(stage, image_paths),
            }
        )

    return {
        "case_id": case_id,
        "board_id": case.get("board_id", ""),
        "variant_type": case.get("variant_type", ""),
        "truth_count": _as_int(summary_row.get("truth_count")),
        "prediction_count": _as_int(summary_row.get("prediction_count")),
        "true_positive": _as_int(summary_row.get("true_positive")),
        "false_positive": _as_int(summary_row.get("false_positive")),
        "false_negative": _as_int(summary_row.get("false_negative")),
        "precision": _as_float(summary_row.get("precision")),
        "recall": _as_float(summary_row.get("recall")),
        "f1_score": _as_float(summary_row.get("f1_score")),
        "success_display": _is_successful_case(summary_row),
        "image_paths": image_paths,
        "report_path": _relative_to_root(report_dest, output_root),
        "evaluation_path": _relative_to_root(evaluation_dest, output_root),
        "ground_truth": evaluation.get("ground_truth", []),
        "evaluated_predictions": evaluation.get("evaluated_predictions", []),
        "final_defects": [_compact_defect(item) for item in report.get("defects", [])],
        "stages": stages,
    }


def _html_template(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PCBA Pipeline Explainer Demo</title>
  <style>
    :root {{
      --bg: #f4f5f1;
      --panel: #ffffff;
      --ink: #222520;
      --muted: #62685e;
      --line: #d9ddd2;
      --accent: #166f65;
      --accent-soft: #e8f3ef;
      --blue: #234c7a;
      --warn: #a85820;
      --good: #2d7431;
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
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 16px 22px;
      position: sticky;
      top: 0;
      z-index: 4;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
    }}
    .subtitle {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
      max-width: 980px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 8px;
      max-width: 1160px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcf8;
      padding: 8px 10px;
      min-height: 56px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .metric strong {{
      font-size: 19px;
      line-height: 1;
    }}
    main {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
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
      top: 128px;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin: 10px 0 5px;
    }}
    select {{
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
      padding: 6px 8px;
      font-size: 14px;
    }}
    .stage-nav {{
      display: grid;
      gap: 6px;
      margin-top: 12px;
    }}
    .stage-button {{
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fbfcf8;
      color: var(--ink);
      text-align: left;
      padding: 7px 9px;
      font-size: 13px;
      cursor: pointer;
    }}
    .stage-button.active {{
      border-color: var(--accent);
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
    }}
    .case-card {{
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .case-card strong {{
      color: var(--ink);
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
      padding: 14px;
      min-width: 0;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.2;
    }}
    h3 {{
      margin: 14px 0 7px;
      font-size: 14px;
      color: var(--blue);
    }}
    p {{
      margin: 0;
      line-height: 1.45;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.45;
    }}
    code {{
      font-family: Consolas, Monaco, monospace;
      background: #f0f2ed;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
    }}
    .stage-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 14px;
      align-items: start;
    }}
    .explain-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .box {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcf8;
      min-width: 0;
    }}
    .box.full {{
      grid-column: 1 / -1;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .table th,
    .table td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      vertical-align: top;
    }}
    .table th {{
      color: var(--muted);
      font-weight: 600;
      width: 42%;
      background: #fafbf7;
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      margin-top: 12px;
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
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }}
    figure img {{
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #eef0eb;
    }}
    .caption {{
      padding: 7px 9px;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
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
      max-height: 330px;
      overflow: auto;
    }}
    .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      background: #fbfcf8;
      color: var(--muted);
      font-size: 12px;
      margin-right: 5px;
      margin-top: 5px;
    }}
    .pill.good {{
      color: var(--good);
      border-color: #b9d6bb;
      background: #eef7ee;
    }}
    @media (max-width: 980px) {{
      header {{ position: static; }}
      main {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .stage-layout {{ grid-template-columns: 1fr; }}
      .explain-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>PCBA Pipeline Explainer Demo</h1>
    <p class="subtitle">Classical reference-based PCBA inspection: input/output evidence and function-level explanation for each pipeline stage.</p>
    <div class="metric-grid" id="datasetMetrics"></div>
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
    <div class="content">
      <section id="stagePanel"></section>
      <section>
        <h2>Final Defect Report</h2>
        <div id="finalSummary"></div>
        <h3>Compact Final Defects</h3>
        <pre id="finalDefects"></pre>
      </section>
    </div>
  </main>
  <script>
    const DATA = {data_json};
    const caseSelect = document.getElementById('caseSelect');
    const stageSelect = document.getElementById('stageSelect');
    const stageNav = document.getElementById('stageNav');
    const caseCard = document.getElementById('caseCard');
    const datasetMetrics = document.getElementById('datasetMetrics');
    const stagePanel = document.getElementById('stagePanel');
    const finalSummary = document.getElementById('finalSummary');
    const finalDefects = document.getElementById('finalDefects');

    const fmt = (value, digits = 3) => {{
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits) : '0.000';
    }};

    function selectedCase() {{
      return DATA.cases.find((item) => item.case_id === caseSelect.value) || DATA.cases[0];
    }}

    function selectedStage(caseData) {{
      return caseData.stages.find((item) => item.id === stageSelect.value) || caseData.stages[0];
    }}

    function renderDatasetMetrics() {{
      const metrics = DATA.dataset_metrics || {{}};
      const overall = metrics.overall || {{}};
      const items = [
        ['Cases evaluated', metrics.case_count || DATA.case_count || 0],
        ['Demo cases', DATA.cases.length],
        ['Precision', fmt(overall.precision)],
        ['Recall', fmt(overall.recall)],
        ['F1 score', fmt(overall.f1_score)],
      ];
      datasetMetrics.innerHTML = items.map(([label, value]) => `
        <div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>
      `).join('');
    }}

    function fillControls() {{
      DATA.cases.forEach((item, index) => {{
        const option = document.createElement('option');
        option.value = item.case_id;
        option.textContent = `${{item.case_id}} · ${{item.variant_type}}`;
        caseSelect.appendChild(option);
        if (index === 0) option.selected = true;
      }});
      const first = DATA.cases[0];
      first.stages.forEach((stage, index) => {{
        const option = document.createElement('option');
        option.value = stage.id;
        option.textContent = stage.title;
        stageSelect.appendChild(option);
        if (index === 0) option.selected = true;
      }});
    }}

    function syncStageNav(caseData) {{
      stageNav.innerHTML = caseData.stages.map((stage) => `
        <button class="stage-button ${{stage.id === stageSelect.value ? 'active' : ''}}" data-stage="${{stage.id}}">
          ${{stage.title}}
        </button>
      `).join('');
      stageNav.querySelectorAll('button').forEach((button) => {{
        button.addEventListener('click', () => {{
          stageSelect.value = button.dataset.stage;
          render();
        }});
      }});
    }}

    function renderCaseCard(caseData) {{
      const success = caseData.success_display ? '<span class="pill good">selected success case</span>' : '<span class="pill">included manually</span>';
      caseCard.innerHTML = `
        <p><strong>${{caseData.case_id}}</strong></p>
        <p>${{caseData.board_id}} · ${{caseData.variant_type}}</p>
        <p>Truth / prediction: ${{caseData.truth_count}} / ${{caseData.prediction_count}}</p>
        <p>TP / FP / FN: ${{caseData.true_positive}} / ${{caseData.false_positive}} / ${{caseData.false_negative}}</p>
        <p>Precision / Recall / F1: ${{fmt(caseData.precision)}} / ${{fmt(caseData.recall)}} / ${{fmt(caseData.f1_score)}}</p>
        <div>${{success}}</div>
      `;
    }}

    function renderList(items) {{
      return `<ul>${{items.map((item) => `<li>${{item}}</li>`).join('')}}</ul>`;
    }}

    function renderMetrics(metrics) {{
      if (!metrics || !metrics.length) return '<p>No numeric evidence for this stage.</p>';
      return `<table class="table"><tbody>${{metrics.map((row) => `
        <tr><th>${{row.label}}</th><td>${{row.value}}</td></tr>
      `).join('')}}</tbody></table>`;
    }}

    function renderImages(stage) {{
      if (!stage.image_items || !stage.image_items.length) return '<p>No saved image artifact for this stage.</p>';
      return `<div class="image-grid">${{stage.image_items.map((item) => `
        <figure>
          <figcaption>${{item.label}}</figcaption>
          <img src="${{item.src}}" alt="${{item.label}}">
          <div class="caption">${{item.caption}}</div>
        </figure>
      `).join('')}}</div>`;
    }}

    function renderStage(caseData, stage) {{
      stagePanel.innerHTML = `
        <h2>${{stage.title}}</h2>
        <p>${{stage.role}}</p>
        <div class="stage-layout">
          <div class="explain-grid">
            <div class="box">
              <h3>Input</h3>
              ${{renderList(stage.input)}}
            </div>
            <div class="box">
              <h3>Function</h3>
              <p><code>${{stage.function}}</code></p>
            </div>
            <div class="box">
              <h3>DIP Methods</h3>
              ${{renderList(stage.methods)}}
            </div>
            <div class="box">
              <h3>Output</h3>
              ${{renderList(stage.output)}}
            </div>
            <div class="box full">
              <h3>Evidence</h3>
              ${{renderImages(stage)}}
            </div>
          </div>
          <div class="box">
            <h3>Stage Metrics</h3>
            ${{renderMetrics(stage.metrics)}}
          </div>
        </div>
      `;
      renderFinalReport(caseData);
    }}

    function renderFinalReport(caseData) {{
      const rows = [
        ['Variant type', caseData.variant_type],
        ['Ground truth defects', caseData.truth_count],
        ['Evaluated predictions', caseData.prediction_count],
        ['TP / FP / FN', `${{caseData.true_positive}} / ${{caseData.false_positive}} / ${{caseData.false_negative}}`],
        ['Precision / Recall / F1', `${{fmt(caseData.precision)}} / ${{fmt(caseData.recall)}} / ${{fmt(caseData.f1_score)}}`],
        ['report.json', caseData.report_path],
        ['evaluation.json', caseData.evaluation_path],
      ];
      finalSummary.innerHTML = `<table class="table"><tbody>${{rows.map(([label, value]) => `
        <tr><th>${{label}}</th><td>${{value}}</td></tr>
      `).join('')}}</tbody></table>`;
      finalDefects.textContent = JSON.stringify(caseData.final_defects || [], null, 2);
    }}

    function render() {{
      const caseData = selectedCase();
      if (!caseData.stages.some((stage) => stage.id === stageSelect.value)) {{
        stageSelect.value = caseData.stages[0].id;
      }}
      const stage = selectedStage(caseData);
      syncStageNav(caseData);
      renderCaseCard(caseData);
      renderStage(caseData, stage);
    }}

    renderDatasetMetrics();
    fillControls();
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
    metrics_path = eval_dir / "metrics.json"
    summary_path = eval_dir / "summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing evaluation summary CSV: {summary_path}")

    rows_by_id = _read_summary(summary_path)
    metrics = _load_json(metrics_path) if metrics_path.exists() else {
        "case_count": 0,
        "overall": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        "defect_free_false_positive_rate": 0.0,
    }

    cases_by_id = {
        case["case_id"]: case
        for case in manifest.get("cases", [])
    }

    if args.case_id:
        selected_cases = _select_requested_cases(
            args.case_id,
            cases_by_id,
            rows_by_id,
            manifest_path,
            eval_dir,
            args.include_failures,
        )
    else:
        selected_cases = _select_default_cases(
            cases_by_id,
            rows_by_id,
            manifest_path,
            eval_dir,
        )

    if not selected_cases:
        raise RuntimeError("No valid cases selected for the explainer demo.")

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
        "dataset_metrics": metrics,
        "case_count": len(manifest.get("cases", [])),
        "selection_policy": {
            "include_failures": bool(args.include_failures),
            "default_filters": [
                "no evaluation error",
                "no false positives",
                "no false negatives",
                "defect cases require F1 == 1",
                "defect-free cases require zero predictions",
                "all referenced images and JSON files exist",
            ],
        },
        "stages": STAGE_DEFINITIONS,
        "cases": demo_cases,
    }

    _write_json(output_root / "demo_data.json", data)
    html_path = output_root / "index.html"
    html_path.write_text(_html_template(data), encoding="utf-8")

    print("Selected demo cases:")
    for case in demo_cases:
        print(f"- {case['case_id']} ({case['variant_type']})")
    print(f"Wrote {html_path}")
    return html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a static PCBA pipeline explainer demo.")
    parser.add_argument("--manifest", default="data/kicad_synth/manifest.json", help="Dataset manifest JSON.")
    parser.add_argument("--eval-dir", default="outputs/kicad_synth_eval", help="Evaluation output directory.")
    parser.add_argument("--output-dir", default="outputs/pipeline_explainer_demo", help="Demo output directory.")
    parser.add_argument("--case-id", nargs="*", default=None, help="Specific case ids to include.")
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Allow requested cases that are not clean success cases.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_demo(args)


if __name__ == "__main__":
    main()
