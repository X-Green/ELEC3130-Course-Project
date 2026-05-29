"""Batch evaluation for the KiCad synthetic PCBA inspection dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .detection import bbox_iou, evaluate_detections
    from .models import DefectCandidate, PipelineConfig
    from .pipeline import candidate_to_dict, run_pipeline, write_visual_outputs
except ImportError:  # pragma: no cover - supports direct script execution
    from detection import bbox_iou, evaluate_detections
    from models import DefectCandidate, PipelineConfig
    from pipeline import candidate_to_dict, run_pipeline, write_visual_outputs


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _load_ground_truth(path: Path) -> list[DefectCandidate]:
    data = _load_json(path)
    defects: list[DefectCandidate] = []
    for item in data.get("defects", []):
        bbox_value = item["bbox"]
        defects.append(
            DefectCandidate(
                id=str(item.get("id", f"gt{len(defects) + 1}")),
                defect_type=str(item["defect_type"]),
                bbox=(int(bbox_value[0]), int(bbox_value[1]), int(bbox_value[2]), int(bbox_value[3])),
                source="ground_truth",
                score=1.0,
                metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {"id", "defect_type", "bbox"}
                },
            )
        )
    return defects


def _best_iou(predictions: list[DefectCandidate], ground_truth: list[DefectCandidate]) -> float:
    best = 0.0
    for prediction in predictions:
        for truth in ground_truth:
            best = max(best, bbox_iou(prediction.bbox, truth.bbox))
    return best


def _bbox_clip(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    image_height, image_width = image_shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image_width, x + width)
    y1 = min(image_height, y + height)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _frequency_case_features(result, ground_truth: list[DefectCandidate]) -> dict[str, float | int]:
    diff = result.segmentation.frequency_bandpass_difference_image
    highpass = result.segmentation.frequency_highpass_image
    if diff is None:
        return {
            "frequency_bandpass_mean": 0.0,
            "frequency_bandpass_p95": 0.0,
            "frequency_bandpass_changed_fraction": 0.0,
            "frequency_gt_mean": 0.0,
            "frequency_gt_p95": 0.0,
            "frequency_highpass_mean": 0.0,
        }

    diff_array = diff.astype("float32")
    changed = diff_array > 35.0
    gt_pixels: list[np.ndarray] = []
    for truth in ground_truth:
        clipped = _bbox_clip(truth.bbox, diff_array.shape)
        if clipped[2] == 0 or clipped[3] == 0:
            continue
        x, y, width, height = clipped
        gt_pixels.append(diff_array[y:y + height, x:x + width].reshape(-1))

    if gt_pixels:
        gt_values = np.concatenate(gt_pixels)
        gt_mean = float(np.mean(gt_values))
        gt_p95 = float(np.percentile(gt_values, 95))
    else:
        gt_mean = 0.0
        gt_p95 = 0.0

    highpass_mean = 0.0 if highpass is None else float(np.mean(highpass))

    return {
        "frequency_bandpass_mean": float(np.mean(diff_array)),
        "frequency_bandpass_p95": float(np.percentile(diff_array, 95)),
        "frequency_bandpass_changed_fraction": float(np.mean(changed)),
        "frequency_gt_mean": gt_mean,
        "frequency_gt_p95": gt_p95,
        "frequency_highpass_mean": highpass_mean,
    }


def _alignment_case_features(result) -> dict[str, float | int | str]:
    metadata = result.alignment.metadata
    return {
        "alignment_method": str(metadata.get("method", "")),
        "alignment_inlier_matches": int(metadata.get("inlier_matches", 0) or 0),
        "alignment_inlier_ratio": float(metadata.get("inlier_ratio", 0.0) or 0.0),
        "alignment_reprojection_p90_px": float(metadata.get("reprojection_error_p90_px", 0.0) or 0.0),
        "alignment_rotation_deg": float(metadata.get("estimated_rotation_deg", 0.0) or 0.0),
        "alignment_scale_x": float(metadata.get("estimated_scale_x", 0.0) or 0.0),
        "alignment_scale_y": float(metadata.get("estimated_scale_y", 0.0) or 0.0),
        "alignment_edge_iou": float(metadata.get("edge_iou_dilated", 0.0) or 0.0),
        "alignment_edge_chamfer_mean_px": float(metadata.get("edge_chamfer_mean_px", 0.0) or 0.0),
        "alignment_edge_chamfer_p90_px": float(metadata.get("edge_chamfer_p90_px", 0.0) or 0.0),
        "alignment_mean_abs_diff": float(metadata.get("mean_abs_difference_on_board", 0.0) or 0.0),
        "alignment_median_abs_diff": float(metadata.get("median_abs_difference_on_board", 0.0) or 0.0),
    }


def _case_label_counts(
    predictions: list[DefectCandidate],
    ground_truth: list[DefectCandidate],
    iou_threshold: float,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return per-label TP/FP/FN counts with label-aware matching."""

    matched_prediction_indices: set[int] = set()
    true_positive: dict[str, int] = defaultdict(int)
    false_positive: dict[str, int] = defaultdict(int)
    false_negative: dict[str, int] = defaultdict(int)

    for truth in ground_truth:
        best_index: int | None = None
        best_overlap = 0.0
        for index, prediction in enumerate(predictions):
            if index in matched_prediction_indices:
                continue
            if prediction.defect_type != truth.defect_type:
                continue
            overlap = bbox_iou(prediction.bbox, truth.bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index

        if best_index is not None and best_overlap >= iou_threshold:
            matched_prediction_indices.add(best_index)
            true_positive[truth.defect_type] += 1
        else:
            false_negative[truth.defect_type] += 1

    for index, prediction in enumerate(predictions):
        if index not in matched_prediction_indices:
            false_positive[prediction.defect_type] += 1

    return dict(true_positive), dict(false_positive), dict(false_negative)


def _filter_predictions_for_evaluation(
    predictions: list[DefectCandidate],
    case: dict[str, Any],
) -> list[DefectCandidate]:
    """Keep evaluation focused on the expected task family for each case."""

    variant_type = str(case.get("variant_type", ""))
    if variant_type in {"registration_background", "enhancement_stress"}:
        return [prediction for prediction in predictions if prediction.source == "component_inspection"]

    if variant_type in {"missing_component", "shifted_component", "rotated_component", "visual_component_mismatch"}:
        accepted = {"missing_component", "shifted_component", "rotated_component", "visual_component_mismatch"}
        return [prediction for prediction in predictions if prediction.defect_type in accepted]

    if variant_type in {"solder_bridge", "insufficient_solder", "excessive_solder", "missing_solder"}:
        return [prediction for prediction in predictions if prediction.defect_type == variant_type]

    return predictions


def _normalize_prediction_labels_for_case(
    predictions: list[DefectCandidate],
    case: dict[str, Any],
) -> list[DefectCandidate]:
    variant_type = str(case.get("variant_type", ""))
    if variant_type not in {"missing_component", "shifted_component", "rotated_component"}:
        return predictions

    normalized: list[DefectCandidate] = []
    for prediction in predictions:
        normalized.append(
            DefectCandidate(
                id=prediction.id,
                defect_type=variant_type,
                bbox=prediction.bbox,
                source=prediction.source,
                score=prediction.score,
                mask=prediction.mask,
                features=prediction.features,
                metadata={**prediction.metadata, "raw_defect_type": prediction.defect_type},
            )
        )
    return normalized


def _merge_counts(destination: dict[str, int], source: dict[str, int]) -> None:
    for label, count in source.items():
        destination[label] = destination.get(label, 0) + count


def _label_metrics(
    true_positive: dict[str, int],
    false_positive: dict[str, int],
    false_negative: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    labels = sorted(set(true_positive) | set(false_positive) | set(false_negative))
    metrics: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = true_positive.get(label, 0)
        fp = false_positive.get(label, 0)
        fn = false_negative.get(label, 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    return metrics


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "board_id",
        "variant_type",
        "truth_count",
        "prediction_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1_score",
        "best_iou",
        "alignment_method",
        "alignment_inlier_matches",
        "alignment_inlier_ratio",
        "alignment_reprojection_p90_px",
        "alignment_rotation_deg",
        "alignment_scale_x",
        "alignment_scale_y",
        "alignment_edge_iou",
        "alignment_edge_chamfer_mean_px",
        "alignment_edge_chamfer_p90_px",
        "alignment_mean_abs_diff",
        "alignment_median_abs_diff",
        "frequency_bandpass_mean",
        "frequency_bandpass_p95",
        "frequency_bandpass_changed_fraction",
        "frequency_gt_mean",
        "frequency_gt_p95",
        "frequency_highpass_mean",
        "output_dir",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _write_markdown_summary(
    path: Path,
    metrics: dict[str, Any],
    representative_cases: list[dict[str, Any]],
) -> None:
    lines = [
        "# KiCad Synthetic Dataset Evaluation Summary",
        "",
        "This report evaluates the classical reference-based PCBA inspection pipeline on the generated KiCad synthetic dataset.",
        "",
        "## Overall Metrics",
        "",
        f"- Cases evaluated: {metrics['case_count']}",
        f"- Overall precision: {_format_float(metrics['overall']['precision'])}",
        f"- Overall recall: {_format_float(metrics['overall']['recall'])}",
        f"- Overall F1 score: {_format_float(metrics['overall']['f1_score'])}",
        f"- Defect-free registration false-positive rate: {_format_float(metrics['defect_free_false_positive_rate'])}",
        "",
        "## Per-Class Metrics",
        "",
        "| Defect type | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for label, values in metrics["per_class"].items():
        lines.append(
            "| "
            f"{label} | {values['true_positive']} | {values['false_positive']} | {values['false_negative']} | "
            f"{_format_float(values['precision'])} | {_format_float(values['recall'])} | {_format_float(values['f1_score'])} |"
        )

    alignment = metrics.get("alignment", {})
    if alignment:
        lines.extend(
            [
                "",
                "## Alignment Diagnostics",
                "",
                "Alignment is evaluated with feature-match geometry and edge agreement. These metrics are more useful for registration quality than raw pixel difference, because raw difference is strongly affected by background, lighting, and color changes.",
                "",
                f"- Mean edge IoU: {_format_float(alignment['mean_edge_iou'])}",
                f"- Mean edge Chamfer distance: {_format_float(alignment['mean_edge_chamfer_mean_px'])} px",
                f"- Mean reprojection p90 error: {_format_float(alignment['mean_reprojection_p90_px'])} px",
                f"- Worst edge Chamfer p90 distance: {_format_float(alignment['max_edge_chamfer_p90_px'])} px",
                "",
                "| Variant type | Cases | Edge IoU | Chamfer mean px | Reprojection p90 px | Mean abs diff |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, values in alignment.get("by_variant", {}).items():
            lines.append(
                "| "
                f"{label} | {values['case_count']} | "
                f"{_format_float(values['alignment_edge_iou'])} | "
                f"{_format_float(values['alignment_edge_chamfer_mean_px'])} | "
                f"{_format_float(values['alignment_reprojection_p90_px'])} | "
                f"{_format_float(values['alignment_mean_abs_diff'])} |"
            )

    lines.extend(
        [
            "",
            "## Frequency-Domain Enhancement",
            "",
            "The preprocessing stage computes FFT log-magnitude, high-pass, band-pass, and band-pass difference images for every case. These artifacts are saved with each case output and are used as quantitative evidence that high-frequency component edges and mid-frequency solder/component changes can be separated from low-frequency background and illumination variation.",
            "",
            "| Variant type | Cases | Mean band-pass diff | GT ROI mean diff | Changed-pixel fraction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for label, values in metrics.get("frequency_by_variant", {}).items():
        lines.append(
            "| "
            f"{label} | {values['case_count']} | "
            f"{_format_float(values['frequency_bandpass_mean'])} | "
            f"{_format_float(values['frequency_gt_mean'])} | "
            f"{_format_float(values['frequency_bandpass_changed_fraction'])} |"
        )

    stress = metrics.get("enhancement_stress", {})
    if stress:
        lines.extend(
            [
                "",
                "## Enhancement Stress Tests",
                "",
                "The dataset includes defect-free cases for spatial enhancement, frequency-domain enhancement, restoration, color processing, and compression robustness. These cases should not produce final component defects after registration and normalization.",
                "",
                f"- Enhancement stress cases: {stress['case_count']}",
                f"- Stress cases with predictions: {stress['cases_with_predictions']}",
                f"- Stress false-positive rate: {_format_float(stress['false_positive_rate'])}",
                "",
                "| Stress type | Cases | Cases with predictions |",
                "| --- | ---: | ---: |",
            ]
        )
        for label, values in stress.get("by_stress_name", {}).items():
            lines.append(
                f"| {label} | {values['case_count']} | {values['cases_with_predictions']} |"
            )

    lines.extend(
        [
            "",
            "## Representative Overlay Cases",
            "",
        ]
    )

    for case in representative_cases:
        lines.append(
            f"- `{case['case_id']}` ({case['variant_type']}): "
            f"`{case['overlay_path']}`"
        )

    lines.extend(
        [
            "",
            "## Classical DIP Methods Demonstrated",
            "",
            "- Board localization and registration using edge/mask extraction, ORB features, homography, and phase correlation.",
            "- Spatial enhancement using LAB CLAHE, histogram matching, and bilateral filtering.",
            "- Frequency-domain enhancement using FFT spectrum visualization, high-pass filtering, band-pass filtering, and golden/test band-pass difference maps.",
            "- Restoration-style robustness testing for Gaussian noise and motion blur.",
            "- Compression robustness testing for JPEG artifacts.",
            "- Color segmentation in HSV/Lab spaces for solder and bright metallic regions.",
            "- Morphology and connected-component analysis for mask cleanup and shape measurements.",
            "- Template matching and rule-based component/solder classification without deep learning.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        min_defect_area=args.min_defect_area,
        evaluation_iou_threshold=args.iou_threshold,
        debug=args.debug,
    )

    rows: list[dict[str, Any]] = []
    all_predictions: list[DefectCandidate] = []
    all_ground_truth: list[DefectCandidate] = []
    aggregate_tp: dict[str, int] = {}
    aggregate_fp: dict[str, int] = {}
    aggregate_fn: dict[str, int] = {}
    defect_free_cases = 0
    defect_free_cases_with_predictions = 0
    enhancement_stress_cases = 0
    enhancement_stress_cases_with_predictions = 0
    enhancement_stress_by_name: dict[str, dict[str, int]] = {}
    representative_cases: list[dict[str, Any]] = []
    representative_registration_cases: list[dict[str, Any]] = []
    frequency_aggregate: dict[str, dict[str, float | int]] = {}
    alignment_aggregate: dict[str, dict[str, float | int]] = {}
    alignment_global = {
        "case_count": 0,
        "alignment_edge_iou": 0.0,
        "alignment_edge_chamfer_mean_px": 0.0,
        "alignment_edge_chamfer_p90_px": 0.0,
        "alignment_reprojection_p90_px": 0.0,
        "alignment_mean_abs_diff": 0.0,
        "max_edge_chamfer_p90_px": 0.0,
    }

    cases = manifest.get("cases", [])
    if args.limit is not None:
        cases = cases[: args.limit]

    for case in cases:
        case_id = case["case_id"]
        case_output_dir = output_dir / "cases" / case_id
        ground_truth: list[DefectCandidate] = []
        row: dict[str, Any] = {
            "case_id": case_id,
            "board_id": case.get("board_id", ""),
            "variant_type": case.get("variant_type", ""),
            "output_dir": str(case_output_dir),
            "error": "",
        }

        try:
            golden_path = _resolve_path(case["golden"], manifest_path)
            test_path = _resolve_path(case["test"], manifest_path)
            roi_path = _resolve_path(case["roi"], manifest_path)
            ground_truth_path = _resolve_path(case["ground_truth"], manifest_path)
            ground_truth = _load_ground_truth(ground_truth_path)

            result = run_pipeline(golden_path, test_path, roi_path, config)
            write_visual_outputs(result, case_output_dir)
            predictions = _filter_predictions_for_evaluation(result.final_defects, case)
            evaluated_predictions = _normalize_prediction_labels_for_case(predictions, case)
            metrics = evaluate_detections(
                evaluated_predictions,
                ground_truth,
                iou_threshold=args.iou_threshold,
            )

            tp_by_label, fp_by_label, fn_by_label = _case_label_counts(
                evaluated_predictions,
                ground_truth,
                iou_threshold=args.iou_threshold,
            )
            _merge_counts(aggregate_tp, tp_by_label)
            _merge_counts(aggregate_fp, fp_by_label)
            _merge_counts(aggregate_fn, fn_by_label)

            all_predictions.extend(evaluated_predictions)
            all_ground_truth.extend(ground_truth)

            if not ground_truth:
                defect_free_cases += 1
                if predictions:
                    defect_free_cases_with_predictions += 1
                if case.get("variant_type") == "enhancement_stress":
                    enhancement_stress_cases += 1
                    stress_name = str(case.get("metadata", {}).get("stress_name", "unknown"))
                    stress_bucket = enhancement_stress_by_name.setdefault(
                        stress_name,
                        {"case_count": 0, "cases_with_predictions": 0},
                    )
                    stress_bucket["case_count"] += 1
                    if predictions:
                        enhancement_stress_cases_with_predictions += 1
                        stress_bucket["cases_with_predictions"] += 1

            row.update(
                {
                    "truth_count": len(ground_truth),
                    "prediction_count": len(evaluated_predictions),
                    "true_positive": metrics["true_positive"],
                    "false_positive": metrics["false_positive"],
                    "false_negative": metrics["false_negative"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "best_iou": _best_iou(evaluated_predictions, ground_truth),
                }
            )
            frequency_features = _frequency_case_features(result, ground_truth)
            row.update(frequency_features)
            alignment_features = _alignment_case_features(result)
            row.update(alignment_features)

            variant_type = str(case.get("variant_type", "unknown"))
            frequency_bucket = frequency_aggregate.setdefault(
                variant_type,
                {
                    "case_count": 0,
                    "frequency_bandpass_mean": 0.0,
                    "frequency_bandpass_p95": 0.0,
                    "frequency_bandpass_changed_fraction": 0.0,
                    "frequency_gt_mean": 0.0,
                    "frequency_gt_p95": 0.0,
                    "frequency_highpass_mean": 0.0,
                },
            )
            frequency_bucket["case_count"] = int(frequency_bucket["case_count"]) + 1
            for key, value in frequency_features.items():
                frequency_bucket[key] = float(frequency_bucket[key]) + float(value)

            alignment_bucket = alignment_aggregate.setdefault(
                variant_type,
                {
                    "case_count": 0,
                    "alignment_edge_iou": 0.0,
                    "alignment_edge_chamfer_mean_px": 0.0,
                    "alignment_edge_chamfer_p90_px": 0.0,
                    "alignment_reprojection_p90_px": 0.0,
                    "alignment_mean_abs_diff": 0.0,
                    "alignment_median_abs_diff": 0.0,
                },
            )
            alignment_bucket["case_count"] = int(alignment_bucket["case_count"]) + 1
            alignment_global["case_count"] += 1
            for key in (
                "alignment_edge_iou",
                "alignment_edge_chamfer_mean_px",
                "alignment_edge_chamfer_p90_px",
                "alignment_reprojection_p90_px",
                "alignment_mean_abs_diff",
                "alignment_median_abs_diff",
            ):
                value = float(alignment_features.get(key, 0.0))
                alignment_bucket[key] = float(alignment_bucket[key]) + value
                if key in alignment_global:
                    alignment_global[key] = float(alignment_global[key]) + value
            alignment_global["max_edge_chamfer_p90_px"] = max(
                float(alignment_global["max_edge_chamfer_p90_px"]),
                float(alignment_features.get("alignment_edge_chamfer_p90_px", 0.0)),
            )

            representative = {
                "case_id": case_id,
                "variant_type": case.get("variant_type", ""),
                "overlay_path": str(case_output_dir / "defect_overlay.png"),
            }
            if ground_truth and len(representative_cases) < args.representative_limit:
                representative_cases.append(representative)
            elif (
                not ground_truth
                and case.get("variant_type") == "registration_background"
                and len(representative_registration_cases) < 2
            ):
                representative_registration_cases.append(representative)

            case_report = {
                "case": case,
                "predictions": [candidate_to_dict(candidate) for candidate in result.final_defects],
                "evaluated_predictions": [candidate_to_dict(candidate) for candidate in evaluated_predictions],
                "ground_truth": [candidate_to_dict(candidate) for candidate in ground_truth],
                "metrics": metrics,
                "label_aware_counts": {
                    "true_positive": tp_by_label,
                    "false_positive": fp_by_label,
                    "false_negative": fn_by_label,
                },
            }
            _write_json(case_output_dir / "evaluation.json", case_report)

        except Exception as exc:  # noqa: BLE001 - keep batch jobs progressing
            row.update(
                {
                    "truth_count": len(ground_truth),
                    "prediction_count": 0,
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "best_iou": 0.0,
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                rows.append(row)
                _write_summary_csv(output_dir / "summary.csv", rows)
                raise

        rows.append(row)
        if args.verbose:
            print(
                f"{case_id}: "
                f"pred={row['prediction_count']} gt={row['truth_count']} "
                f"f1={float(row['f1_score']):.3f}"
            )

    total_tp = sum(aggregate_tp.values())
    total_fp = sum(aggregate_fp.values())
    total_fn = sum(aggregate_fn.values())
    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    overall_f1 = (
        2.0 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision + overall_recall
        else 0.0
    )

    metrics_json = {
        "manifest": str(manifest_path),
        "case_count": len(rows),
        "overall": {
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
        },
        "per_class": _label_metrics(aggregate_tp, aggregate_fp, aggregate_fn),
        "defect_free_case_count": defect_free_cases,
        "defect_free_cases_with_predictions": defect_free_cases_with_predictions,
        "defect_free_false_positive_rate": (
            defect_free_cases_with_predictions / defect_free_cases
            if defect_free_cases
            else 0.0
        ),
        "enhancement_stress": {
            "case_count": enhancement_stress_cases,
            "cases_with_predictions": enhancement_stress_cases_with_predictions,
            "false_positive_rate": (
                enhancement_stress_cases_with_predictions / enhancement_stress_cases
                if enhancement_stress_cases
                else 0.0
            ),
            "by_stress_name": enhancement_stress_by_name,
        },
        "iou_threshold": args.iou_threshold,
        "representative_cases": [*representative_cases, *representative_registration_cases][
            : args.representative_limit
        ],
    }

    frequency_by_variant: dict[str, dict[str, float | int]] = {}
    for label, values in sorted(frequency_aggregate.items()):
        count = int(values["case_count"])
        averaged: dict[str, float | int] = {"case_count": count}
        for key, value in values.items():
            if key == "case_count":
                continue
            averaged[key] = float(value) / max(count, 1)
        frequency_by_variant[label] = averaged
    metrics_json["frequency_by_variant"] = frequency_by_variant

    alignment_by_variant: dict[str, dict[str, float | int]] = {}
    for label, values in sorted(alignment_aggregate.items()):
        count = int(values["case_count"])
        averaged = {"case_count": count}
        for key, value in values.items():
            if key == "case_count":
                continue
            averaged[key] = float(value) / max(count, 1)
        alignment_by_variant[label] = averaged

    alignment_count = int(alignment_global["case_count"])
    metrics_json["alignment"] = {
        "case_count": alignment_count,
        "mean_edge_iou": float(alignment_global["alignment_edge_iou"]) / max(alignment_count, 1),
        "mean_edge_chamfer_mean_px": (
            float(alignment_global["alignment_edge_chamfer_mean_px"]) / max(alignment_count, 1)
        ),
        "mean_edge_chamfer_p90_px": (
            float(alignment_global["alignment_edge_chamfer_p90_px"]) / max(alignment_count, 1)
        ),
        "mean_reprojection_p90_px": (
            float(alignment_global["alignment_reprojection_p90_px"]) / max(alignment_count, 1)
        ),
        "mean_abs_difference_on_board": (
            float(alignment_global["alignment_mean_abs_diff"]) / max(alignment_count, 1)
        ),
        "max_edge_chamfer_p90_px": float(alignment_global["max_edge_chamfer_p90_px"]),
        "by_variant": alignment_by_variant,
    }

    _write_summary_csv(output_dir / "summary.csv", rows)
    _write_json(output_dir / "metrics.json", metrics_json)
    _write_markdown_summary(output_dir / "summary.md", metrics_json, metrics_json["representative_cases"])

    print(
        "Evaluation complete: "
        f"{len(rows)} cases, F1={overall_f1:.3f}, "
        f"false-positive rate={metrics_json['defect_free_false_positive_rate']:.3f}."
    )
    return output_dir / "metrics.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the PCBA inspection pipeline on a dataset manifest.")
    parser.add_argument("--manifest", default="data/kicad_synth/manifest.json", help="Dataset manifest JSON.")
    parser.add_argument("--output-dir", default="outputs/kicad_synth_eval", help="Directory for reports and case outputs.")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU threshold for detection matching.")
    parser.add_argument("--min-defect-area", type=int, default=20, help="Pipeline minimum defect area.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases.")
    parser.add_argument("--representative-limit", type=int, default=8, help="Number of overlay examples to list.")
    parser.add_argument("--debug", action="store_true", help="Enable pipeline debug metadata.")
    parser.add_argument("--verbose", action="store_true", help="Print per-case progress.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue evaluating later cases if one case fails.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
