"""Task 5: defect fusion, rule-based classification, and evaluation hooks."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    from .models import DefectCandidate, PipelineConfig, ROIMap, SegmentationResult
except ImportError:  # pragma: no cover - supports direct script execution
    from models import DefectCandidate, PipelineConfig, ROIMap, SegmentationResult


def bbox_iou(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax + aw, bx + bw)
    inter_y1 = min(ay + ah, by + bh)

    inter_width = max(0, inter_x1 - inter_x0)
    inter_height = max(0, inter_y1 - inter_y0)
    inter_area = inter_width * inter_height

    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0
    return float(inter_area / union_area)


def _candidate_priority(candidate: DefectCandidate) -> tuple[int, float]:
    source_priority = {
        "component_inspection": 2,
        "solder_inspection": 1,
    }.get(candidate.source, 0)
    return source_priority, float(candidate.score or 0.0)


def _deduplicate_candidates(
    candidates: list[DefectCandidate],
    iou_threshold: float,
) -> list[DefectCandidate]:
    ordered = sorted(candidates, key=_candidate_priority, reverse=True)
    kept: list[DefectCandidate] = []

    for candidate in ordered:
        duplicate_index: int | None = None
        for index, existing in enumerate(kept):
            same_type = candidate.defect_type == existing.defect_type
            same_region = bbox_iou(candidate.bbox, existing.bbox) >= iou_threshold
            if same_type and same_region:
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(candidate)
            continue

        existing = kept[duplicate_index]
        candidate_priority = _candidate_priority(candidate)
        existing_priority = _candidate_priority(existing)
        if candidate_priority > existing_priority:
            candidate.metadata["merged_duplicate"] = existing.id
            kept[duplicate_index] = candidate
        else:
            existing.metadata.setdefault("merged_duplicates", []).append(candidate.id)

    return kept


def _mask_to_candidates(
    mask: np.ndarray,
    source: str,
    defect_type: str,
    min_area: int,
) -> list[DefectCandidate]:
    if mask is None:
        return []

    mask = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[DefectCandidate] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        candidates.append(
            DefectCandidate(
                id=f"{source}:{defect_type}:{label}",
                defect_type=defect_type,
                bbox=(x, y, width, height),
                source=source,
                score=min(1.0, area / max(width * height, 1)),
                features={"area": area},
            )
        )

    return candidates


def fuse_and_classify_defects(
    component_candidates: list[DefectCandidate],
    solder_candidates: list[DefectCandidate],
    segmentation: SegmentationResult,
    roi_map: ROIMap,
    config: PipelineConfig | None = None,
) -> list[DefectCandidate]:
    """Merge candidates and assign final rule-based defect labels."""

    active_config = config or PipelineConfig()
    _ = roi_map

    candidates = [*component_candidates, *solder_candidates]

    if (
        active_config.enable_segmentation_fallback
        and not candidates
        and segmentation.component_mask is not None
        and segmentation.solder_mask is not None
    ):
        component_only = cv2.subtract(segmentation.component_mask, segmentation.solder_mask)
        candidates.extend(
            _mask_to_candidates(
                component_only,
                "segmentation",
                "unclassified_visual_difference",
                active_config.min_defect_area,
            )
        )

    final_candidates = _deduplicate_candidates(
        candidates,
        iou_threshold=0.40,
    )

    for index, candidate in enumerate(final_candidates, start=1):
        candidate.metadata["final_rank"] = index
        candidate.metadata["rule_based_label"] = candidate.defect_type

    return final_candidates


def evaluate_detections(
    predictions: list[DefectCandidate],
    ground_truth: list[DefectCandidate],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute precision, recall, F1, and a simple confusion matrix."""

    matched_ground_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    false_positives: list[int] = []

    for prediction_index, prediction in enumerate(predictions):
        best_index: int | None = None
        best_iou = 0.0

        for truth_index, truth in enumerate(ground_truth):
            if truth_index in matched_ground_truth:
                continue
            overlap = bbox_iou(prediction.bbox, truth.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_index = truth_index

        if best_index is None or best_iou < iou_threshold:
            false_positives.append(prediction_index)
            continue

        matched_ground_truth.add(best_index)
        matches.append((prediction_index, best_index, best_iou))

    false_negatives = [
        truth_index
        for truth_index in range(len(ground_truth))
        if truth_index not in matched_ground_truth
    ]

    true_positive = len(matches)
    false_positive = len(false_positives)
    false_negative = len(false_negatives)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    labels = sorted(
        {
            candidate.defect_type
            for candidate in [*predictions, *ground_truth]
        }
    )
    confusion_matrix = {
        truth_label: {prediction_label: 0 for prediction_label in [*labels, "missed"]}
        for truth_label in labels
    }
    confusion_matrix["background"] = {prediction_label: 0 for prediction_label in labels}
    confusion_matrix["background"]["missed"] = 0

    for prediction_index, truth_index, _ in matches:
        truth_label = ground_truth[truth_index].defect_type
        prediction_label = predictions[prediction_index].defect_type
        confusion_matrix[truth_label][prediction_label] += 1

    for prediction_index in false_positives:
        prediction_label = predictions[prediction_index].defect_type
        confusion_matrix["background"][prediction_label] += 1

    for truth_index in false_negatives:
        truth_label = ground_truth[truth_index].defect_type
        confusion_matrix[truth_label]["missed"] += 1

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matches": matches,
        "false_positive_indices": false_positives,
        "false_negative_indices": false_negatives,
        "confusion_matrix": confusion_matrix,
        "iou_threshold": iou_threshold,
    }
