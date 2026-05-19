"""Task 4: solder joint inspection."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .models import DefectCandidate, ImageArray, PipelineConfig, ROIMap, SegmentationResult
except ImportError:  # pragma: no cover - supports direct script execution
    from models import DefectCandidate, ImageArray, PipelineConfig, ROIMap, SegmentationResult


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    max_height, max_width = image_shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(max_width, x + width)
    y1 = min(max_height, y + height)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    return image[y:y + height, x:x + width]


def _ensure_color(image: ImageArray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image *= 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def _local_solder_mask(color_roi: np.ndarray) -> np.ndarray:
    if color_roi.size == 0:
        return np.zeros(color_roi.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(color_roi, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    luminance = lab[:, :, 0]

    low_saturation = cv2.inRange(saturation, 0, 115)
    bright_value = cv2.inRange(value, 135, 255)
    bright_luminance = cv2.inRange(luminance, 145, 255)
    specular = cv2.bitwise_and(low_saturation, cv2.bitwise_or(bright_value, bright_luminance))

    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    adaptive_bright = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -4,
    )

    adaptive_dark = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        4,
    )

    local_difference = cv2.absdiff(gray, cv2.GaussianBlur(gray, (11, 11), 0))
    _, contrast_mask = cv2.threshold(local_difference, 18, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(specular, adaptive_bright)
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(adaptive_dark, contrast_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _mask_features(mask: np.ndarray) -> dict[str, float | int]:
    if mask.size == 0:
        return {
            "area": 0.0,
            "component_count": 0,
            "largest_area": 0.0,
            "extent": 0.0,
            "mean_width": 0.0,
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, num_labels)
        if int(stats[label, cv2.CC_STAT_AREA]) > 0
    ]
    area = float(np.count_nonzero(mask))

    if not component_areas:
        return {
            "area": area,
            "component_count": 0,
            "largest_area": 0.0,
            "extent": 0.0,
            "mean_width": 0.0,
        }

    largest_label = 1 + int(np.argmax(component_areas))
    x = float(stats[largest_label, cv2.CC_STAT_LEFT])
    y = float(stats[largest_label, cv2.CC_STAT_TOP])
    width = float(stats[largest_label, cv2.CC_STAT_WIDTH])
    height = float(stats[largest_label, cv2.CC_STAT_HEIGHT])
    largest_area = float(stats[largest_label, cv2.CC_STAT_AREA])
    extent = largest_area / max(width * height, 1.0)

    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    mean_width = float(np.mean(distance[mask > 0]) * 2.0) if np.any(mask > 0) else 0.0

    _ = x, y
    return {
        "area": area,
        "component_count": len(component_areas),
        "largest_area": largest_area,
        "extent": float(extent),
        "mean_width": mean_width,
    }


def _local_visual_difference(reference_roi: np.ndarray, test_roi: np.ndarray) -> tuple[float, float]:
    if reference_roi.shape != test_roi.shape:
        test_roi = cv2.resize(test_roi, (reference_roi.shape[1], reference_roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(reference_roi, test_roi)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
    mean_diff = float(np.mean(gray) / 255.0)
    changed_fraction = float(np.mean(gray > 18))
    return mean_diff, changed_fraction


def _local_difference_stats(reference_roi: np.ndarray, test_roi: np.ndarray) -> dict[str, float]:
    if reference_roi.shape != test_roi.shape:
        test_roi = cv2.resize(test_roi, (reference_roi.shape[1], reference_roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(reference_roi, test_roi)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
    if gray.size == 0:
        return {
            "local_mean_difference": 0.0,
            "local_changed_fraction": 0.0,
            "local_p90_difference": 0.0,
            "local_p95_difference": 0.0,
        }
    return {
        "local_mean_difference": float(np.mean(gray) / 255.0),
        "local_changed_fraction": float(np.mean(gray > 18)),
        "local_p90_difference": float(np.percentile(gray, 90) / 255.0),
        "local_p95_difference": float(np.percentile(gray, 95) / 255.0),
    }


def _corridor_difference_stats(
    reference_roi: np.ndarray,
    test_roi: np.ndarray,
    first_bbox: tuple[int, int, int, int],
    second_bbox: tuple[int, int, int, int],
    union_bbox: tuple[int, int, int, int],
) -> dict[str, float | int]:
    if reference_roi.shape != test_roi.shape:
        test_roi = cv2.resize(test_roi, (reference_roi.shape[1], reference_roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(reference_roi, test_roi)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
    fx, fy, fw, fh = first_bbox
    sx, sy, sw, sh = second_bbox
    ux, uy, _, _ = union_bbox
    first_center = (fx - ux + fw // 2, fy - uy + fh // 2)
    second_center = (sx - ux + sw // 2, sy - uy + sh // 2)
    thickness = max(3, min(fw, fh, sw, sh) // 3)
    corridor = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.line(corridor, first_center, second_center, 255, thickness, cv2.LINE_AA)
    values = gray[corridor > 0]
    if values.size == 0:
        return {
            "corridor_area": 0,
            "corridor_mean_difference": 0.0,
            "corridor_changed_fraction": 0.0,
            "corridor_p90_difference": 0.0,
        }
    return {
        "corridor_area": int(values.size),
        "corridor_mean_difference": float(np.mean(values) / 255.0),
        "corridor_changed_fraction": float(np.mean(values > 18)),
        "corridor_p90_difference": float(np.percentile(values, 90) / 255.0),
    }


def _boxes_near_or_overlap(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    padding: int = 2,
) -> bool:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    horizontal_gap = max(bx - (ax + aw), ax - (bx + bw), 0)
    vertical_gap = max(by - (ay + ah), ay - (by + bh), 0)
    max_reasonable_gap = max(aw, ah, bw, bh) * 3 + padding
    return float(np.hypot(horizontal_gap, vertical_gap)) <= max_reasonable_gap


def _line_overlap_score(
    mask: np.ndarray,
    first_bbox: tuple[int, int, int, int],
    second_bbox: tuple[int, int, int, int],
    union_bbox: tuple[int, int, int, int],
) -> int:
    fx, fy, fw, fh = first_bbox
    sx, sy, sw, sh = second_bbox
    ux, uy, _, _ = union_bbox
    first_center = (fx - ux + fw // 2, fy - uy + fh // 2)
    second_center = (sx - ux + sw // 2, sy - uy + sh // 2)
    thickness = max(3, min(fw, fh, sw, sh) // 3)
    corridor = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.line(corridor, first_center, second_center, 255, thickness, cv2.LINE_AA)
    return int(np.count_nonzero(cv2.bitwise_and((mask > 0).astype(np.uint8), (corridor > 0).astype(np.uint8))))


def inspect_solder_joints(
    golden_image: ImageArray,
    aligned_test_image: ImageArray,
    segmentation: SegmentationResult,
    roi_map: ROIMap,
    config: PipelineConfig | None = None,
) -> list[DefectCandidate]:
    """Detect visible solder bridge, missing solder, or abnormal solder amount."""

    active_config = config or PipelineConfig()
    _ = golden_image
    _ = aligned_test_image
    golden = _ensure_color(segmentation.enhanced_golden_image)
    test = _ensure_color(segmentation.enhanced_test_image)
    defects: list[DefectCandidate] = []

    solder_mask = segmentation.solder_mask
    if solder_mask is None:
        solder_mask = np.zeros(golden.shape[:2], dtype=np.uint8)

    roi_feature_cache: dict[str, dict[str, float | int]] = {}
    roi_test_masks: dict[str, np.ndarray] = {}

    for solder_joint in roi_map.solder_joints:
        bbox = _clip_bbox(solder_joint.bbox, golden.shape)
        if bbox[2] == 0 or bbox[3] == 0:
            continue

        golden_roi = _crop(golden, bbox)
        test_roi = _crop(test, bbox)
        golden_mask = _local_solder_mask(golden_roi)
        test_mask = _local_solder_mask(test_roi)

        global_mask_roi = _crop(solder_mask, bbox)
        if global_mask_roi.shape == test_mask.shape:
            test_mask = cv2.bitwise_or(test_mask, global_mask_roi)

        golden_features = _mask_features(golden_mask)
        test_features = _mask_features(test_mask)
        local_stats = _local_difference_stats(golden_roi, test_roi)
        local_mean_diff = float(local_stats["local_mean_difference"])
        local_changed_fraction = float(local_stats["local_changed_fraction"])
        local_p90_diff = float(local_stats["local_p90_difference"])
        local_p95_diff = float(local_stats["local_p95_difference"])
        roi_feature_cache[solder_joint.id] = test_features
        roi_test_masks[solder_joint.id] = test_mask

        golden_area = float(golden_features["area"])
        test_area = float(test_features["area"])
        area_delta_ratio = (
            (test_area - golden_area) / max(golden_area, 1.0)
        )
        component_count_delta = int(test_features["component_count"]) - int(golden_features["component_count"])

        defect_type: str | None = None
        defect_score: float | None = None

        if golden_area > active_config.min_defect_area and test_area < golden_area * (1.0 - active_config.solder_area_tolerance):
            defect_type = "insufficient_solder"
            defect_score = min(1.0, abs(area_delta_ratio))
        elif test_area > golden_area * (1.0 + active_config.solder_area_tolerance) and test_area > active_config.min_defect_area:
            defect_type = "excessive_solder"
            defect_score = min(1.0, abs(area_delta_ratio))
        elif golden_area <= active_config.min_defect_area and test_area > active_config.min_defect_area * 2:
            defect_type = "unexpected_solder_blob"
            defect_score = min(1.0, test_area / max(bbox[2] * bbox[3], 1))
        elif (
            component_count_delta > 1
            and abs(area_delta_ratio) > max(0.20, active_config.solder_area_tolerance * 0.50)
            and test_area > active_config.min_defect_area
        ):
            defect_type = "abnormal_solder_shape"
            defect_score = min(1.0, component_count_delta / 5.0)

        if (
            defect_type is None
            and area_delta_ratio < -0.05
            and local_changed_fraction > 0.18
            and local_mean_diff > 0.08
        ):
            defect_type = "insufficient_solder"
            defect_score = min(1.0, local_changed_fraction + local_mean_diff)
        elif (
            defect_type is None
            and local_changed_fraction > 0.18
            and local_p90_diff > 0.10
            and local_mean_diff > 0.035
            and test_area <= golden_area * 1.20
        ):
            defect_type = "insufficient_solder"
            defect_score = min(1.0, local_changed_fraction + local_mean_diff + local_p90_diff)
        elif (
            defect_type is None
            and local_changed_fraction > 0.12
            and local_p95_diff > 0.18
            and area_delta_ratio < 0.15
            and test_area <= golden_area * 1.25
        ):
            defect_type = "insufficient_solder"
            defect_score = min(1.0, local_changed_fraction + local_p95_diff)

        if defect_type is None:
            continue

        if defect_type == "insufficient_solder" and test_area <= max(active_config.min_defect_area, golden_area * 0.12):
            defect_type = "missing_solder"

        defects.append(
            DefectCandidate(
                id=f"solder:{solder_joint.id}:{defect_type}",
                defect_type=defect_type,
                bbox=bbox,
                source="solder_inspection",
                score=defect_score,
                mask=test_mask,
                features={
                    "solder_joint_id": solder_joint.id,
                    "component_id": solder_joint.component_id,
                    "golden": golden_features,
                    "test": test_features,
                    "area_delta_ratio": float(area_delta_ratio),
                    "local_mean_difference": local_mean_diff,
                    "local_changed_fraction": local_changed_fraction,
                    "local_p90_difference": local_p90_diff,
                    "local_p95_difference": local_p95_diff,
                },
            )
        )

    bridge_id = 0
    for i, first in enumerate(roi_map.solder_joints):
        for second in roi_map.solder_joints[i + 1:]:
            if first.component_id != second.component_id:
                continue
            if not _boxes_near_or_overlap(first.bbox, second.bbox, padding=3):
                continue
            if first.id not in roi_test_masks or second.id not in roi_test_masks:
                continue

            first_bbox = _clip_bbox(first.bbox, golden.shape)
            second_bbox = _clip_bbox(second.bbox, golden.shape)
            union_x0 = min(first_bbox[0], second_bbox[0])
            union_y0 = min(first_bbox[1], second_bbox[1])
            union_x1 = max(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
            union_y1 = max(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])
            union_bbox = _clip_bbox(
                (union_x0, union_y0, union_x1 - union_x0, union_y1 - union_y0),
                golden.shape,
            )
            bridge_margin = max(
                3,
                min(first_bbox[2], first_bbox[3], second_bbox[2], second_bbox[3]) // 2,
            )
            bridge_bbox = _clip_bbox(
                (
                    union_bbox[0] - bridge_margin,
                    union_bbox[1] - bridge_margin,
                    union_bbox[2] + 2 * bridge_margin,
                    union_bbox[3] + 2 * bridge_margin,
                ),
                golden.shape,
            )

            union_test_mask = _local_solder_mask(_crop(test, union_bbox))
            union_golden_mask = _local_solder_mask(_crop(golden, union_bbox))
            union_test_features = _mask_features(union_test_mask)
            union_golden_features = _mask_features(union_golden_mask)
            union_area_delta = (
                (float(union_test_features["area"]) - float(union_golden_features["area"]))
                / max(float(union_golden_features["area"]), 1.0)
            )

            extra_mask = cv2.subtract(union_test_mask, union_golden_mask)
            corridor_extra = _line_overlap_score(
                extra_mask,
                first_bbox,
                second_bbox,
                union_bbox,
            )
            union_golden_roi = _crop(golden, union_bbox)
            union_test_roi = _crop(test, union_bbox)
            union_stats = _local_difference_stats(union_golden_roi, union_test_roi)
            corridor_stats = _corridor_difference_stats(
                union_golden_roi,
                union_test_roi,
                first_bbox,
                second_bbox,
                union_bbox,
            )
            union_mean_diff = float(union_stats["local_mean_difference"])
            union_changed_fraction = float(union_stats["local_changed_fraction"])
            corridor_changed_fraction = float(corridor_stats["corridor_changed_fraction"])
            corridor_mean_difference = float(corridor_stats["corridor_mean_difference"])
            corridor_p90_difference = float(corridor_stats["corridor_p90_difference"])
            strong_corridor_difference = (
                corridor_changed_fraction > 0.80
                and corridor_mean_difference > 0.15
                and corridor_p90_difference > 0.16
            )
            if corridor_extra < active_config.min_defect_area and not strong_corridor_difference:
                continue

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                union_test_mask,
                connectivity=8,
            )

            connected_large_region = False
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] < active_config.min_defect_area:
                    continue

                component_mask = labels == label
                fx, fy, fw, fh = first_bbox
                sx, sy, sw, sh = second_bbox
                first_local = (
                    fx - union_bbox[0],
                    fy - union_bbox[1],
                    fw,
                    fh,
                )
                second_local = (
                    sx - union_bbox[0],
                    sy - union_bbox[1],
                    sw,
                    sh,
                )
                first_overlap = np.any(_crop(component_mask, first_local))
                second_overlap = np.any(_crop(component_mask, second_local))
                if first_overlap and second_overlap:
                    connected_large_region = True
                    break

            if not connected_large_region and not strong_corridor_difference:
                continue

            golden_labels, _, _, _ = cv2.connectedComponentsWithStats(
                union_golden_mask,
                connectivity=8,
            )
            test_labels, _, _, _ = cv2.connectedComponentsWithStats(
                union_test_mask,
                connectivity=8,
            )
            if test_labels >= golden_labels and not strong_corridor_difference:
                continue

            bridge_id += 1
            defects.append(
                DefectCandidate(
                    id=f"solder:bridge:{bridge_id}",
                    defect_type="solder_bridge",
                    bbox=bridge_bbox,
                    source="solder_inspection",
                    score=0.90,
                    mask=union_test_mask,
                    features={
                        "solder_joint_ids": [first.id, second.id],
                        "component_id": first.component_id,
                        "measurement_bbox": union_bbox,
                        "union_area_delta_ratio": float(union_area_delta),
                        "corridor_extra_area": corridor_extra,
                        "union_mean_difference": union_mean_diff,
                        "union_changed_fraction": union_changed_fraction,
                        **corridor_stats,
                    },
                )
            )

    return defects
