"""Task 4: component-level inspection."""

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


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    margin_ratio: float,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))
    return _clip_bbox(
        (x - margin_x, y - margin_y, width + 2 * margin_x, height + 2 * margin_y),
        image_shape,
    )


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    return image[y:y + height, x:x + width]


def _to_gray(image: ImageArray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.dtype != np.uint8:
        gray = gray.astype(np.float32)
        if gray.max() <= 1.0:
            gray *= 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _edge_density(gray_roi: np.ndarray) -> float:
    if gray_roi.size == 0:
        return 0.0
    edges = cv2.Canny(gray_roi, 50, 150)
    return float(np.mean(edges > 0))


def _largest_contour_angle(gray_roi: np.ndarray) -> tuple[float | None, float]:
    if gray_roi.size == 0:
        return None, 0.0

    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if len(contour) < 5 or area <= 0:
        return None, area

    rect = cv2.minAreaRect(contour)
    angle = float(rect[-1])
    if angle < -45:
        angle += 90
    return angle, area


def _angle_delta(angle_a: float | None, angle_b: float | None) -> float | None:
    if angle_a is None or angle_b is None:
        return None

    delta = abs(angle_a - angle_b) % 180.0
    if delta > 90.0:
        delta = 180.0 - delta
    return float(delta)


def _mean_absdiff_score(reference: np.ndarray, test: np.ndarray) -> float:
    if reference.shape != test.shape:
        test = cv2.resize(test, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    difference = cv2.absdiff(reference, test)
    return float(np.mean(difference) / 255.0)


def _difference_features(reference: np.ndarray, test: np.ndarray) -> dict[str, float]:
    if reference.shape != test.shape:
        test = cv2.resize(test, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)

    difference = cv2.absdiff(reference, test)
    if difference.ndim == 3:
        difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area = float(np.count_nonzero(mask))
    fraction = area / max(mask.shape[0] * mask.shape[1], 1)
    if area == 0:
        return {
            "difference_area": 0.0,
            "difference_fraction": 0.0,
            "difference_centroid_dx": 0.0,
            "difference_centroid_dy": 0.0,
            "difference_spread_ratio": 0.0,
        }

    moments = cv2.moments(mask)
    center_x = mask.shape[1] / 2.0
    center_y = mask.shape[0] / 2.0
    if moments["m00"] > 0:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
    else:
        ys, xs = np.where(mask > 0)
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))

    ys, xs = np.where(mask > 0)
    spread_x = float(np.std(xs)) / max(mask.shape[1], 1)
    spread_y = float(np.std(ys)) / max(mask.shape[0], 1)

    return {
        "difference_area": area,
        "difference_fraction": float(fraction),
        "difference_centroid_dx": float((centroid_x - center_x) / max(mask.shape[1], 1)),
        "difference_centroid_dy": float((centroid_y - center_y) / max(mask.shape[0], 1)),
        "difference_spread_ratio": float(max(spread_x, spread_y)),
    }


def _template_match(
    reference_roi: np.ndarray,
    search_roi: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    if reference_roi.size == 0 or search_roi.size == 0:
        return 0.0, (0, 0)

    if search_roi.shape[0] < reference_roi.shape[0] or search_roi.shape[1] < reference_roi.shape[1]:
        return 0.0, (0, 0)

    if float(np.std(reference_roi)) < 1e-6 or float(np.std(search_roi)) < 1e-6:
        return 0.0, (0, 0)

    reference = cv2.equalizeHist(reference_roi)
    search = cv2.equalizeHist(search_roi)

    result = cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_location = cv2.minMaxLoc(result)
    return float(max_score), (int(max_location[0]), int(max_location[1]))


def _rotate_roi_keep_size(roi: np.ndarray, angle_degrees: float) -> np.ndarray:
    if roi.size == 0:
        return roi
    height, width = roi.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    border = float(np.median(roi)) if roi.ndim == 2 else tuple(float(v) for v in np.median(roi, axis=(0, 1)))
    return cv2.warpAffine(
        roi,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _same_size_template_score(reference_roi: np.ndarray, test_roi: np.ndarray) -> float:
    if reference_roi.size == 0 or test_roi.size == 0:
        return 0.0
    if reference_roi.shape != test_roi.shape:
        test_roi = cv2.resize(
            test_roi,
            (reference_roi.shape[1], reference_roi.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if float(np.std(reference_roi)) < 1e-6 or float(np.std(test_roi)) < 1e-6:
        return 0.0
    reference = cv2.equalizeHist(reference_roi)
    test = cv2.equalizeHist(test_roi)
    result = cv2.matchTemplate(test, reference, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, _ = cv2.minMaxLoc(result)
    return float(max_score)


def _rotation_template_features(
    reference_roi: np.ndarray,
    test_roi: np.ndarray,
) -> dict[str, float]:
    if reference_roi.size == 0 or test_roi.size == 0:
        return {
            "best_rotated_template_score": 0.0,
            "best_rotated_template_angle": 0.0,
            "rotation_score_gain": 0.0,
        }

    if reference_roi.shape != test_roi.shape:
        test_roi = cv2.resize(
            test_roi,
            (reference_roi.shape[1], reference_roi.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    base_score = _same_size_template_score(reference_roi, test_roi)
    best_score = base_score
    best_angle = 0.0
    for angle in (45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
        rotated_reference = _rotate_roi_keep_size(reference_roi, angle)
        rotated_score = _same_size_template_score(rotated_reference, test_roi)
        if rotated_score > best_score:
            best_score = rotated_score
            best_angle = angle

    return {
        "best_rotated_template_score": float(best_score),
        "best_rotated_template_angle": float(best_angle),
        "rotation_score_gain": float(best_score - base_score),
    }


def inspect_components(
    golden_image: ImageArray,
    aligned_test_image: ImageArray,
    segmentation: SegmentationResult,
    roi_map: ROIMap,
    config: PipelineConfig | None = None,
) -> list[DefectCandidate]:
    """Detect missing, shifted, rotated, or visually inconsistent components."""

    active_config = config or PipelineConfig()
    golden_gray = _to_gray(golden_image)
    test_gray = _to_gray(aligned_test_image)
    _ = segmentation

    defects: list[DefectCandidate] = []

    for component in roi_map.components:
        bbox = _clip_bbox(component.bbox, golden_gray.shape)
        if bbox[2] == 0 or bbox[3] == 0:
            continue

        search_bbox = _expand_bbox(
            bbox,
            active_config.component_search_margin_ratio,
            test_gray.shape,
        )

        golden_roi = _crop(golden_gray, bbox)
        search_roi = _crop(test_gray, search_bbox)
        score, location = _template_match(golden_roi, search_roi)

        expected_x, expected_y, width, height = bbox
        found_x = search_bbox[0] + location[0]
        found_y = search_bbox[1] + location[1]
        dx = float(found_x - expected_x)
        dy = float(found_y - expected_y)
        displacement = float(np.hypot(dx, dy))

        matched_bbox = _clip_bbox((found_x, found_y, width, height), test_gray.shape)
        matched_test_roi = _crop(test_gray, matched_bbox)
        difference_score = _mean_absdiff_score(golden_roi, matched_test_roi)
        aligned_test_roi = _crop(test_gray, bbox)
        direct_difference_score = _mean_absdiff_score(golden_roi, aligned_test_roi)
        difference_features = _difference_features(golden_roi, aligned_test_roi)
        rotation_template = _rotation_template_features(golden_roi, aligned_test_roi)

        golden_edge_density = _edge_density(golden_roi)
        test_edge_density = _edge_density(matched_test_roi)
        edge_ratio = (
            test_edge_density / golden_edge_density
            if golden_edge_density > 1e-6
            else 1.0
        )

        golden_angle, golden_contour_area = _largest_contour_angle(golden_roi)
        test_angle, test_contour_area = _largest_contour_angle(matched_test_roi)
        rotation_delta = _angle_delta(golden_angle, test_angle)

        features = {
            "component_id": component.id,
            "template_score": score,
            "difference_score": difference_score,
            "direct_difference_score": direct_difference_score,
            "displacement_px": displacement,
            "dx": dx,
            "dy": dy,
            "golden_edge_density": golden_edge_density,
            "test_edge_density": test_edge_density,
            "edge_density_ratio": edge_ratio,
            "rotation_delta_deg": rotation_delta,
            "golden_contour_area": golden_contour_area,
            "test_contour_area": test_contour_area,
            **difference_features,
            **rotation_template,
        }

        defect_type: str | None = None
        defect_score: float | None = None

        direct_diff_fraction = float(difference_features["difference_fraction"])
        direct_diff_spread = float(difference_features["difference_spread_ratio"])
        best_rotated_template_score = float(rotation_template["best_rotated_template_score"])
        best_rotated_template_angle = float(rotation_template["best_rotated_template_angle"])
        rotation_score_gain = float(rotation_template["rotation_score_gain"])
        direct_diff_centroid_offset = float(
            np.hypot(
                difference_features["difference_centroid_dx"],
                difference_features["difference_centroid_dy"],
            )
        )
        global_contrast_like_change = (
            displacement <= 1.0
            and score > 0.85
            and edge_ratio > 0.80
            and direct_diff_fraction > 0.50
            and direct_diff_centroid_offset < 0.08
            and direct_difference_score < 0.16
        )

        if global_contrast_like_change:
            defect_type = None
        elif (
            displacement <= active_config.component_shift_tolerance_px
            and best_rotated_template_angle > 0.0
            and rotation_score_gain > 0.08
            and best_rotated_template_score > max(score + 0.05, 0.62)
            and direct_diff_fraction > 0.04
            and direct_difference_score > 0.025
            and edge_ratio > 0.55
        ):
            defect_type = "rotated_component"
            defect_score = min(
                1.0,
                max(rotation_score_gain * 2.5, direct_diff_fraction * 2.0 + direct_difference_score),
            )
        elif (
            displacement <= active_config.component_shift_tolerance_px
            and rotation_delta is not None
            and rotation_delta > active_config.component_rotation_tolerance_deg
            and direct_diff_fraction > 0.04
            and direct_difference_score > 0.025
            and edge_ratio > 0.55
        ):
            defect_type = "rotated_component"
            defect_score = min(1.0, max(rotation_delta / 90.0, direct_diff_fraction * 2.0))
        elif (
            displacement <= max(active_config.component_shift_tolerance_px * 1.50, 12.0)
            and 0.30 <= score <= 0.82
            and direct_diff_fraction > 0.08
            and direct_difference_score > 0.045
            and direct_diff_spread > 0.24
            and edge_ratio > 0.65
        ):
            defect_type = "rotated_component"
            defect_score = min(1.0, direct_diff_fraction * 2.2 + direct_difference_score + max(0.0, 0.82 - score))
        elif displacement > active_config.component_shift_tolerance_px and score > active_config.component_missing_threshold:
            defect_type = "shifted_component"
            defect_score = min(1.0, displacement / max(width, height, 1))
        elif direct_diff_fraction > 0.10 and direct_difference_score > 0.07 and score > 0.55:
            defect_type = "shifted_component"
            defect_score = min(1.0, direct_diff_fraction * 2.0 + direct_difference_score)
        elif (
            direct_diff_fraction > 0.22
            and direct_difference_score > 0.10
            and edge_ratio > 0.70
            and direct_diff_spread > 0.18
        ):
            defect_type = "rotated_component"
            defect_score = min(1.0, direct_diff_fraction * 2.0 + direct_difference_score)
        elif score < active_config.component_missing_threshold and direct_difference_score > 0.08:
            defect_type = "missing_component"
            defect_score = max(1.0 - score, 1.0 - edge_ratio)
        elif (
            score < active_config.component_match_threshold
            and rotation_delta is not None
            and rotation_delta > active_config.component_rotation_tolerance_deg
            and test_contour_area > active_config.min_defect_area
            and direct_diff_spread > 0.18
        ):
            defect_type = "rotated_component"
            defect_score = min(1.0, rotation_delta / 90.0)
        elif edge_ratio < 0.35 and direct_difference_score > 0.08:
            defect_type = "missing_component"
            defect_score = max(1.0 - score, 1.0 - edge_ratio)
        elif direct_diff_fraction > 0.24 and direct_difference_score > 0.11:
            defect_type = "missing_component"
            defect_score = min(1.0, direct_diff_fraction * 2.0 + direct_difference_score)
        elif score < active_config.component_match_threshold and direct_difference_score > 0.18:
            defect_type = "visual_component_mismatch"
            defect_score = max(1.0 - score, direct_difference_score)

        if defect_type is None:
            continue

        defects.append(
            DefectCandidate(
                id=f"component:{component.id}:{defect_type}",
                defect_type=defect_type,
                bbox=bbox,
                source="component_inspection",
                score=defect_score,
                features=features,
                metadata={
                    "component_type": component.type,
                    "polarity": component.polarity,
                    "matched_bbox": matched_bbox,
                },
            )
        )

    return defects
