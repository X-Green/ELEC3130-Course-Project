"""Task 2: image enhancement, color normalization, filtering, and segmentation."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .models import ImageArray, PipelineConfig, ROIMap, SegmentationResult
except ImportError:  # pragma: no cover - supports direct script execution
    from models import ImageArray, PipelineConfig, ROIMap, SegmentationResult


def _to_uint8(img: ImageArray) -> np.ndarray:
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    if img.max() <= 1.0:
        img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def _ensure_color(img: ImageArray) -> np.ndarray:
    img = _to_uint8(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def _apply_clahe_color(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    l_equalized = clahe.apply(l_channel)

    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    enhanced = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    return enhanced


def _match_histogram_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_shape = source.shape

    source = source.ravel()
    reference = reference.ravel()

    source_values, source_indices, source_counts = np.unique(
        source,
        return_inverse=True,
        return_counts=True,
    )

    reference_values, reference_counts = np.unique(
        reference,
        return_counts=True,
    )

    source_quantiles = np.cumsum(source_counts).astype(np.float64)
    source_quantiles /= source_quantiles[-1]

    reference_quantiles = np.cumsum(reference_counts).astype(np.float64)
    reference_quantiles /= reference_quantiles[-1]

    interp_values = np.interp(
        source_quantiles,
        reference_quantiles,
        reference_values,
    )

    matched = interp_values[source_indices].reshape(source_shape)

    return np.clip(matched, 0, 255).astype(np.uint8)


def _match_histogram_color(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    matched = np.zeros_like(source)

    for channel in range(3):
        matched[:, :, channel] = _match_histogram_channel(
            source[:, :, channel],
            reference[:, :, channel],
        )

    return matched


def _denoise_preserve_edges(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(
        img,
        d=5,
        sigmaColor=50,
        sigmaSpace=50,
    )


def _create_board_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    white_ratio = np.mean(mask == 255)

    if white_ratio > 0.70:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def _create_component_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        blurred,
        threshold1=50,
        threshold2=150,
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def _create_solder_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    bright_mask = cv2.inRange(value, 180, 255)
    low_saturation_mask = cv2.inRange(saturation, 0, 90)

    solder_mask = cv2.bitwise_and(bright_mask, low_saturation_mask)

    kernel = np.ones((3, 3), np.uint8)

    solder_mask = cv2.morphologyEx(solder_mask, cv2.MORPH_OPEN, kernel)
    solder_mask = cv2.morphologyEx(solder_mask, cv2.MORPH_CLOSE, kernel)

    return solder_mask


def _apply_roi_constraints(
    component_mask: np.ndarray,
    solder_mask: np.ndarray,
    roi_map: ROIMap | None,
) -> tuple[np.ndarray, np.ndarray]:
    if roi_map is None:
        return component_mask, solder_mask

    constrained_component_mask = np.zeros_like(component_mask)
    constrained_solder_mask = np.zeros_like(solder_mask)

    for component in roi_map.components:
        x, y, w, h = component.bbox
        constrained_component_mask[y:y + h, x:x + w] = component_mask[y:y + h, x:x + w]

    for solder_joint in roi_map.solder_joints:
        x, y, w, h = solder_joint.bbox
        constrained_solder_mask[y:y + h, x:x + w] = solder_mask[y:y + h, x:x + w]

    return constrained_component_mask, constrained_solder_mask


def enhance_and_segment(
    golden_image: ImageArray,
    aligned_test_image: ImageArray,
    roi_map: ROIMap | None = None,
    config: PipelineConfig | None = None,
) -> SegmentationResult:
    """Enhance images and generate masks for later inspection."""

    golden = _ensure_color(golden_image)
    test = _ensure_color(aligned_test_image)

    if golden.shape[:2] != test.shape[:2]:
        test = cv2.resize(
            test,
            (golden.shape[1], golden.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    enhanced_golden = _apply_clahe_color(golden)
    enhanced_test = _apply_clahe_color(test)

    enhanced_golden = _denoise_preserve_edges(enhanced_golden)
    enhanced_test = _denoise_preserve_edges(enhanced_test)

    normalized_test = _match_histogram_color(
        enhanced_test,
        enhanced_golden,
    )

    board_mask = _create_board_mask(normalized_test)
    component_mask = _create_component_mask(normalized_test)
    solder_mask = _create_solder_mask(normalized_test)

    component_mask, solder_mask = _apply_roi_constraints(
        component_mask,
        solder_mask,
        roi_map,
    )

    metadata = {
        "status": "completed",
        "methods": [
            "CLAHE on LAB luminance channel",
            "bilateral filtering",
            "histogram matching",
            "Otsu thresholding",
            "Canny edge detection",
            "HSV solder-color segmentation",
            "morphological opening and closing",
        ],
        "has_roi_map": roi_map is not None,
        "debug": False if config is None else config.debug,
    }

    return SegmentationResult(
        enhanced_golden_image=enhanced_golden,
        enhanced_test_image=normalized_test,
        board_mask=board_mask,
        component_mask=component_mask,
        solder_mask=solder_mask,
        metadata=metadata,
    )
