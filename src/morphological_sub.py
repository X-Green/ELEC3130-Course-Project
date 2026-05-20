"""Morphology and subtraction helper signatures for defect extraction."""

from __future__ import annotations

import cv2
import numpy as np

ImageArray = np.ndarray

try:
    from .models import ImageArray
except ImportError:  # pragma: no cover - supports direct script execution
    from models import ImageArray


def subtract_binary_masks(
    reference_mask: ImageArray,
    test_mask: ImageArray,
) -> ImageArray:
    """Compute difference between reference and test binary masks.

    TODO: Implement XOR, missing-material, and extra-material masks.
    """
    reference_mask = reference_mask.astype(np.uint8)
    test_mask = test_mask.astype(np.uint8)

    xor_mask = cv2.bitwise_xor(reference_mask, test_mask)

    missing = cv2.bitwise_and(reference_mask, cv2.bitwise_not(test_mask))
    extra = cv2.bitwise_and(test_mask, cv2.bitwise_not(reference_mask))

    defect_mask = cv2.bitwise_or(xor_mask, missing)
    defect_mask = cv2.bitwise_or(defect_mask, extra)

    return defect_mask


def clean_defect_mask(
    defect_mask: ImageArray,
    min_area: int = 20,
) -> ImageArray:
    """Clean a raw defect mask using morphology and area filtering.

    TODO: Apply opening, closing, and connected-component area filtering.
    """

    defect_mask = defect_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opened = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    cleaned = np.zeros_like(closed)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned
