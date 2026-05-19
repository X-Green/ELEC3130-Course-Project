"""Task 4: solder joint inspection."""

from __future__ import annotations

import cv2
import numpy as np

ImageArray = np.ndarray

try:
    from .models import DefectCandidate, ImageArray, PipelineConfig, ROIMap, SegmentationResult
except ImportError:  # pragma: no cover - supports direct script execution
    from models import DefectCandidate, ImageArray, PipelineConfig, ROIMap, SegmentationResult


def inspect_solder_joints(
    golden_image: ImageArray,
    aligned_test_image: ImageArray,
    segmentation: SegmentationResult,
    roi_map: ROIMap,
    config: PipelineConfig | None = None,
) -> list[DefectCandidate]:
    """Detect visible solder bridge, missing solder, or abnormal solder amount.

    TODO:
    - Locate solder joint ROIs around expected pads.
    - Segment solder using color, brightness, and local thresholding.
    - Measure area, shape, and connectivity.
    - Return solder-level defect candidates.

    Placeholder behavior: return no solder defects.
    """

     _ = segmentation
    _ = roi_map
    _ = config

    defects = []

 
    # 1. color images
    ref_color = golden_image.copy()
    test_color = aligned_test_image.copy()


    # 2. change to grayscale
    ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    test = cv2.cvtColor(test_color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


    # 3. normalization
    if np.mean(test) > 0:
        test *= (np.mean(ref) / np.mean(test))
    test = np.clip(test, 0, 1)

 
    # 4. difference
    diff = np.abs(ref - test)
    diff_img = (diff * 255).astype(np.uint8)


    # 5. binary image
    _, binary = cv2.threshold(diff_img, 10, 255, cv2.THRESH_BINARY)


    # 6. morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)


    # 7. connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        morph, connectivity=8
    )
    result_overlay = test_color.copy()

    # 8. classification
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        if area < 20:
            continue

        if area < 80:
            defect_type = "Insufficient Solder"
        elif area < 250:
            defect_type = "Possible Solder Defect"
        elif area < 600:
            defect_type = "Excess Solder"
        else:
            defect_type = "Solder Bridge"

        defects.append({
            "type": defect_type,
            "area": float(area),
            "bbox": (x, y, w, h),
            "centroid": (float(cx), float(cy)),
        })


        if not np.isnan(cx) and not np.isnan(cy):
            cx_i = int(round(cx))
            cy_i = int(round(cy))

            h_img, w_img = result_overlay.shape[:2]
            cx_i = np.clip(cx_i, 0, w_img - 1)
            cy_i = np.clip(cy_i, 0, h_img - 1)

            cv2.circle(result_overlay, (cx_i, cy_i), 10, (0, 0, 255), -1)

        # 9. debug output
        debug = {
            "ref": ref_color,
            "test": test_color,
            "diff": diff_img,
            "binary": binary,
            "morph": morph,
            "result": result_overlay,
        }
    
        return defects, debug
