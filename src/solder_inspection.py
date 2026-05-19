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
        #Classify the defect base on area
        if area < 80:
            defect_type = "minor defect"
        elif area < 400:
            defect_type = "moderate defect"
        elif area < 600:
            defect_type = "major defect"
        else:
            defect_type = "severe defect"

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

#The test code(for task 4): (place it under morphological.py and this function for use)

import matplotlib.pyplot as plt

def show_results(defects, debug):


    print("SOLDER DEFECT REPORT")
    print(f"Total defects detected: {len(defects)}\n")

    for i, d in enumerate(defects, 1):
        cx, cy = d["centroid"]
        print(f"Defect {i}")
        print(f"Type: {d['type']}")
        print(f"Area: {d['area']:.2f}")
        print(f"Centroid: ({cx:.2f}, {cy:.2f})")
        print("-------------------------------------")

    plt.figure(figsize=(18, 10))

    plt.subplot(231)
    plt.imshow(cv2.cvtColor(debug["ref"], cv2.COLOR_BGR2RGB))
    plt.title("Reference Image")
    plt.axis("off")

    plt.subplot(232)
    plt.imshow(cv2.cvtColor(debug["test"], cv2.COLOR_BGR2RGB))
    plt.title("Test Image")
    plt.axis("off")

    plt.subplot(233)
    plt.imshow(debug["diff"], cmap="gray")
    plt.title("Difference")
    plt.axis("off")

    plt.subplot(234)
    plt.imshow(debug["binary"], cmap="gray")
    plt.title("Binary")
    plt.axis("off")

    plt.subplot(235)
    plt.imshow(debug["morph"], cmap="gray")
    plt.title("Morphology Result")
    plt.axis("off")

    plt.subplot(236)
    plt.imshow(cv2.cvtColor(debug["result"], cv2.COLOR_BGR2RGB))
    plt.title("Defect Result (Red Dots)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    #change the input jpg file here
    ref = cv2.imread("01.JPG")
    test = cv2.imread("01_open_circuit_01.jpg")

    defects, debug = inspect_solder_joints(ref, test)
    show_results(defects, debug)
