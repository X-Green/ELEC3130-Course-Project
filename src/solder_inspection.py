"""Task 4: solder joint inspection."""

from __future__ import annotations

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

    _ = golden_image
    _ = aligned_test_image
    _ = segmentation
    _ = roi_map
    _ = config
    return []

