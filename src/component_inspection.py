"""Task 3: component-level inspection."""

from __future__ import annotations

try:
    from .models import DefectCandidate, ImageArray, PipelineConfig, ROIMap, SegmentationResult
except ImportError:  # pragma: no cover - supports direct script execution
    from models import DefectCandidate, ImageArray, PipelineConfig, ROIMap, SegmentationResult


def inspect_components(
    golden_image: ImageArray,
    aligned_test_image: ImageArray,
    segmentation: SegmentationResult,
    roi_map: ROIMap,
    config: PipelineConfig | None = None,
) -> list[DefectCandidate]:
    """Detect missing, shifted, rotated, or polarity-related component defects.

    TODO:
    - Compare each component ROI against the golden reference.
    - Use template matching, edge difference, contour features, or SSIM.
    - Estimate displacement and rotation angle.
    - Return component-level defect candidates.

    Placeholder behavior: return no component defects.
    """

    _ = golden_image
    _ = aligned_test_image
    _ = segmentation
    _ = roi_map
    _ = config
    return []

