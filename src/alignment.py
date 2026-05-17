"""Task 1: board localization and image registration."""

from __future__ import annotations

try:
    from .models import AlignmentResult, ImageArray, PipelineConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from models import AlignmentResult, ImageArray, PipelineConfig


def register_board(
    golden_image: ImageArray,
    test_image: ImageArray,
    config: PipelineConfig | None = None,
) -> AlignmentResult:
    """Align the test PCBA image to the golden reference image.

    TODO:
    - Detect board edges, corners, or fiducial markers.
    - Estimate translation, rotation, scale, or perspective transform.
    - Warp the test image into golden-image coordinates.

    Placeholder behavior: return the test image unchanged.
    """

    _ = golden_image
    _ = config
    return AlignmentResult(
        aligned_test_image=test_image,
        transform_matrix=None,
        board_mask=None,
        metadata={"status": "placeholder", "method": "identity"},
    )
