"""Morphology and subtraction helper signatures for defect extraction."""

from __future__ import annotations

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

    _ = reference_mask
    _ = test_mask
    raise NotImplementedError("Morphological subtraction is not implemented yet.")


def clean_defect_mask(
    defect_mask: ImageArray,
    min_area: int = 20,
) -> ImageArray:
    """Clean a raw defect mask using morphology and area filtering.

    TODO: Apply opening, closing, and connected-component area filtering.
    """

    _ = defect_mask
    _ = min_area
    raise NotImplementedError("Defect mask cleanup is not implemented yet.")
