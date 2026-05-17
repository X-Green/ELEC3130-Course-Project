"""Frequency-domain filtering helper signatures for preprocessing."""

from __future__ import annotations

try:
    from .models import ImageArray
except ImportError:  # pragma: no cover - supports direct script execution
    from models import ImageArray


def apply_frequency_filter(
    image: ImageArray,
    filter_name: str = "lowpass",
    cutoff: float | None = None,
) -> ImageArray:
    """Apply a frequency-domain filter to an image.

    TODO:
    - Implement FFT conversion.
    - Add low-pass, high-pass, and band-pass masks.
    - Transform the filtered result back to the spatial domain.
    """

    _ = filter_name
    _ = cutoff
    raise NotImplementedError("Frequency-domain filtering is not implemented yet.")
