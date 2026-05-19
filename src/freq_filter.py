"""Frequency-domain filtering helpers for preprocessing and evaluation."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .models import ImageArray
except ImportError:  # pragma: no cover - supports direct script execution
    from models import ImageArray


def _radial_mask(
    shape: tuple[int, int],
    filter_name: str,
    cutoff: float,
) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_radius = np.sqrt(ccol ** 2 + crow ** 2)
    radius = max(1.0, cutoff * max_radius if cutoff <= 1.0 else cutoff)

    if filter_name == "lowpass":
        return (distance <= radius).astype(np.float32)

    if filter_name == "highpass":
        return (distance >= radius).astype(np.float32)

    if filter_name == "bandpass":
        inner = radius * 0.50
        outer = radius
        return ((distance >= inner) & (distance <= outer)).astype(np.float32)

    raise ValueError(f"Unsupported filter_name: {filter_name}")


def _as_float_gray(image: ImageArray) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim == 3:
        if image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2GRAY)
        else:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    gray = image_array.astype(np.float32)
    if gray.max() <= 1.0:
        gray *= 255.0
    return gray


def _filtered_float_single_channel(channel: np.ndarray, filter_name: str, cutoff: float) -> np.ndarray:
    channel_float = channel.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(channel_float))
    mask = _radial_mask(channel.shape, filter_name, cutoff)
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    return np.real(filtered).astype(np.float32)


def _normalize_debug_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = image - float(np.min(image))
    high = float(np.percentile(image, 99.5))
    if high <= 1e-6:
        high = float(np.max(image))
    if high <= 1e-6:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip(image / high, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def _to_uint8_image(image: ImageArray) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.dtype == np.uint8:
        return image_array.copy()

    image_float = image_array.astype(np.float32)
    if image_float.max() <= 1.0:
        image_float *= 255.0
    return np.clip(image_float, 0, 255).astype(np.uint8)


def _periodic_peak_mask(
    magnitude: np.ndarray,
    min_radius_ratio: float,
    max_radius_ratio: float,
    peak_sigma: float,
    max_peaks: int,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_radius = np.sqrt(ccol ** 2 + crow ** 2)

    search_mask = (
        (distance >= max(1.0, min_radius_ratio * max_radius))
        & (distance <= max(1.0, max_radius_ratio * max_radius))
    )
    if not np.any(search_mask):
        return [], np.full(magnitude.shape, 255, dtype=np.uint8)

    smooth = cv2.GaussianBlur(magnitude.astype(np.float32), (0, 0), sigmaX=8.0, sigmaY=8.0)
    residual = magnitude.astype(np.float32) - smooth
    search_values = residual[search_mask]
    median = float(np.median(search_values))
    mad = float(np.median(np.abs(search_values - median)))
    robust_std = max(1e-6, 1.4826 * mad)
    sigma_threshold = median + peak_sigma * robust_std
    percentile_threshold = float(np.percentile(search_values, 99.85))
    threshold = max(sigma_threshold, percentile_threshold)

    local_max = residual == cv2.dilate(residual, np.ones((7, 7), np.float32))
    candidate_mask = search_mask & local_max & (residual > threshold)
    candidate_points = np.argwhere(candidate_mask)
    if candidate_points.size == 0:
        return [], np.full(magnitude.shape, 255, dtype=np.uint8)

    candidates: list[tuple[int, int, float]] = [
        (int(row), int(col), float(residual[row, col]))
        for row, col in candidate_points
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)

    selected: list[tuple[int, int, float]] = []
    min_separation = 8.0
    for row, col, score in candidates:
        mirror_row = 2 * crow - row
        mirror_col = 2 * ccol - col
        if not (0 <= mirror_row < rows and 0 <= mirror_col < cols):
            continue
        if any(np.hypot(row - kept_row, col - kept_col) < min_separation for kept_row, kept_col, _ in selected):
            continue
        selected.append((row, col, score))
        if len(selected) >= max_peaks:
            break

    peak_debug = np.zeros(magnitude.shape, dtype=np.uint8)
    for row, col, _ in selected:
        mirror_row = 2 * crow - row
        mirror_col = 2 * ccol - col
        cv2.circle(peak_debug, (col, row), 4, 255, thickness=-1)
        cv2.circle(peak_debug, (mirror_col, mirror_row), 4, 255, thickness=-1)

    return selected, peak_debug


def _suppress_periodic_noise_gray(
    gray: np.ndarray,
    min_radius_ratio: float,
    max_radius_ratio: float,
    peak_sigma: float,
    notch_radius: int,
    max_peaks: int,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    gray_float = gray.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray_float))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    peaks, peak_debug = _periodic_peak_mask(
        magnitude,
        min_radius_ratio=min_radius_ratio,
        max_radius_ratio=max_radius_ratio,
        peak_sigma=peak_sigma,
        max_peaks=max_peaks,
    )

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    notch_mask = np.ones(gray.shape, dtype=np.float32)
    for row, col, _ in peaks:
        mirror_row = 2 * crow - row
        mirror_col = 2 * ccol - col
        cv2.circle(notch_mask, (col, row), notch_radius, 0.0, thickness=-1)
        cv2.circle(notch_mask, (mirror_col, mirror_row), notch_radius, 0.0, thickness=-1)

    if peaks:
        filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * notch_mask))).astype(np.float32)
        filtered += float(np.mean(gray_float) - np.mean(filtered))
        filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
    else:
        filtered_uint8 = gray.copy()

    pass_mask_debug = (notch_mask * 255.0).astype(np.uint8)
    notch_debug = cv2.cvtColor(pass_mask_debug, cv2.COLOR_GRAY2BGR)
    notch_debug[peak_debug > 0] = (0, 0, 255)

    metadata = {
        "status": "completed",
        "method": "fft_periodic_noise_notch",
        "active": bool(peaks),
        "selected_peak_count": len(peaks),
        "notched_frequency_count": len(peaks) * 2,
        "notch_radius": int(notch_radius),
        "min_radius_ratio": float(min_radius_ratio),
        "max_radius_ratio": float(max_radius_ratio),
        "peak_sigma": float(peak_sigma),
        "peaks_yx": [(int(row), int(col)) for row, col, _ in peaks],
    }
    return filtered_uint8, metadata, notch_debug


def _notch_filter_gray_with_peaks(
    gray: np.ndarray,
    peaks: list[tuple[int, int, float]],
    notch_radius: int,
    method: str,
    peak_source: str,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    gray_float = gray.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray_float))
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    notch_mask = np.ones(gray.shape, dtype=np.float32)
    peak_debug = np.zeros(gray.shape, dtype=np.uint8)

    for row, col, _ in peaks:
        mirror_row = 2 * crow - row
        mirror_col = 2 * ccol - col
        if not (0 <= mirror_row < rows and 0 <= mirror_col < cols):
            continue
        cv2.circle(notch_mask, (col, row), notch_radius, 0.0, thickness=-1)
        cv2.circle(notch_mask, (mirror_col, mirror_row), notch_radius, 0.0, thickness=-1)
        cv2.circle(peak_debug, (col, row), 4, 255, thickness=-1)
        cv2.circle(peak_debug, (mirror_col, mirror_row), 4, 255, thickness=-1)

    if peaks:
        filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * notch_mask))).astype(np.float32)
        filtered += float(np.mean(gray_float) - np.mean(filtered))
        filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
    else:
        filtered_uint8 = gray.copy()

    pass_mask_debug = (notch_mask * 255.0).astype(np.uint8)
    notch_debug = cv2.cvtColor(pass_mask_debug, cv2.COLOR_GRAY2BGR)
    notch_debug[peak_debug > 0] = (0, 0, 255)

    metadata = {
        "status": "completed",
        "method": method,
        "peak_source": peak_source,
        "active": bool(peaks),
        "selected_peak_count": len(peaks),
        "notched_frequency_count": len(peaks) * 2,
        "notch_radius": int(notch_radius),
        "peaks_yx": [(int(row), int(col)) for row, col, _ in peaks],
    }
    return filtered_uint8, metadata, notch_debug


def suppress_periodic_noise(
    image: ImageArray,
    min_radius_ratio: float = 0.08,
    max_radius_ratio: float = 0.48,
    peak_sigma: float = 6.0,
    notch_radius: int = 5,
    max_peaks: int = 24,
) -> tuple[ImageArray, dict[str, object], ImageArray]:
    """Suppress strong repetitive noise with an FFT notch filter.

    The detector searches for sharp non-DC peaks in the log-magnitude spectrum.
    If no strong periodic peaks are found, the input image is returned unchanged.
    For color images, only luminance is filtered so component and solder color
    cues remain available for downstream inspection.
    """

    image_array = _to_uint8_image(image)
    safe_notch_radius = max(1, int(notch_radius))
    safe_max_peaks = max(1, int(max_peaks))

    if image_array.ndim == 2:
        return _suppress_periodic_noise_gray(
            image_array,
            min_radius_ratio=min_radius_ratio,
            max_radius_ratio=max_radius_ratio,
            peak_sigma=peak_sigma,
            notch_radius=safe_notch_radius,
            max_peaks=safe_max_peaks,
        )

    if image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)

    ycrcb = cv2.cvtColor(image_array, cv2.COLOR_BGR2YCrCb)
    filtered_y, metadata, notch_debug = _suppress_periodic_noise_gray(
        ycrcb[:, :, 0],
        min_radius_ratio=min_radius_ratio,
        max_radius_ratio=max_radius_ratio,
        peak_sigma=peak_sigma,
        notch_radius=safe_notch_radius,
        max_peaks=safe_max_peaks,
    )
    ycrcb[:, :, 0] = filtered_y
    filtered_color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return filtered_color, metadata, notch_debug


def suppress_reference_periodic_noise(
    reference_image: ImageArray,
    test_image: ImageArray,
    min_radius_ratio: float = 0.015,
    max_radius_ratio: float = 0.48,
    peak_sigma: float = 5.0,
    notch_radius: int = 5,
    max_peaks: int = 24,
) -> tuple[ImageArray, dict[str, object], ImageArray]:
    """Suppress periodic noise found in the golden/test difference spectrum.

    This is the version used by the inspection pipeline. It detects narrow
    repeated-noise peaks from ``test - reference`` in the frequency domain,
    then applies those notch locations to the test image luminance. Local
    defects are spatially limited and usually do not create strong symmetric
    spectral peaks, while sinusoidal/stripe interference does.
    """

    reference = _to_uint8_image(reference_image)
    test = _to_uint8_image(test_image)
    safe_notch_radius = max(1, int(notch_radius))
    safe_max_peaks = max(1, int(max_peaks))

    if reference.shape[:2] != test.shape[:2]:
        test = cv2.resize(
            test,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    if reference.ndim == 3 and reference.shape[2] == 4:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGRA2BGR)
    if test.ndim == 3 and test.shape[2] == 4:
        test = cv2.cvtColor(test, cv2.COLOR_BGRA2BGR)

    reference_gray = _as_float_gray(reference)
    test_gray = _as_float_gray(test)
    difference = test_gray - reference_gray
    spectrum = np.fft.fftshift(np.fft.fft2(difference))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    peaks, _ = _periodic_peak_mask(
        magnitude,
        min_radius_ratio=min_radius_ratio,
        max_radius_ratio=max_radius_ratio,
        peak_sigma=peak_sigma,
        max_peaks=safe_max_peaks,
    )

    if test.ndim == 2:
        filtered_gray, metadata, notch_debug = _notch_filter_gray_with_peaks(
            test,
            peaks,
            notch_radius=safe_notch_radius,
            method="fft_reference_guided_periodic_noise_notch",
            peak_source="golden_test_difference_spectrum",
        )
    else:
        ycrcb = cv2.cvtColor(test, cv2.COLOR_BGR2YCrCb)
        filtered_y, metadata, notch_debug = _notch_filter_gray_with_peaks(
            ycrcb[:, :, 0],
            peaks,
            notch_radius=safe_notch_radius,
            method="fft_reference_guided_periodic_noise_notch",
            peak_source="golden_test_difference_spectrum",
        )
        ycrcb[:, :, 0] = filtered_y
        filtered_gray = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    metadata.update(
        {
            "min_radius_ratio": float(min_radius_ratio),
            "max_radius_ratio": float(max_radius_ratio),
            "peak_sigma": float(peak_sigma),
        }
    )
    return filtered_gray, metadata, notch_debug


def _filter_single_channel(channel: np.ndarray, filter_name: str, cutoff: float) -> np.ndarray:
    response = _filtered_float_single_channel(channel, filter_name, cutoff)
    return _normalize_debug_image(response)


def apply_frequency_filter(
    image: ImageArray,
    filter_name: str = "lowpass",
    cutoff: float | None = None,
) -> ImageArray:
    """Apply an FFT low-pass, high-pass, or band-pass filter."""

    active_cutoff = 0.15 if cutoff is None else float(cutoff)
    if active_cutoff <= 0:
        raise ValueError("cutoff must be positive")

    image_array = np.asarray(image)

    if image_array.ndim == 2:
        return _filter_single_channel(image_array, filter_name, active_cutoff)

    channels = cv2.split(image_array)
    filtered_channels = [
        _filter_single_channel(channel, filter_name, active_cutoff)
        for channel in channels
    ]
    return cv2.merge(filtered_channels)


def log_magnitude_spectrum(image: ImageArray) -> ImageArray:
    """Return a displayable log-magnitude FFT spectrum for a grayscale image."""

    gray = _as_float_gray(image)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    return _normalize_debug_image(magnitude)


def frequency_difference_image(
    reference_image: ImageArray,
    test_image: ImageArray,
    filter_name: str = "bandpass",
    cutoff: float = 0.22,
) -> ImageArray:
    """Highlight local differences after applying the same FFT filter to two images."""

    reference_gray = _as_float_gray(reference_image)
    test_gray = _as_float_gray(test_image)
    if reference_gray.shape != test_gray.shape:
        test_gray = cv2.resize(
            test_gray,
            (reference_gray.shape[1], reference_gray.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    reference_response = _filtered_float_single_channel(reference_gray, filter_name, cutoff)
    test_response = _filtered_float_single_channel(test_gray, filter_name, cutoff)
    difference = np.abs(test_response - reference_response)
    return _normalize_debug_image(difference)
