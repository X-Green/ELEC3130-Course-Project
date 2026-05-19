"""Task 1: board localization and image registration."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .models import AlignmentResult, ImageArray, PipelineConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from models import AlignmentResult, ImageArray, PipelineConfig


def _to_uint8_color(image: ImageArray) -> np.ndarray:
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image *= 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def _board_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    if np.mean(mask == 255) > 0.70:
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(gray.shape, 255, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def _safe_int_tuple(values: tuple[int, int]) -> tuple[int, int]:
    return int(values[0]), int(values[1])


def _draw_keypoints(image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def _draw_matches(
    test: np.ndarray,
    keypoints_test: list[cv2.KeyPoint],
    golden: np.ndarray,
    keypoints_golden: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    inlier_mask: np.ndarray | None = None,
    max_matches: int = 80,
) -> np.ndarray:
    if inlier_mask is not None:
        flat_mask = inlier_mask.ravel().astype(bool)
        matches = [match for match, keep in zip(matches, flat_mask) if keep]

    shown = matches[:max_matches]
    return cv2.drawMatches(
        test,
        keypoints_test,
        golden,
        keypoints_golden,
        shown,
        None,
        matchColor=(0, 220, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def _make_alignment_overlay(golden: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    overlay = np.zeros(golden.shape, dtype=np.uint8)
    overlay[:, :, 1] = golden_gray
    overlay[:, :, 2] = aligned_gray
    return overlay


def _make_checkerboard_overlay(golden: np.ndarray, aligned: np.ndarray, tile_size: int = 80) -> np.ndarray:
    height, width = golden.shape[:2]
    yy, xx = np.indices((height, width))
    selector = ((xx // tile_size + yy // tile_size) % 2).astype(bool)
    checker = golden.copy()
    checker[selector] = aligned[selector]
    return checker


def _make_difference_heatmap(golden: np.ndarray, aligned: np.ndarray, board_mask: np.ndarray) -> np.ndarray:
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(golden_gray, aligned_gray)
    if board_mask is not None:
        diff = cv2.bitwise_and(diff, diff, mask=board_mask)
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    if board_mask is not None:
        heatmap[board_mask == 0] = 0
    return heatmap


def _edge_alignment_diagnostics(
    golden: np.ndarray,
    aligned: np.ndarray,
    board_mask: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    golden_edges = cv2.Canny(golden_gray, 50, 150)
    aligned_edges = cv2.Canny(aligned_gray, 50, 150)
    if board_mask is not None:
        golden_edges = cv2.bitwise_and(golden_edges, golden_edges, mask=board_mask)
        aligned_edges = cv2.bitwise_and(aligned_edges, aligned_edges, mask=board_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    golden_dilated = cv2.dilate(golden_edges, kernel)
    aligned_dilated = cv2.dilate(aligned_edges, kernel)
    golden_count = int(np.count_nonzero(golden_edges))
    aligned_count = int(np.count_nonzero(aligned_edges))
    overlap = int(np.count_nonzero(cv2.bitwise_and(golden_dilated, aligned_dilated)))
    union = int(np.count_nonzero(cv2.bitwise_or(golden_dilated, aligned_dilated)))
    edge_iou = float(overlap / union) if union else 1.0

    distance_map = cv2.distanceTransform(255 - golden_edges, cv2.DIST_L2, 3)
    aligned_edge_pixels = aligned_edges > 0
    chamfer_mean = (
        float(np.mean(distance_map[aligned_edge_pixels]))
        if np.any(aligned_edge_pixels)
        else 0.0
    )
    chamfer_p90 = (
        float(np.percentile(distance_map[aligned_edge_pixels], 90))
        if np.any(aligned_edge_pixels)
        else 0.0
    )

    edge_overlay = np.zeros(golden.shape, dtype=np.uint8)
    edge_overlay[:, :, 1] = golden_edges
    edge_overlay[:, :, 2] = aligned_edges

    mismatch = cv2.bitwise_xor(golden_dilated, aligned_dilated)
    mismatch_heatmap = cv2.applyColorMap(mismatch, cv2.COLORMAP_JET)
    if board_mask is not None:
        mismatch_heatmap[board_mask == 0] = 0

    return (
        {
            "edge_count_golden": golden_count,
            "edge_count_aligned": aligned_count,
            "edge_iou_dilated": edge_iou,
            "edge_chamfer_mean_px": chamfer_mean,
            "edge_chamfer_p90_px": chamfer_p90,
        },
        {
            "alignment_edge_overlay_red_test_green_golden": edge_overlay,
            "alignment_edge_mismatch_heatmap": mismatch_heatmap,
        },
    )


def _transform_corners(
    transform: np.ndarray,
    image_shape: tuple[int, ...],
) -> list[tuple[int, int]]:
    height, width = image_shape[:2]
    corners = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
    return [_safe_int_tuple((round(x), round(y))) for x, y in transformed]


def _draw_projected_test_outline(
    golden: np.ndarray,
    transform: np.ndarray | None,
    test_shape: tuple[int, ...],
) -> np.ndarray:
    debug = golden.copy()
    if transform is None:
        return debug
    points = np.array(_transform_corners(transform, test_shape), dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [points], isClosed=True, color=(0, 0, 255), thickness=3)
    return debug


def _homography_diagnostics(
    transform: np.ndarray,
    image_shape: tuple[int, ...],
) -> dict[str, object]:
    height, width = image_shape[:2]
    matrix = transform / transform[2, 2] if abs(float(transform[2, 2])) > 1e-8 else transform
    affine = matrix[:2, :2]
    determinant = float(np.linalg.det(affine))
    scale_x = float(np.linalg.norm(affine[:, 0]))
    scale_y = float(np.linalg.norm(affine[:, 1]))
    rotation_deg = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
    translation_xy = (float(matrix[0, 2]), float(matrix[1, 2]))

    corners = _transform_corners(matrix, image_shape)
    inside = 0
    for x, y in corners:
        if 0 <= x < width and 0 <= y < height:
            inside += 1

    return {
        "homography": matrix.tolist(),
        "estimated_scale_x": scale_x,
        "estimated_scale_y": scale_y,
        "estimated_rotation_deg": rotation_deg,
        "estimated_translation_xy": translation_xy,
        "affine_determinant": determinant,
        "projected_test_corners": corners,
        "projected_corners_inside_image": inside,
    }


def _register_with_orb(
    golden: np.ndarray,
    test: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object], dict[str, np.ndarray]]:
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=config.registration_features)
    keypoints_golden, descriptors_golden = orb.detectAndCompute(golden_gray, None)
    keypoints_test, descriptors_test = orb.detectAndCompute(test_gray, None)

    metadata: dict[str, object] = {
        "method": "orb_homography",
        "golden_keypoints": len(keypoints_golden),
        "test_keypoints": len(keypoints_test),
        "raw_matches": 0,
        "inlier_matches": 0,
    }
    debug_images: dict[str, np.ndarray] = {
        "alignment_keypoints_golden": _draw_keypoints(golden, keypoints_golden),
        "alignment_keypoints_test": _draw_keypoints(test, keypoints_test),
    }

    if descriptors_golden is None or descriptors_test is None:
        metadata["failure_reason"] = "no_orb_descriptors"
        return test, None, metadata, debug_images

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_test, descriptors_golden)
    matches = sorted(matches, key=lambda match: match.distance)
    metadata["raw_matches"] = len(matches)
    if matches:
        distances = np.array([match.distance for match in matches], dtype=np.float32)
        metadata["match_distance_min"] = float(np.min(distances))
        metadata["match_distance_median"] = float(np.median(distances))
        metadata["match_distance_p90"] = float(np.percentile(distances, 90))

    if len(matches) < config.registration_min_matches:
        metadata["failure_reason"] = "too_few_matches"
        return test, None, metadata, debug_images

    keep = matches[: max(config.registration_min_matches, int(len(matches) * 0.50))]
    debug_images["alignment_matches_top"] = _draw_matches(
        test,
        keypoints_test,
        golden,
        keypoints_golden,
        keep,
    )
    src_points = np.float32([keypoints_test[match.queryIdx].pt for match in keep]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints_golden[match.trainIdx].pt for match in keep]).reshape(-1, 1, 2)

    transform, inlier_mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        config.registration_ransac_reproj_threshold,
    )

    if transform is None:
        metadata["failure_reason"] = "homography_failed"
        return test, None, metadata, debug_images

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    metadata["inlier_matches"] = inliers
    metadata["kept_matches"] = len(keep)
    metadata["inlier_ratio"] = float(inliers / max(len(keep), 1))

    if inliers < config.registration_min_matches:
        metadata["failure_reason"] = "too_few_inliers"
        return test, None, metadata, debug_images

    reprojection_errors = cv2.perspectiveTransform(src_points, transform) - dst_points
    reprojection_distances = np.linalg.norm(reprojection_errors.reshape(-1, 2), axis=1)
    if inlier_mask is not None:
        inlier_distances = reprojection_distances[inlier_mask.ravel().astype(bool)]
    else:
        inlier_distances = reprojection_distances
    if len(inlier_distances) > 0:
        metadata["reprojection_error_mean_px"] = float(np.mean(inlier_distances))
        metadata["reprojection_error_median_px"] = float(np.median(inlier_distances))
        metadata["reprojection_error_p90_px"] = float(np.percentile(inlier_distances, 90))
    metadata.update(_homography_diagnostics(transform, golden.shape))
    debug_images["alignment_matches_inliers"] = _draw_matches(
        test,
        keypoints_test,
        golden,
        keypoints_golden,
        keep,
        inlier_mask,
    )
    debug_images["alignment_projected_outline"] = _draw_projected_test_outline(
        golden,
        transform,
        test.shape,
    )

    height, width = golden.shape[:2]
    aligned = cv2.warpPerspective(
        test,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    metadata["status"] = "completed"
    return aligned, transform, metadata, debug_images


def _register_with_phase_correlation(
    golden: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY).astype(np.float32)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32)
    debug_images: dict[str, np.ndarray] = {}

    if golden_gray.shape != test_gray.shape:
        test = cv2.resize(test, (golden.shape[1], golden.shape[0]), interpolation=cv2.INTER_LINEAR)
        test_gray = cv2.resize(
            test_gray,
            (golden_gray.shape[1], golden_gray.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    window = cv2.createHanningWindow((golden_gray.shape[1], golden_gray.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(golden_gray, test_gray, window)
    dx, dy = shift

    max_shift = 0.25 * min(golden_gray.shape)
    if response < 0.20 or abs(dx) > max_shift or abs(dy) > max_shift:
        identity = np.eye(3, dtype=np.float32)
        return test, identity, {
            "status": "completed",
            "method": "identity_after_low_confidence_phase_correlation",
            "rejected_shift_xy": (float(dx), float(dy)),
            "response": float(response),
        }, debug_images

    transform = np.array(
        [
            [1.0, 0.0, -dx],
            [0.0, 1.0, -dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    aligned = cv2.warpPerspective(
        test,
        transform,
        (golden.shape[1], golden.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned, transform, {
        "status": "completed",
        "method": "phase_correlation_translation",
        "shift_xy": (float(dx), float(dy)),
        "response": float(response),
    }, debug_images


def register_board(
    golden_image: ImageArray,
    test_image: ImageArray,
    config: PipelineConfig | None = None,
) -> AlignmentResult:
    """Align the test PCBA image to the golden reference image."""

    active_config = config or PipelineConfig()
    golden = _to_uint8_color(golden_image)
    test = _to_uint8_color(test_image)

    if golden.shape[:2] != test.shape[:2]:
        test = cv2.resize(
            test,
            (golden.shape[1], golden.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    board_mask = _board_mask(golden)
    aligned, transform, metadata, debug_images = _register_with_orb(golden, test, active_config)

    if transform is None:
        phase_aligned, phase_transform, phase_metadata, phase_debug_images = _register_with_phase_correlation(golden, test)
        phase_metadata["fallback_after"] = metadata
        aligned = phase_aligned
        transform = phase_transform
        metadata = phase_metadata
        debug_images.update(phase_debug_images)

    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(golden_gray, aligned_gray)
    board_pixels = board_mask > 0
    metadata["mean_abs_difference_on_board"] = (
        float(np.mean(diff[board_pixels])) if np.any(board_pixels) else float(np.mean(diff))
    )
    if np.any(board_pixels):
        metadata["median_abs_difference_on_board"] = float(np.median(diff[board_pixels]))
        metadata["p90_abs_difference_on_board"] = float(np.percentile(diff[board_pixels], 90))
    else:
        metadata["median_abs_difference_on_board"] = float(np.median(diff))
        metadata["p90_abs_difference_on_board"] = float(np.percentile(diff, 90))

    debug_images["alignment_overlay_red_test_green_golden"] = _make_alignment_overlay(golden, aligned)
    debug_images["alignment_checkerboard"] = _make_checkerboard_overlay(golden, aligned)
    debug_images["alignment_difference_heatmap"] = _make_difference_heatmap(golden, aligned, board_mask)
    edge_metadata, edge_debug_images = _edge_alignment_diagnostics(golden, aligned, board_mask)
    metadata.update(edge_metadata)
    debug_images.update(edge_debug_images)

    return AlignmentResult(
        aligned_test_image=aligned,
        transform_matrix=transform,
        board_mask=board_mask,
        debug_images=debug_images,
        metadata=metadata,
    )
