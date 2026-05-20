"""Build a one-image qualitative walkthrough for the PCBA pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEMO_STRIPE_CYCLES = 36
DEMO_STRIPE_HARMONIC_CYCLES = 72
DEMO_STRIPE_NOTCH_RADIUS = 16

try:
    from .kicad_synth_dataset import (
        DEFAULT_KICAD_CLI,
        PixelMapper,
        _build_roi_json,
        _iter_blocks,
        _render_board,
        _white_composite,
        parse_board,
    )
    from .models import PipelineConfig
    from .pipeline import run_pipeline, write_visual_outputs
    from .preprocessing import (
        _apply_clahe_color,
        _denoise_preserve_edges,
        _match_histogram_color,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from kicad_synth_dataset import (
        DEFAULT_KICAD_CLI,
        PixelMapper,
        _build_roi_json,
        _iter_blocks,
        _render_board,
        _white_composite,
        parse_board,
    )
    from models import PipelineConfig
    from pipeline import run_pipeline, write_visual_outputs
    from preprocessing import (
        _apply_clahe_color,
        _denoise_preserve_edges,
        _match_histogram_color,
    )


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def _texture_background(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = shape[:2]
    base = np.full((height, width, 3), (228, 231, 224), dtype=np.float32)
    texture = rng.normal(0.0, 3.0, base.shape).astype(np.float32)
    x_wave = np.sin(np.linspace(0.0, np.pi * 5.0, width, dtype=np.float32))[None, :, None]
    y_wave = np.cos(np.linspace(0.0, np.pi * 3.0, height, dtype=np.float32))[:, None, None]
    base += texture + 4.0 * x_wave + 2.5 * y_wave
    return np.clip(base, 0, 255).astype(np.uint8)


def _motion_blur(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.line(kernel, (1, kernel_size - 2), (kernel_size - 2, 1), 1.0, 1)
    kernel /= np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)


def _apply_demo_perturbation_stack(image: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]

    source = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    )
    target = np.float32(
        [
            [64, 28],
            [width - 86, 8],
            [width - 28, height - 72],
            [82, height - 34],
        ]
    )
    camera_transform = cv2.getPerspectiveTransform(source, target)
    background = _texture_background(image.shape, seed + 17)
    camera = cv2.warpPerspective(
        image,
        camera_transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
        dst=background.copy(),
    )

    x_gain = np.linspace(0.80, 1.18, width, dtype=np.float32)[None, :, None]
    y_gain = np.linspace(1.10, 0.86, height, dtype=np.float32)[:, None, None]
    yy, xx = np.indices((height, width), dtype=np.float32)
    radius = np.sqrt(((xx - width * 0.58) / width) ** 2 + ((yy - height * 0.43) / height) ** 2)
    vignette = np.clip(1.13 - 0.78 * radius, 0.74, 1.12)[:, :, None]
    lighting = camera.astype(np.float32) * x_gain * y_gain * vignette

    channel_gains = np.array([0.92, 1.02, 1.12], dtype=np.float32)[None, None, :]
    lighting *= channel_gains
    lighting += np.array([2.0, 0.0, 8.0], dtype=np.float32)[None, None, :]
    lighting = np.clip(lighting, 0, 255).astype(np.uint8)

    noisy = _motion_blur(lighting, kernel_size=7).astype(np.float32)
    noisy += rng.normal(0.0, 5.5, noisy.shape).astype(np.float32)

    stripe_x = np.linspace(0.0, 2.0 * np.pi * DEMO_STRIPE_CYCLES, width, dtype=np.float32)[None, :, None]
    stripe = 18.0 * np.sin(stripe_x)
    stripe += 7.0 * np.sin(
        np.linspace(0.0, 2.0 * np.pi * DEMO_STRIPE_HARMONIC_CYCLES, width, dtype=np.float32)
    )[None, :, None]
    noisy += stripe

    salt_mask = rng.random((height, width)) < 0.0012
    pepper_mask = rng.random((height, width)) < 0.0009
    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    success, encoded = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), 58])
    final = noisy
    if success:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            final = decoded

    return {
        "camera": camera,
        "lighting": lighting,
        "noise_blur_jpeg": final,
        "final": final,
    }


def _difference_heatmap(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    if reference.shape != test.shape:
        test = cv2.resize(test, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(reference, test)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _spatial_normalize_for_demo(golden: np.ndarray, aligned_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    enhanced_golden = _denoise_preserve_edges(_apply_clahe_color(golden))
    enhanced_test = _denoise_preserve_edges(_apply_clahe_color(aligned_test))
    normalized_test = _match_histogram_color(enhanced_test, enhanced_golden)
    return enhanced_golden, normalized_test


def _fft_log_spectrum(gray: np.ndarray) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray_float))
    magnitude = np.log1p(np.abs(spectrum))
    magnitude -= float(np.min(magnitude))
    high = float(np.percentile(magnitude, 99.75))
    if high <= 1e-6:
        high = float(np.max(magnitude))
    if high <= 1e-6:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip(magnitude / high * 255.0, 0, 255).astype(np.uint8)


def _annotate_frequency_points(
    spectrum_gray: np.ndarray,
    points_yx: list[list[int]] | list[tuple[int, int]],
    radius: int,
) -> np.ndarray:
    annotated = cv2.cvtColor(spectrum_gray, cv2.COLOR_GRAY2BGR)
    for row, col in points_yx:
        cv2.circle(annotated, (int(col), int(row)), radius + 5, (0, 0, 255), thickness=3)
    return annotated


def _frequency_point_energy(gray: np.ndarray, points_yx: list[list[int]], radius: int) -> float:
    gray_float = gray.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray_float))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for row, col in points_yx:
        cv2.circle(mask, (int(col), int(row)), max(1, radius), 1, thickness=-1)
    if not np.any(mask):
        return 0.0
    return float(np.mean(magnitude[mask > 0]))


def _demo_frequency_notch_filter(
    image: np.ndarray,
    cycles: tuple[int, ...] = (DEMO_STRIPE_CYCLES, DEMO_STRIPE_HARMONIC_CYCLES),
    radius: int = DEMO_STRIPE_NOTCH_RADIUS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove the known synthetic vertical stripe frequencies for the demo page."""

    image_uint8 = image.astype(np.uint8, copy=False)
    ycrcb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float32)
    rows, cols = luminance.shape
    center_row, center_col = rows // 2, cols // 2
    spectrum = np.fft.fftshift(np.fft.fft2(luminance))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    notch_mask = np.ones((rows, cols), dtype=np.float32)
    marked_points: list[tuple[int, int]] = []

    for cycle in cycles:
        target_offset = int(round(cycle))
        if target_offset <= 0 or center_col + target_offset >= cols or center_col - target_offset < 0:
            continue
        search_half_width = 14
        start_col = max(center_col + target_offset - search_half_width, center_col + 3)
        end_col = min(center_col + target_offset + search_half_width + 1, cols)
        search_band = magnitude[
            max(0, center_row - 3): min(rows, center_row + 4),
            start_col:end_col,
        ]
        if search_band.size == 0:
            continue
        local_row, local_col = np.unravel_index(int(np.argmax(search_band)), search_band.shape)
        positive_row = max(0, center_row - 3) + int(local_row)
        positive_col = start_col + int(local_col)
        negative_row = 2 * center_row - positive_row
        negative_col = 2 * center_col - positive_col
        for row, col in ((positive_row, positive_col), (negative_row, negative_col)):
            if 0 <= row < rows and 0 <= col < cols:
                cv2.circle(notch_mask, (col, row), radius, 0.0, thickness=-1)
                marked_points.append((row, col))

    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * notch_mask))).astype(np.float32)
    filtered += float(np.mean(luminance) - np.mean(filtered))
    ycrcb[:, :, 0] = np.clip(filtered, 0, 255).astype(np.uint8)
    filtered_color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    mask_debug = cv2.cvtColor((notch_mask * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for row, col in marked_points:
        cv2.circle(mask_debug, (col, row), radius + 5, (0, 0, 255), thickness=3)

    metadata = {
        "method": "demo_fixed_fft_notch_for_synthetic_vertical_stripes",
        "active": bool(marked_points),
        "stripe_cycles": list(cycles),
        "notch_radius": int(radius),
        "notched_frequency_count": len(marked_points),
        "peaks_yx": [[int(row), int(col)] for row, col in marked_points],
    }
    return filtered_color, mask_debug, metadata


def _write_demo_frequency_outputs(
    case_dir: Path,
    golden: np.ndarray,
    aligned_test: np.ndarray,
) -> dict[str, Any]:
    enhanced_golden, spatial_test = _spatial_normalize_for_demo(golden, aligned_test)
    frequency_filtered, notch_mask, metadata = _demo_frequency_notch_filter(spatial_test)
    before_gray = cv2.cvtColor(spatial_test, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(frequency_filtered, cv2.COLOR_BGR2GRAY)
    points_yx = metadata.get("peaks_yx", [])
    before_fft = _annotate_frequency_points(
        _fft_log_spectrum(before_gray),
        points_yx,
        int(metadata["notch_radius"]),
    )
    after_fft = _annotate_frequency_points(
        _fft_log_spectrum(after_gray),
        points_yx,
        int(metadata["notch_radius"]),
    )
    stripe_difference = cv2.absdiff(spatial_test, frequency_filtered)
    stripe_difference = cv2.applyColorMap(
        cv2.normalize(cv2.cvtColor(stripe_difference, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX),
        cv2.COLORMAP_TURBO,
    )

    cv2.imwrite(str(case_dir / "spatial_filtered_test.png"), spatial_test)
    cv2.imwrite(str(case_dir / "frequency_notch_demo_filtered.png"), frequency_filtered)
    cv2.imwrite(str(case_dir / "frequency_fft_before_notch.png"), before_fft)
    cv2.imwrite(str(case_dir / "frequency_fft_after_notch.png"), after_fft)
    cv2.imwrite(str(case_dir / "frequency_notch_demo_mask.png"), notch_mask)
    cv2.imwrite(str(case_dir / "frequency_notch_removed_stripes.png"), stripe_difference)
    cv2.imwrite(str(case_dir / "frequency_demo_enhanced_golden.png"), enhanced_golden)

    before_notch_energy = _frequency_point_energy(before_gray, points_yx, int(metadata["notch_radius"]))
    after_notch_energy = _frequency_point_energy(after_gray, points_yx, int(metadata["notch_radius"]))
    metadata.update(
        {
            "before_notch_band_log_energy": before_notch_energy,
            "after_notch_band_log_energy": after_notch_energy,
            "notch_band_energy_reduction_percent": (
                100.0 * max(0.0, before_notch_energy - after_notch_energy) / max(1e-6, before_notch_energy)
            ),
        }
    )
    return metadata


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _metric(report: dict[str, Any], key: str, default: Any = "") -> Any:
    return report.get("alignment", {}).get(key, default)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _image_item(label: str, path: Path, root: Path, note: str) -> dict[str, str]:
    return {
        "label": label,
        "src": _rel(path, root),
        "note": note,
    }


def _find_component(parsed: Any, reference: str) -> Any:
    for component in parsed.components:
        if component.reference == reference:
            return component
    available = ", ".join(sorted(component.reference for component in parsed.components[:40]))
    raise ValueError(f"Component {reference} was not found in {parsed.path}. Available examples: {available}")


def _hide_component_3d_model_block(component_block: str) -> str:
    updated = component_block
    for start, end, model_block in reversed(_iter_blocks(component_block, "model")):
        replacement = re.sub(
            r"\(scale\s+\(xyz\s+[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?\s*\)\s*\)",
            "(scale (xyz 0.001 0.001 0.001))",
            model_block,
            count=1,
        )
        if replacement == model_block:
            insertion = "\n\t\t\t(scale (xyz 0.001 0.001 0.001))"
            replacement = model_block[:-1] + insertion + "\n\t\t)"
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _replace_model_path(component_block: str, new_model_path: str) -> str:
    updated = component_block
    for start, end, model_block in reversed(_iter_blocks(component_block, "model")):
        replacement = re.sub(
            r'\(model\s+"[^"]+"',
            f'(model "{new_model_path}"',
            model_block,
            count=1,
        )
        if replacement == model_block:
            raise ValueError("Could not replace 3D model path")
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _write_compound_variant(
    parsed: Any,
    output_path: Path,
    replacements: dict[str, str],
) -> None:
    pieces: list[str] = []
    cursor = 0
    for component in sorted(
        (component for component in parsed.components if component.reference in replacements),
        key=lambda item: item.span[0],
    ):
        start, end = component.span
        pieces.append(parsed.text[cursor:start])
        pieces.append(replacements[component.reference])
        cursor = end
    pieces.append(parsed.text[cursor:])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(pieces), encoding="utf-8")


def _missing_body_replacement(component: Any) -> str:
    replacement = _hide_component_3d_model_block(component.block)
    if replacement == component.block:
        raise ValueError(f"Component {component.reference} has no editable 3D model block to hide")
    return replacement


def _wrong_component_replacement(component: Any, replacement_model: str) -> str:
    replacement = _replace_model_path(component.block, replacement_model)
    if replacement == component.block:
        raise ValueError(f"Component {component.reference} has no editable 3D model block to replace")
    return replacement


def _component_body_roi_from_component(roi_json: dict[str, Any], component_id: str) -> None:
    component_entry = next((item for item in roi_json.get("components", []) if item["id"] == component_id), None)
    pad_boxes = [
        item["bbox"]
        for item in roi_json.get("solder_joints", [])
        if item.get("component_id") == component_id
    ]
    if component_entry is None or len(pad_boxes) < 2:
        return

    min_x = min(int(box[0]) for box in pad_boxes)
    min_y = min(int(box[1]) for box in pad_boxes)
    max_x = max(int(box[0] + box[2]) for box in pad_boxes)
    max_y = max(int(box[1] + box[3]) for box in pad_boxes)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span_x = max_x - min_x
    span_y = max_y - min_y

    if span_y >= span_x:
        body_w = max(30, int(round(span_x * 0.86)))
        body_h = max(54, int(round(span_y * 0.80)))
    else:
        body_w = max(54, int(round(span_x * 0.80)))
        body_h = max(30, int(round(span_y * 0.86)))

    image_width, image_height = [int(value) for value in roi_json["image_size"]]
    x0 = max(0, int(round(center_x - body_w / 2.0)))
    y0 = max(0, int(round(center_y - body_h / 2.0)))
    x1 = min(image_width, x0 + body_w)
    y1 = min(image_height, y0 + body_h)
    component_entry["bbox"] = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
    component_entry.setdefault("metadata", {})["roi_scope"] = (
        "component body focused between retained pad ROIs for qualitative missing-component inspection"
    )


def _draw_bridge_on_image(
    image_path: Path,
    golden: np.ndarray,
    bridge_joints: list[dict[str, Any]],
) -> list[int]:
    image = _load_image(image_path)
    if len(bridge_joints) < 2:
        raise ValueError("Solder bridge demo requires two solder joint ROIs")
    first_bbox = bridge_joints[0]["bbox"]
    second_bbox = bridge_joints[1]["bbox"]
    first_center = (first_bbox[0] + first_bbox[2] // 2, first_bbox[1] + first_bbox[3] // 2)
    second_center = (second_bbox[0] + second_bbox[2] // 2, second_bbox[1] + second_bbox[3] // 2)
    base_thickness = max(10, min(first_bbox[2], first_bbox[3], second_bbox[2], second_bbox[3]) // 2)
    bridge_bbox = [
        max(0, min(first_bbox[0], second_bbox[0]) - base_thickness * 2),
        max(0, min(first_bbox[1], second_bbox[1]) - base_thickness * 2),
        min(image.shape[1], max(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2]) + base_thickness * 2),
        min(image.shape[0], max(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3]) + base_thickness * 2),
    ]
    bridge_bbox = [bridge_bbox[0], bridge_bbox[1], bridge_bbox[2] - bridge_bbox[0], bridge_bbox[3] - bridge_bbox[1]]
    for multiplier, color in ((1.0, (225, 225, 220)), (1.35, (238, 238, 232)), (1.70, (245, 245, 238))):
        thickness = max(8, int(round(base_thickness * multiplier)))
        cv2.line(image, first_center, second_center, color, thickness, cv2.LINE_AA)
        cv2.circle(image, first_center, max(3, thickness // 2), (250, 250, 245), -1, cv2.LINE_AA)
        cv2.circle(image, second_center, max(3, thickness // 2), (250, 250, 245), -1, cv2.LINE_AA)
    cv2.imwrite(str(image_path), image)
    return bridge_bbox


def _draw_wrong_resistor_body(
    image_path: Path,
    component_bbox: list[int],
) -> None:
    image = _load_image(image_path)
    x, y, width, height = [int(value) for value in component_bbox]
    pad = max(5, min(width, height) // 8)
    body_x0 = x + pad
    body_y0 = y + pad
    body_x1 = x + width - pad
    body_y1 = y + height - pad
    if body_x1 <= body_x0 or body_y1 <= body_y0:
        return

    cv2.rectangle(image, (body_x0, body_y0), (body_x1, body_y1), (42, 46, 50), -1, cv2.LINE_AA)
    stripe_count = 3
    if width >= height:
        for index, color in enumerate(((70, 110, 170), (80, 62, 150), (55, 130, 80)), start=1):
            sx = body_x0 + int((body_x1 - body_x0) * index / (stripe_count + 1))
            cv2.line(image, (sx, body_y0), (sx, body_y1), color, max(3, width // 28), cv2.LINE_AA)
    else:
        for index, color in enumerate(((70, 110, 170), (80, 62, 150), (55, 130, 80)), start=1):
            sy = body_y0 + int((body_y1 - body_y0) * index / (stripe_count + 1))
            cv2.line(image, (body_x0, sy), (body_x1, sy), color, max(3, height // 28), cv2.LINE_AA)
    cv2.rectangle(image, (body_x0, body_y0), (body_x1, body_y1), (212, 218, 214), 2, cv2.LINE_AA)
    cv2.imwrite(str(image_path), image)


def _ensure_compound_defect_assets(args: argparse.Namespace, case_dir: Path, golden_path: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    existing_test = Path(args.test) if args.test else None
    existing_roi = Path(args.roi) if args.roi else None
    if existing_test is not None and existing_roi is not None:
        roi_json = _read_json(existing_roi)
        return existing_test, existing_roi, roi_json, {"case_id": "custom_override"}

    source_pcb = Path(args.source_pcb)
    parsed = parse_board(source_pcb)
    missing_component = _find_component(parsed, args.missing_component)
    wrong_component = _find_component(parsed, args.wrong_component)
    bridge_component = _find_component(parsed, args.bridge_component)
    golden = _load_image(golden_path)
    image_width, image_height = golden.shape[1], golden.shape[0]
    default_roi = _read_json(Path(args.base_roi))
    board_bbox = tuple(int(value) for value in default_roi["board_bbox_px"])

    mapper = PixelMapper(
        edge_bounds_mm=parsed.edge_bounds_mm,
        board_bbox_px=board_bbox,
        image_width=image_width,
        image_height=image_height,
    )

    case_id = (
        f"board01_compound_{args.missing_component}_missing_"
        f"{args.bridge_component}_bridge_{args.wrong_component}_wrong"
    )
    variant_board = case_dir / f"{case_id}.kicad_pcb"
    transparent_render = case_dir / f"{case_id}_transparent.png"
    base_defect = case_dir / "generated_compound_base.png"
    roi_path = case_dir / "roi_compound_defects.json"

    replacements = {
        missing_component.reference: _missing_body_replacement(missing_component),
        wrong_component.reference: _wrong_component_replacement(wrong_component, args.wrong_component_model),
    }
    _write_compound_variant(parsed, variant_board, replacements)
    _render_board(
        Path(args.kicad_cli),
        variant_board,
        transparent_render,
        args.render_width,
        args.render_height,
        force=True,
    )
    _white_composite(transparent_render, base_defect)

    roi_json = _build_roi_json(
        case_id,
        parsed,
        [missing_component, bridge_component, wrong_component],
        mapper,
        roi_path,
    )
    _component_body_roi_from_component(roi_json, missing_component.reference)
    wrong_component_entry = next(
        component
        for component in roi_json["components"]
        if component["id"] == wrong_component.reference
    )
    _draw_wrong_resistor_body(base_defect, wrong_component_entry["bbox"])
    bridge_joints = [
        joint
        for joint in roi_json["solder_joints"]
        if joint.get("component_id") == bridge_component.reference
    ]
    bridge_bbox = _draw_bridge_on_image(base_defect, golden, bridge_joints)
    roi_json["components"] = [
        component
        for component in roi_json["components"]
        if component["id"] in {missing_component.reference, wrong_component.reference}
    ]
    roi_json["solder_joints"] = bridge_joints
    roi_json["metadata"] = {
        "variant": "compound_missing_bridge_wrong_component",
        "missing_component": missing_component.reference,
        "missing_component_footprint": missing_component.footprint,
        "bridge_component": bridge_component.reference,
        "bridge_bbox": bridge_bbox,
        "wrong_component": wrong_component.reference,
        "wrong_component_original": wrong_component.footprint,
        "wrong_component_replacement_model": args.wrong_component_model,
        "method": "KiCad 3D-model edits plus a measurable pixel-level solder bridge on one base image",
    }
    _write_json(roi_path, roi_json)
    metadata = {
        "case_id": case_id,
        "missing_component": missing_component.reference,
        "bridge_component": bridge_component.reference,
        "wrong_component": wrong_component.reference,
        "wrong_component_original": wrong_component.footprint,
        "wrong_component_replacement": "Resistor_SMD:R_0603_1608Metric 3D body",
        "bridge_bbox": bridge_bbox,
    }
    return base_defect, roi_path, roi_json, metadata


def _expand_bbox_for_crop(
    bbox: list[int] | tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    scale: float = 3.6,
) -> tuple[int, int, int, int]:
    x, y, width, height = [int(v) for v in bbox]
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    crop_w = max(int(round(width * scale)), width + 24)
    crop_h = max(int(round(height * scale)), height + 24)
    x0 = max(0, int(round(center_x - crop_w / 2.0)))
    y0 = max(0, int(round(center_y - crop_h / 2.0)))
    x1 = min(image_shape[1], x0 + crop_w)
    y1 = min(image_shape[0], y0 + crop_h)
    x0 = max(0, x1 - crop_w)
    y0 = max(0, y1 - crop_h)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _crop_with_roi_box(
    image: np.ndarray,
    component_bbox: list[int] | tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x, y, width, height = crop_bbox
    crop = image[y:y + height, x:x + width].copy()
    roi_x, roi_y, roi_w, roi_h = [int(v) for v in component_bbox]
    top_left = (max(0, roi_x - x), max(0, roi_y - y))
    bottom_right = (min(width - 1, roi_x + roi_w - x), min(height - 1, roi_y + roi_h - y))
    cv2.rectangle(crop, top_left, bottom_right, (0, 0, 255), 2)
    return crop


def _component_entry(roi_json: dict[str, Any], component_id: str) -> dict[str, Any] | None:
    return next((item for item in roi_json.get("components", []) if item.get("id") == component_id), None)


def _solder_entries(roi_json: dict[str, Any], component_id: str) -> list[dict[str, Any]]:
    return [item for item in roi_json.get("solder_joints", []) if item.get("component_id") == component_id]


def _write_component_crops(
    case_dir: Path,
    golden: np.ndarray,
    base_test: np.ndarray,
    aligned_test: np.ndarray,
    roi_json: dict[str, Any],
    component_id: str,
    prefix: str,
    scale: float = 3.6,
) -> None:
    component = _component_entry(roi_json, component_id)
    if component is None:
        return
    component_bbox = component["bbox"]
    crop_bbox = _expand_bbox_for_crop(component_bbox, golden.shape, scale=scale)
    cv2.imwrite(str(case_dir / f"{prefix}_golden_crop.png"), _crop_with_roi_box(golden, component_bbox, crop_bbox))
    cv2.imwrite(str(case_dir / f"{prefix}_base_crop.png"), _crop_with_roi_box(base_test, component_bbox, crop_bbox))
    cv2.imwrite(str(case_dir / f"{prefix}_aligned_crop.png"), _crop_with_roi_box(aligned_test, component_bbox, crop_bbox))


def _write_solder_bridge_crops(
    case_dir: Path,
    golden: np.ndarray,
    base_test: np.ndarray,
    aligned_test: np.ndarray,
    roi_json: dict[str, Any],
    component_id: str,
) -> None:
    joints = _solder_entries(roi_json, component_id)
    if len(joints) < 2:
        return
    xs = [joint["bbox"][0] for joint in joints]
    ys = [joint["bbox"][1] for joint in joints]
    x2s = [joint["bbox"][0] + joint["bbox"][2] for joint in joints]
    y2s = [joint["bbox"][1] + joint["bbox"][3] for joint in joints]
    bbox = [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)]
    crop_bbox = _expand_bbox_for_crop(bbox, golden.shape, scale=2.5)
    cv2.imwrite(str(case_dir / "solder_bridge_golden_crop.png"), _crop_with_roi_box(golden, bbox, crop_bbox))
    cv2.imwrite(str(case_dir / "solder_bridge_base_crop.png"), _crop_with_roi_box(base_test, bbox, crop_bbox))
    cv2.imwrite(str(case_dir / "solder_bridge_aligned_crop.png"), _crop_with_roi_box(aligned_test, bbox, crop_bbox))


def _write_demo_crops(
    case_dir: Path,
    golden: np.ndarray,
    base_test: np.ndarray,
    aligned_test: np.ndarray,
    roi_json: dict[str, Any],
    case_metadata: dict[str, Any],
) -> None:
    _write_component_crops(
        case_dir,
        golden,
        base_test,
        aligned_test,
        roi_json,
        str(case_metadata["missing_component"]),
        "missing_led",
        scale=3.6,
    )
    _write_component_crops(
        case_dir,
        golden,
        base_test,
        aligned_test,
        roi_json,
        str(case_metadata["wrong_component"]),
        "wrong_component",
        scale=4.2,
    )
    _write_solder_bridge_crops(
        case_dir,
        golden,
        base_test,
        aligned_test,
        roi_json,
        str(case_metadata["bridge_component"]),
    )


def _component_metric(component_defects: list[dict[str, Any]], key: str, default: Any = "none") -> Any:
    if not component_defects:
        return default
    return component_defects[0].get("features", {}).get(key, default)


def _build_payload(
    case_dir: Path,
    output_root: Path,
    report: dict[str, Any],
    roi_path: Path,
    roi_json: dict[str, Any],
    case_metadata: dict[str, Any],
    frequency_metadata: dict[str, Any],
) -> dict[str, Any]:
    final_defects = report.get("defects", [])
    component_defects = report.get("component_defects", [])
    solder_defects = report.get("solder_defects", [])
    alignment = report.get("alignment", {})
    segmentation = report.get("segmentation", {})
    missing_component = str(case_metadata["missing_component"])
    bridge_component = str(case_metadata["bridge_component"])
    wrong_component = str(case_metadata["wrong_component"])

    stages = [
        {
            "title": "1. Disturbed Input",
            "function": "run_pipeline(golden, disturbed_test, roi)",
            "point": f"The test image is one PCB photo-like image with strong perspective shift, textured background, uneven illumination, color cast, motion blur, sensor noise, periodic noise, JPEG compression, and three local defects: {missing_component} missing with pads retained, a solder bridge on {bridge_component}, and {wrong_component} rendered with a resistor body.",
            "inputs": ["Golden reference board", "One disturbed compound-defect test image", "Demo ROI JSON for all three local defects"],
            "outputs": ["Loaded BGR arrays", "ROI map in golden coordinates"],
            "images": [
                _image_item("Golden reference", case_dir / "input_golden.png", output_root, "Clean reference image."),
                _image_item("Missing LED reference", case_dir / "missing_led_golden_crop.png", output_root, f"Reference crop around {missing_component}; the LED body is present."),
                _image_item("Missing LED defect", case_dir / "missing_led_base_crop.png", output_root, f"{missing_component} body is invisible, but pads and board features are retained."),
                _image_item("Solder bridge defect", case_dir / "solder_bridge_base_crop.png", output_root, f"Bridge drawn between the two {bridge_component} pad ROIs."),
                _image_item("Wrong component defect", case_dir / "wrong_component_base_crop.png", output_root, f"{wrong_component} keeps its pads and location, but its 3D body is rendered as a resistor."),
                _image_item("Base defect before disturbance", case_dir / "input_base_defect.png", output_root, "Compound defect render before camera and appearance perturbations."),
                _image_item("Camera perspective and offset", case_dir / "input_camera_transform.png", output_root, "Board projected onto a textured background with clear translation and perspective distortion."),
                _image_item("Lighting and color cast", case_dir / "input_lighting_color.png", output_root, "Uneven illumination and warm camera white-balance shift."),
                _image_item("Final disturbed pipeline input", case_dir / "input_perturbed.png", output_root, "The single test image used for this walkthrough, after blur, noise, periodic bands, and JPEG compression."),
            ],
            "metrics": [
                ["Perturbations", "perspective, translation, textured background, illumination gradient, color cast, motion blur, Gaussian noise, salt/pepper noise, periodic bands, JPEG"],
                ["Expected defects", "missing_component, solder_bridge, wrong_component"],
                ["Missing component", missing_component],
                ["Solder bridge component", bridge_component],
                ["Wrong component", f"{wrong_component}: capacitor body replaced by resistor body"],
                ["ROI file", _rel(roi_path, output_root)],
                ["ROI contents", f"{len(roi_json.get('components', []))} component ROI, {len(roi_json.get('solder_joints', []))} solder joint ROIs"],
            ],
        },
        {
            "title": "2. Board Registration",
            "function": "register_board()",
            "point": "Registration removes camera shift, scale, and rotation so all later ROI comparisons happen in the golden image coordinate system.",
            "inputs": ["Golden image", "Disturbed test image"],
            "outputs": ["Aligned test image", "Homography diagnostics", "Visual alignment checks"],
            "images": [
                _image_item("Aligned test", case_dir / "aligned_test.png", output_root, "Test image warped back into golden coordinates."),
                _image_item("ORB inlier matches", case_dir / "alignment_matches_inliers.png", output_root, "Feature matches kept by RANSAC."),
                _image_item("Checkerboard overlay", case_dir / "alignment_checkerboard.png", output_root, "Alternating golden/test tiles show registration quality."),
                _image_item("Edge overlay", case_dir / "alignment_edge_overlay_red_test_green_golden.png", output_root, "Red/green edge overlap indicates geometric alignment."),
            ],
            "metrics": [
                ["Method", str(alignment.get("method", ""))],
                ["Inlier matches", str(alignment.get("inlier_matches", ""))],
                ["Edge IoU", _fmt(alignment.get("edge_iou_dilated"))],
                ["Chamfer mean px", _fmt(alignment.get("edge_chamfer_mean_px"))],
                ["Reprojection p90 px", _fmt(alignment.get("reprojection_error_p90_px"))],
            ],
        },
        {
            "title": "3. Spatial Enhancement And Normalization",
            "function": "enhance_and_segment(): CLAHE -> bilateral filter -> histogram matching",
            "point": "Spatial filtering handles local contrast, blur-preserving denoising, and color normalization first. Its output still contains the synthetic periodic stripe noise, which is easier to isolate in the frequency domain.",
            "inputs": ["Golden image", "Aligned test image"],
            "outputs": ["Spatially enhanced golden image", "Spatially normalized test image", "Board-level difference view"],
            "images": [
                _image_item("Enhanced golden", case_dir / "frequency_demo_enhanced_golden.png", output_root, "LAB CLAHE plus edge-preserving bilateral filtering."),
                _image_item("Spatial-filtered test", case_dir / "spatial_filtered_test.png", output_root, "Spatial output before FFT notch filtering; periodic stripe bands are intentionally still visible."),
                _image_item("Alignment difference heatmap", case_dir / "alignment_difference_heatmap.png", output_root, "Residual pixel differences after registration."),
            ],
            "metrics": [
                ["Mean abs diff on board", _fmt(alignment.get("mean_abs_difference_on_board"))],
                ["Median abs diff on board", _fmt(alignment.get("median_abs_difference_on_board"))],
                ["Segmentation status", str(segmentation.get("status", ""))],
            ],
        },
        {
            "title": "4. Frequency Notch Filtering",
            "function": "demo FFT notch filter after spatial filtering",
            "point": "A simple FFT notch filter removes the known vertical stripe frequencies from the demo image. The before/after Fourier spectra make the effect visible: narrow off-center peaks are suppressed while component geometry remains.",
            "inputs": ["Spatial-filtered test image", "Known synthetic stripe frequencies"],
            "outputs": ["Frequency-filtered test image", "Notch mask", "FFT before filtering", "FFT after filtering", "Removed-stripe difference view"],
            "images": [
                _image_item("Spatial-filtered test", case_dir / "spatial_filtered_test.png", output_root, "Input to the frequency filter; vertical periodic bands are still present."),
                _image_item("FFT before notch", case_dir / "frequency_fft_before_notch.png", output_root, "Log-magnitude Fourier spectrum before notch filtering; stripe noise appears as symmetric peaks."),
                _image_item("Notch mask", case_dir / "frequency_notch_demo_mask.png", output_root, "White frequencies pass; red circles mark the stripe frequencies removed by the demo notch filter."),
                _image_item("Frequency-filtered test", case_dir / "frequency_notch_demo_filtered.png", output_root, "Output after removing the periodic bands in the luminance channel."),
                _image_item("FFT after notch", case_dir / "frequency_fft_after_notch.png", output_root, "The same spectrum after filtering, with the stripe peaks attenuated."),
                _image_item("Removed stripe evidence", case_dir / "frequency_notch_removed_stripes.png", output_root, "Absolute difference between spatial output and frequency-filtered output, highlighting removed periodic content."),
            ],
            "metrics": [
                ["Notch active", str(frequency_metadata.get("active", ""))],
                ["Synthetic stripe cycles", ", ".join(str(value) for value in frequency_metadata.get("stripe_cycles", []))],
                ["Notched frequency points", str(frequency_metadata.get("notched_frequency_count", ""))],
                ["Notch radius px", str(frequency_metadata.get("notch_radius", ""))],
                ["Notch-band energy reduction", f"{_fmt(frequency_metadata.get('notch_band_energy_reduction_percent'), 1)}%"],
                ["Pipeline debug notch active", str(segmentation.get("frequency_domain", {}).get("notch_filter", {}).get("active", ""))],
            ],
        },
        {
            "title": "5. Segmentation And Morphology",
            "function": "enhance_and_segment() mask builders",
            "point": "After spatial and frequency filtering, thresholding, Canny edges, color segmentation, and morphology turn the normalized image into interpretable masks.",
            "inputs": ["Frequency-filtered normalized test image", "ROI map"],
            "outputs": ["Board mask", "Component mask", "Solder mask"],
            "images": [
                _image_item("Frequency-filtered test", case_dir / "frequency_notch_demo_filtered.png", output_root, "Cleaner input after stripe suppression, shown before mask construction."),
                _image_item("Board mask", case_dir / "board_mask.png", output_root, "Board localization after thresholding and morphology."),
                _image_item("Component mask", case_dir / "component_mask.png", output_root, "Canny/morphology evidence constrained by component ROIs."),
                _image_item("Solder mask", case_dir / "solder_mask.png", output_root, "Bright low-saturation solder evidence constrained by solder ROIs."),
            ],
            "metrics": [
                ["Methods", ", ".join(segmentation.get("methods", [])[:5])],
                ["ROI constrained", str(segmentation.get("has_roi_map", ""))],
            ],
        },
        {
            "title": "6. Component And Solder Inspection",
            "function": "inspect_components(), inspect_solder_joints()",
            "point": f"The pipeline measures local ROI evidence instead of using a learned model. In this demo, the component branch sees {missing_component} as a missing body with retained pads and {wrong_component} as a wrong component body, while the solder branch checks the {bridge_component} bridge.",
            "inputs": ["Golden/test ROI crops", "Component mask", "Solder mask"],
            "outputs": ["Component candidates", "Solder candidates"],
            "images": [
                _image_item("Golden LED crop", case_dir / "missing_led_golden_crop.png", output_root, "The expected component body inside the ROI."),
                _image_item("Aligned missing LED crop", case_dir / "missing_led_aligned_crop.png", output_root, "After registration, the same ROI shows retained pads but no LED body."),
                _image_item("Wrong component reference", case_dir / "wrong_component_golden_crop.png", output_root, f"Golden {wrong_component} capacitor body."),
                _image_item("Wrong component aligned", case_dir / "wrong_component_aligned_crop.png", output_root, f"Aligned test crop with {wrong_component} rendered as a resistor body."),
                _image_item("Solder bridge aligned", case_dir / "solder_bridge_aligned_crop.png", output_root, f"Aligned test crop around the bridge on {bridge_component}."),
                _image_item("Component mask", case_dir / "component_mask.png", output_root, "Component-region evidence used by inspect_components()."),
                _image_item("Intermediate overlay", case_dir / "defect_overlay.png", output_root, "Detected candidates drawn over the aligned image."),
            ],
            "metrics": [
                ["Component candidates", str(len(component_defects))],
                ["Solder candidates", str(len(solder_defects))],
                ["Top component label", str(component_defects[0].get("defect_type", "none") if component_defects else "none")],
                ["Template score", _fmt(_component_metric(component_defects, "template_score"))],
                ["Direct ROI difference", _fmt(_component_metric(component_defects, "direct_difference_score"))],
                ["Edge-density ratio", _fmt(_component_metric(component_defects, "edge_density_ratio"))],
            ],
        },
        {
            "title": "7. Fusion And Final Qualitative Result",
            "function": "fuse_and_classify_defects(), write_visual_outputs()",
            "point": "Final fusion deduplicates candidates and produces the AOI-style overlay and JSON report. For presentation, this is the visible end of the pipeline story.",
            "inputs": ["Component candidates", "Solder candidates", "Segmentation result"],
            "outputs": ["Final defect list", "Defect overlay", "report.json"],
            "images": [
                _image_item("Final defect overlay", case_dir / "defect_overlay.png", output_root, "Final rule-based labels and bounding boxes."),
                _image_item("Raw disturbed-vs-golden heatmap", case_dir / "input_disturbance_heatmap.png", output_root, "How different the input looked before registration and normalization."),
                _image_item("Noise/blur/JPEG input", case_dir / "input_noise_blur_jpeg.png", output_root, "Final disturbance stack before the registration stage."),
                _image_item("Aligned missing LED crop", case_dir / "missing_led_aligned_crop.png", output_root, "Local visual proof that the missing LED remains visible after geometric correction."),
                _image_item("Aligned solder bridge crop", case_dir / "solder_bridge_aligned_crop.png", output_root, "Local bridge evidence after geometric correction."),
                _image_item("Aligned wrong component crop", case_dir / "wrong_component_aligned_crop.png", output_root, "Local wrong-component evidence after geometric correction."),
            ],
            "metrics": [
                ["Final defects", str(len(final_defects))],
                ["Final labels", ", ".join(sorted({item.get("defect_type", "") for item in final_defects})) or "none"],
                ["Report", "qualitative_case/report.json"],
            ],
        },
    ]

    tree_nodes = [
        {
            "id": "run_pipeline",
            "parents": [],
            "title": "run_pipeline",
            "stage_title": "Pipeline Orchestrator",
            "function": "run_pipeline(golden_image_path, test_image_path, roi_json_path, config)",
            "kind": "root",
            "col": 1,
            "row": 2,
            "point": "The top-level function calls every classical image-processing stage in order and keeps the data flow reference-based.",
            "inputs": ["Golden image path", "Disturbed test image path", "ROI JSON path", "PipelineConfig"],
            "outputs": ["PipelineResult", "Final defects", "Debug images and metadata"],
            "methods": ["Reference-based AOI pipeline", "Sequential orchestration", "Structured report serialization"],
            "images": [
                _image_item("Final disturbed pipeline input", case_dir / "input_perturbed.png", output_root, "The single photo-like test image processed by the pipeline."),
                _image_item("Final defect overlay", case_dir / "defect_overlay.png", output_root, "The visible output produced after the full pipeline."),
                _image_item("Aligned missing LED crop", case_dir / "missing_led_aligned_crop.png", output_root, f"Local crop around {missing_component} after geometric correction."),
                _image_item("Aligned solder bridge crop", case_dir / "solder_bridge_aligned_crop.png", output_root, f"Local crop around {bridge_component} after geometric correction."),
                _image_item("Aligned wrong component crop", case_dir / "wrong_component_aligned_crop.png", output_root, f"Local crop around {wrong_component} after geometric correction."),
            ],
            "metrics": [
                ["Case", str(case_metadata["case_id"])],
                ["Final defects", str(len(final_defects))],
            ],
        },
        {
            "id": "load_inputs",
            "parents": ["run_pipeline"],
            "title": "load inputs",
            "stage_title": "Input Loading",
            "function": "load_image(), load_roi_json()",
            "kind": "input",
            "col": 2,
            "row": 2,
            "point": "The pipeline starts from a clean reference, one disturbed PCB image, and KiCad ROI metadata in golden-board coordinates.",
            "inputs": stages[0]["inputs"],
            "outputs": stages[0]["outputs"],
            "methods": ["BGR image decoding", "ROI JSON parsing", "Golden-coordinate region map"],
            "images": stages[0]["images"],
            "metrics": stages[0]["metrics"],
        },
        {
            "id": "register_board",
            "parents": ["load_inputs"],
            "title": "register_board",
            "stage_title": "Board Registration",
            "function": "register_board()",
            "kind": "process",
            "col": 3,
            "row": 2,
            "point": stages[1]["point"],
            "inputs": stages[1]["inputs"],
            "outputs": stages[1]["outputs"],
            "methods": ["ORB keypoints", "Brute-force Hamming matches", "RANSAC homography", "Perspective warping", "Edge-alignment diagnostics"],
            "images": stages[1]["images"],
            "metrics": stages[1]["metrics"],
        },
        {
            "id": "enhance_and_segment",
            "parents": ["register_board"],
            "title": "enhance_and_segment",
            "stage_title": "Enhancement And Segmentation",
            "function": "enhance_and_segment()",
            "kind": "process",
            "col": 4,
            "row": 2,
            "point": "This stage first applies spatial filtering, then a frequency-domain stripe suppressor, then mask construction for local inspection.",
            "inputs": stages[2]["inputs"],
            "outputs": ["Spatially enhanced images", "Frequency-filtered image", "Board mask", "Component mask", "Solder mask"],
            "methods": ["LAB CLAHE", "Bilateral filtering", "Histogram matching", "FFT notch filtering", "Otsu thresholding", "Canny edges", "Morphological cleanup"],
            "images": stages[2]["images"] + stages[3]["images"][:4] + stages[4]["images"],
            "metrics": stages[2]["metrics"] + stages[3]["metrics"] + stages[4]["metrics"],
        },
        {
            "id": "frequency_domain_filter",
            "parents": ["enhance_and_segment"],
            "title": "frequency domain filter",
            "stage_title": "Frequency Domain Filter",
            "function": "FFT notch filter after spatial filtering",
            "kind": "process",
            "col": 5,
            "row": 2,
            "point": stages[3]["point"],
            "inputs": stages[3]["inputs"],
            "outputs": stages[3]["outputs"],
            "methods": ["2D FFT", "Log-magnitude spectrum", "Symmetric notch mask", "Inverse FFT", "Before/after spectrum comparison"],
            "images": stages[3]["images"],
            "metrics": stages[3]["metrics"],
        },
        {
            "id": "inspect_components",
            "parents": ["frequency_domain_filter"],
            "title": "inspect_components",
            "stage_title": "Component Inspection Branch",
            "function": "inspect_components()",
            "kind": "branch",
            "col": 6,
            "row": 1,
            "point": f"Component ROIs are compared against the golden reference. This branch sees both {missing_component} missing from its retained pads and {wrong_component} changed from a capacitor body to a resistor body.",
            "inputs": ["Golden/test component ROI crops", "Component mask", "ROI map"],
            "outputs": ["Component defect candidates"],
            "methods": ["Template correlation", "ROI difference fraction", "Edge-density comparison", "Shift/rotation evidence"],
            "images": [
                _image_item("Golden LED crop", case_dir / "missing_led_golden_crop.png", output_root, "Reference component ROI with LED body present."),
                _image_item("Aligned missing LED crop", case_dir / "missing_led_aligned_crop.png", output_root, "Registered test ROI with pads retained and body missing."),
                _image_item("Wrong component reference", case_dir / "wrong_component_golden_crop.png", output_root, "Reference capacitor body inside the ROI."),
                _image_item("Wrong component aligned", case_dir / "wrong_component_aligned_crop.png", output_root, "Registered test ROI with a resistor body in the same footprint location."),
                _image_item("Component mask", case_dir / "component_mask.png", output_root, "Component-region evidence used by the branch."),
                _image_item("Intermediate overlay", case_dir / "defect_overlay.png", output_root, "Component candidates are drawn together with any solder candidates."),
            ],
            "metrics": [
                ["Component candidates", str(len(component_defects))],
                ["Component labels", ", ".join(item.get("defect_type", "") for item in component_defects) or "none"],
                ["Top component label", str(component_defects[0].get("defect_type", "none") if component_defects else "none")],
                ["Template score", _fmt(_component_metric(component_defects, "template_score"))],
                ["Direct ROI difference", _fmt(_component_metric(component_defects, "direct_difference_score"))],
                ["Edge-density ratio", _fmt(_component_metric(component_defects, "edge_density_ratio"))],
            ],
        },
        {
            "id": "inspect_solder_joints",
            "parents": ["frequency_domain_filter"],
            "title": "inspect_solder_joints",
            "stage_title": "Solder Inspection Branch",
            "function": "inspect_solder_joints()",
            "kind": "branch",
            "col": 6,
            "row": 3,
            "point": f"Solder ROIs are inspected with local solder-area, shape, difference, and bridge-corridor evidence. The visible solder bridge is drawn between the two {bridge_component} pad ROIs.",
            "inputs": ["Golden/test solder ROI crops", "Solder mask", "ROI map"],
            "outputs": ["Solder defect candidates", "Bridge candidates"],
            "methods": ["Solder color segmentation", "Local difference statistics", "Connected components", "Pad-pair bridge corridor measurement"],
            "images": [
                _image_item("Solder bridge reference", case_dir / "solder_bridge_golden_crop.png", output_root, f"Clean {bridge_component} pads before bridge drawing."),
                _image_item("Solder bridge aligned", case_dir / "solder_bridge_aligned_crop.png", output_root, "Registered test ROI with the bridge visible between pads."),
                _image_item("Solder mask", case_dir / "solder_mask.png", output_root, "Solder evidence after segmentation and morphology."),
                _image_item("Intermediate overlay", case_dir / "defect_overlay.png", output_root, "Any solder candidates are drawn on the aligned board."),
            ],
            "metrics": [
                ["Solder candidates", str(len(solder_defects))],
                ["Solder labels", ", ".join(item.get("defect_type", "") for item in solder_defects) or "none"],
                ["Top solder label", str(solder_defects[0].get("defect_type", "none") if solder_defects else "none")],
            ],
        },
        {
            "id": "fuse_and_classify_defects",
            "parents": ["inspect_components", "inspect_solder_joints"],
            "title": "fuse_and_classify",
            "stage_title": "Candidate Fusion",
            "function": "fuse_and_classify_defects()",
            "kind": "fusion",
            "col": 7,
            "row": 2,
            "point": "The branch outputs are merged into one AOI-style candidate list with labels, scores, bounding boxes, and metadata.",
            "inputs": ["Component candidates", "Solder candidates", "Segmentation result", "ROI map"],
            "outputs": ["Final defect list"],
            "methods": ["Candidate aggregation", "Rule-based label preservation", "Final ranking metadata"],
            "images": [
                _image_item("Final defect overlay", case_dir / "defect_overlay.png", output_root, "Final labels and bounding boxes after candidate fusion."),
            ],
            "metrics": [
                ["Final defects", str(len(final_defects))],
                ["Final labels", ", ".join(sorted({item.get("defect_type", "") for item in final_defects})) or "none"],
            ],
        },
        {
            "id": "write_visual_outputs",
            "parents": ["fuse_and_classify_defects"],
            "title": "write outputs",
            "stage_title": "Visual And JSON Output",
            "function": "write_visual_outputs()",
            "kind": "output",
            "col": 8,
            "row": 2,
            "point": "The final stage writes the aligned image, masks, frequency images, overlay, and report JSON for presentation.",
            "inputs": ["PipelineResult", "Output directory"],
            "outputs": ["PNG artifacts", "report.json", "Static demo assets"],
            "methods": ["OpenCV image writing", "Overlay rendering", "JSON serialization"],
            "images": stages[6]["images"],
            "metrics": stages[6]["metrics"],
        },
    ]

    return {
        "case_id": f"qualitative_{case_metadata['case_id']}_disturbed",
        "source_case": str(case_metadata["case_id"]),
        "golden": "data/kicad_synth/golden/board01_top.png",
        "roi": _rel(roi_path, output_root),
        "case_metadata": case_metadata,
        "frequency_metadata": frequency_metadata,
        "final_defects": final_defects,
        "stages": stages,
        "tree": tree_nodes,
    }


def _html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qualitative PCBA Pipeline Walkthrough</title>
  <style>
    :root {{
      --bg: #f3f4f1;
      --panel: #ffffff;
      --ink: #202421;
      --muted: #5e665f;
      --line: #d8ddd6;
      --accent: #0f766e;
      --accent-soft: #e6f3ef;
      --input: #1d4ed8;
      --process: #0f766e;
      --branch: #9a5b16;
      --debug: #6d5bd0;
      --fusion: #b42318;
      --output: #374151;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
      overflow-x: hidden;
    }}
    header {{
      padding: 22px 24px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 6px; font-size: 25px; line-height: 1.2; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; }}
    p {{ margin: 0 0 10px; color: var(--muted); line-height: 1.5; }}
    code {{ background: #eef1ed; border: 1px solid var(--line); padding: 2px 5px; overflow-wrap: anywhere; }}
    .panel,
    .overview,
    .tree-area,
    .tree-grid,
    .detail-grid,
    .box,
    figure {{
      min-width: 0;
    }}
    main {{
      padding: 16px;
      display: grid;
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 14px;
    }}
    .overview {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 14px;
      align-items: start;
    }}
    .case-note {{
      border-left: 4px solid var(--accent);
      padding-left: 12px;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .case-note * {{
      overflow-wrap: anywhere;
    }}
    .tree-area {{
      position: relative;
      overflow-x: auto;
      padding: 18px;
      background: #f9faf7;
      border: 1px solid var(--line);
    }}
    .tree-links {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      overflow: visible;
      z-index: 1;
    }}
    .tree-links path {{
      fill: none;
      stroke: #a7b0a8;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .tree-grid {{
      position: relative;
      z-index: 2;
      display: grid;
      grid-template-columns: repeat(8, 165px);
      grid-template-rows: repeat(3, 118px);
      gap: 28px;
      min-width: 1513px;
    }}
    button {{
      font-family: inherit;
    }}
    .tree-node {{
      grid-column: var(--col);
      grid-row: var(--row);
      border: 1px solid var(--line);
      background: #ffffff;
      padding: 10px;
      text-align: left;
      cursor: pointer;
      color: var(--ink);
      min-height: 118px;
      display: grid;
      align-content: start;
      gap: 6px;
      border-left: 5px solid var(--process);
    }}
    .tree-node:hover,
    .tree-node.active {{
      border-color: var(--accent);
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .tree-node.input {{ border-left-color: var(--input); }}
    .tree-node.process {{ border-left-color: var(--process); }}
    .tree-node.branch {{ border-left-color: var(--branch); }}
    .tree-node.debug {{ border-left-color: var(--debug); }}
    .tree-node.fusion {{ border-left-color: var(--fusion); }}
    .tree-node.output {{ border-left-color: var(--output); }}
    .node-title {{
      font-weight: 700;
      font-size: 13px;
      line-height: 1.25;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .node-function {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .node-kind {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) 320px;
      gap: 14px;
      align-items: start;
    }}
    .boxes {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .box {{
      border: 1px solid var(--line);
      background: #fbfcfb;
      padding: 10px;
      min-width: 0;
    }}
    .box.full {{ grid-column: 1 / -1; }}
    ul {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.45; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
    th {{ width: 42%; color: var(--muted); font-weight: 400; }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      background: #eef0ed;
    }}
    figcaption {{
      padding: 7px 8px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
    }}
    img {{
      display: block;
      width: 100%;
      height: 280px;
      object-fit: contain;
      background: #eef0ed;
    }}
    .caption {{
      min-height: 44px;
      padding: 7px 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      border-top: 1px solid var(--line);
    }}
    .json-panel {{
      margin-top: 16px;
    }}
    pre {{
      margin: 0;
      padding: 10px;
      background: #f7f8f6;
      border: 1px solid var(--line);
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      .overview, .detail-grid {{ grid-template-columns: 1fr; }}
      .tree-grid {{
        min-width: 0;
        grid-template-columns: 1fr;
        grid-template-rows: none;
        gap: 10px;
      }}
      .tree-node {{
        grid-column: 1 !important;
        grid-row: auto !important;
        min-height: 0;
      }}
      .tree-links {{ display: none; }}
      .boxes, .image-grid {{ grid-template-columns: 1fr; }}
      img {{ height: auto; max-height: 320px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Qualitative PCBA Pipeline Walkthrough</h1>
    <p>One disturbed PCB image, followed through every major function in <code>src/pipeline.py</code>. Function names stay in English; the analysis focuses on qualitative visual progress.</p>
  </header>
  <main>
    <section class="panel overview">
      <div>
        <h2>Pipeline Input/Output Tree</h2>
        <p>Click a node to inspect that function's inputs, outputs, classical image-processing methods, metrics, and saved visual evidence.</p>
      </div>
      <div class="case-note">
        <p><strong id="caseId"></strong></p>
        <p>Single photo-like input with camera disturbance plus missing component, solder bridge, and wrong component evidence.</p>
      </div>
    </section>
    <section class="panel">
      <div class="tree-area" id="treeArea">
        <svg class="tree-links" id="treeLinks"></svg>
        <div class="tree-grid" id="tree"></div>
      </div>
    </section>
    <section class="panel" id="detail"></section>
    <section class="panel json-panel">
      <h2>Final Defect JSON</h2>
      <pre id="json"></pre>
    </section>
  </main>
  <script>
    const DATA = {data_json};
    let activeId = 'run_pipeline';
    const tree = document.getElementById('tree');
    const treeArea = document.getElementById('treeArea');
    const treeLinks = document.getElementById('treeLinks');
    const detail = document.getElementById('detail');
    const json = document.getElementById('json');
    document.getElementById('caseId').textContent = DATA.case_id;

    function renderList(items) {{
      return `<ul>${{items.map((item) => `<li>${{item}}</li>`).join('')}}</ul>`;
    }}
    function renderMetrics(rows) {{
      return `<table><tbody>${{rows.map(([label, value]) => `<tr><th>${{label}}</th><td>${{value}}</td></tr>`).join('')}}</tbody></table>`;
    }}
    function renderImages(items) {{
      return `<div class="image-grid">${{items.map((item) => `
        <figure>
          <figcaption>${{item.label}}</figcaption>
          <img src="${{item.src}}" alt="${{item.label}}">
          <div class="caption">${{item.note}}</div>
        </figure>
      `).join('')}}</div>`;
    }}

    function nodeById(id) {{
      return DATA.tree.find((node) => node.id === id) || DATA.tree[0];
    }}

    function renderTree() {{
      tree.innerHTML = DATA.tree.map((node) => `
        <button
          class="tree-node ${{node.kind}} ${{node.id === activeId ? 'active' : ''}}"
          data-id="${{node.id}}"
          style="--col: ${{node.col}}; --row: ${{node.row}};"
          aria-label="${{node.stage_title}}">
          <span class="node-kind">${{node.kind}}</span>
          <span class="node-title">${{node.title}}</span>
          <span class="node-function">${{node.function}}</span>
        </button>
      `).join('');
      tree.querySelectorAll('button').forEach((button) => {{
        button.addEventListener('click', () => {{
          activeId = button.dataset.id;
          render();
        }});
      }});
      requestAnimationFrame(drawLinks);
    }}

    function drawLinks() {{
      const areaBox = treeArea.getBoundingClientRect();
      const width = treeArea.scrollWidth;
      const height = treeArea.scrollHeight;
      treeLinks.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      treeLinks.setAttribute('width', width);
      treeLinks.setAttribute('height', height);
      const paths = [];
      DATA.tree.forEach((node) => {{
        node.parents.forEach((parentId) => {{
          const parentEl = tree.querySelector(`[data-id="${{parentId}}"]`);
          const childEl = tree.querySelector(`[data-id="${{node.id}}"]`);
          if (!parentEl || !childEl) return;
          const parentBox = parentEl.getBoundingClientRect();
          const childBox = childEl.getBoundingClientRect();
          const x1 = parentBox.right - areaBox.left + treeArea.scrollLeft;
          const y1 = parentBox.top + parentBox.height / 2 - areaBox.top + treeArea.scrollTop;
          const x2 = childBox.left - areaBox.left + treeArea.scrollLeft;
          const y2 = childBox.top + childBox.height / 2 - areaBox.top + treeArea.scrollTop;
          const mid = x1 + Math.max(24, (x2 - x1) / 2);
          paths.push(`<path d="M ${{x1}} ${{y1}} L ${{mid}} ${{y1}} L ${{mid}} ${{y2}} L ${{x2}} ${{y2}}" />`);
        }});
      }});
      treeLinks.innerHTML = paths.join('');
    }}

    function render() {{
      const item = nodeById(activeId);
      renderTree();
      detail.innerHTML = `
        <h2>${{item.stage_title}}</h2>
        <p>${{item.point}}</p>
        <div class="detail-grid">
          <div class="boxes">
            <div class="box"><h3>Input</h3>${{renderList(item.inputs)}}</div>
            <div class="box"><h3>Function</h3><p><code>${{item.function}}</code></p></div>
            <div class="box"><h3>Output</h3>${{renderList(item.outputs)}}</div>
            <div class="box"><h3>DIP Methods</h3>${{renderList(item.methods)}}</div>
            <div class="box full"><h3>Qualitative Metrics</h3>${{renderMetrics(item.metrics)}}</div>
            <div class="box full"><h3>Visual Evidence</h3>${{renderImages(item.images)}}</div>
          </div>
          <div class="box">
            <h3>How To Present This Stage</h3>
            <p>${{item.point}}</p>
          </div>
        </div>
      `;
      json.textContent = JSON.stringify(DATA.final_defects, null, 2);
    }}
    window.addEventListener('resize', drawLinks);
    render();
  </script>
</body>
</html>
"""


def build_demo(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir)
    case_dir = output_root / "qualitative_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    golden_path = Path(args.golden)
    base_test_path, roi_path, roi_json, case_metadata = _ensure_compound_defect_assets(args, case_dir, golden_path)
    golden = _load_image(golden_path)
    base_test = _load_image(base_test_path)
    perturbation_stack = _apply_demo_perturbation_stack(base_test, seed=args.seed)
    perturbed = perturbation_stack["final"]

    cv2.imwrite(str(case_dir / "input_golden.png"), golden)
    cv2.imwrite(str(case_dir / "input_base_defect.png"), base_test)
    cv2.imwrite(str(case_dir / "input_camera_transform.png"), perturbation_stack["camera"])
    cv2.imwrite(str(case_dir / "input_lighting_color.png"), perturbation_stack["lighting"])
    cv2.imwrite(str(case_dir / "input_noise_blur_jpeg.png"), perturbation_stack["noise_blur_jpeg"])
    cv2.imwrite(str(case_dir / "input_perturbed.png"), perturbed)
    cv2.imwrite(str(case_dir / "input_disturbance_heatmap.png"), _difference_heatmap(golden, perturbed))

    perturbed_path = case_dir / "input_perturbed.png"
    result = run_pipeline(
        golden_path,
        perturbed_path,
        roi_path,
        PipelineConfig(debug=True),
    )
    write_visual_outputs(result, case_dir)
    cv2.imwrite(str(case_dir / "enhanced_golden.png"), result.segmentation.enhanced_golden_image)
    cv2.imwrite(str(case_dir / "enhanced_test.png"), result.segmentation.enhanced_test_image)
    frequency_metadata = _write_demo_frequency_outputs(case_dir, golden, result.alignment.aligned_test_image)
    _write_demo_crops(case_dir, golden, base_test, result.alignment.aligned_test_image, roi_json, case_metadata)

    report = _read_json(case_dir / "report.json")
    payload = _build_payload(case_dir, output_root, report, roi_path, roi_json, case_metadata, frequency_metadata)
    _write_json(output_root / "qualitative_data.json", payload)
    html_path = output_root / "index.html"
    html_path.write_text(_html(payload), encoding="utf-8")
    print(f"Wrote {html_path}")
    print(f"Final defects: {len(report.get('defects', []))}")
    for defect in report.get("defects", []):
        print(f"- {defect.get('defect_type')} at {defect.get('bbox')} from {defect.get('source')}")
    return html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a qualitative one-image pipeline walkthrough.")
    parser.add_argument("--golden", default="data/kicad_synth/golden/board01_top.png", help="Golden reference image.")
    parser.add_argument("--test", default=None, help="Optional pre-rendered base defect test image override.")
    parser.add_argument("--roi", default=None, help="Optional ROI JSON override. When omitted, a compound-defect demo ROI is generated.")
    parser.add_argument(
        "--source-pcb",
        default=r"E:\SmartTBBatteryCharger\hardware_test_new_2\SmartTBBatteryChargerNewTest.kicad_pcb",
        help="Source KiCad PCB used to generate the missing-LED demo variant.",
    )
    parser.add_argument("--missing-component", default="D27", help="Reference designator to hide as a missing LED/component.")
    parser.add_argument("--bridge-component", default="D11", help="Reference designator whose two pads receive a visible solder bridge.")
    parser.add_argument("--wrong-component", default="C48", help="Reference designator rendered with the wrong 3D component body.")
    parser.add_argument(
        "--wrong-component-model",
        default="${KICAD9_3DMODEL_DIR}/Resistor_SMD.3dshapes/R_0603_1608Metric.wrl",
        help="3D model path used for the wrong-component body.",
    )
    parser.add_argument("--kicad-cli", default=str(DEFAULT_KICAD_CLI), help="Path to kicad-cli.exe.")
    parser.add_argument("--base-roi", default="data/kicad_synth/roi/board01.json", help="Existing board01 ROI used for board bbox calibration.")
    parser.add_argument("--render-width", type=int, default=2400, help="KiCad requested render width for board01.")
    parser.add_argument("--render-height", type=int, default=1800, help="KiCad requested render height for board01.")
    parser.add_argument("--output-dir", default="outputs/course_pipeline_demo", help="Static demo output directory.")
    parser.add_argument("--seed", type=int, default=3130, help="Deterministic perturbation seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_demo(args)


if __name__ == "__main__":
    main()
