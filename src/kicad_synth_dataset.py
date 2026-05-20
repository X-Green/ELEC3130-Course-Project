"""Generate a reproducible KiCad-first synthetic PCBA inspection dataset.

The generator creates golden images, camera/background robustness cases,
assembly-defect variants, solder-defect PNG variants, ROI JSON files, ground
truth JSON files, and a dataset manifest. It keeps the original KiCad projects
read-only by copying edited board variants into the dataset output tree.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .background_augment import composite_rgba_on_background, generate_backgrounds
except ImportError:  # pragma: no cover - supports direct script execution
    from background_augment import composite_rgba_on_background, generate_backgrounds


DEFAULT_KICAD_CLI = Path(r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe")
DEFAULT_BOARD_SOURCES = {
    "board01": Path(r"E:\SmartTBBatteryCharger\hardware_test_new_2\SmartTBBatteryChargerNewTest.kicad_pcb"),
    "board02": Path(r"E:\LightSkin\demo-rec-2004.00685-sensorboard\demo-rec-2004.00685-sensorboard.kicad_pcb"),
    "board03": Path(r"E:\Matrice4D-NewBattery-HolderPCB\Matrice4D-NewBattery-HolderPCB.kicad_pcb"),
    "board04": Path(r"E:\X-Tray-Dev-Board\ProPrj_XTray-PCB - v3.0_2025-06-02.kicad_pcb"),
}

IMAGE_WIDTH = 2400
IMAGE_HEIGHT = 1800
DATASET_BACKGROUNDS = ("white", "blue_mat", "noisy_lab")
CAMERA_VARIANTS = (
    {"name": "identity", "zoom": "1"},
    {"name": "shift_pan", "zoom": "0.85", "pan": "1.2,0.8,0"},
    {"name": "rotated_5deg", "zoom": "0.85", "rotate": "0,0,5"},
    {"name": "shift_rotate", "zoom": "0.85", "pan": "0.8,-0.6,0", "rotate": "0,0,-4"},
)
DEFECT_FREE_VARIANT_TYPES = {"registration_background", "enhancement_stress"}
ENHANCEMENT_STRESS_VARIANTS = (
    {
        "name": "low_contrast",
        "course_topic": "spatial_domain_enhancement",
        "description": "Reduced global contrast and raised black level.",
    },
    {
        "name": "illumination_gradient",
        "course_topic": "spatial_domain_enhancement",
        "description": "Smooth left-to-right illumination and shading change.",
    },
    {
        "name": "gaussian_noise",
        "course_topic": "image_restoration",
        "description": "Additive sensor-like Gaussian noise.",
    },
    {
        "name": "motion_blur",
        "course_topic": "image_restoration",
        "description": "Short linear camera-motion blur.",
    },
    {
        "name": "color_cast",
        "course_topic": "color_image_processing",
        "description": "Mild channel-dependent color cast.",
    },
    {
        "name": "jpeg_compression",
        "course_topic": "image_compression",
        "description": "Lossy JPEG quantization and blocking artifacts.",
    },
    {
        "name": "periodic_noise",
        "course_topic": "frequency_domain_enhancement",
        "description": "Low-amplitude sinusoidal stripe interference.",
    },
)


@dataclass(slots=True)
class PadInfo:
    """Parsed SMD pad in footprint-local coordinates."""

    name: str
    pad_type: str
    shape: str
    x: float
    y: float
    angle: float
    width: float
    height: float
    block: str


@dataclass(slots=True)
class ComponentInfo:
    """Parsed top-side footprint and its useful SMD pads."""

    reference: str
    footprint: str
    layer: str
    x: float
    y: float
    angle: float
    pads: list[PadInfo]
    span: tuple[int, int]
    block: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedBoard:
    """Minimal board model needed for dataset generation."""

    path: Path
    text: str
    edge_bounds_mm: tuple[float, float, float, float]
    components: list[ComponentInfo]


@dataclass(slots=True)
class PixelMapper:
    """Map KiCad millimeter coordinates to rendered image pixels."""

    edge_bounds_mm: tuple[float, float, float, float]
    board_bbox_px: tuple[int, int, int, int]
    image_width: int
    image_height: int

    def point(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        min_x, min_y, max_x, max_y = self.edge_bounds_mm
        x_px, y_px, width_px, height_px = self.board_bbox_px
        scale_x = width_px / max(max_x - min_x, 1e-6)
        scale_y = height_px / max(max_y - min_y, 1e-6)
        return (
            x_px + (x_mm - min_x) * scale_x,
            y_px + (y_mm - min_y) * scale_y,
        )

    def bbox_from_points(
        self,
        points_mm: list[tuple[float, float]],
        margin_px: int,
        min_size_px: int = 8,
    ) -> tuple[int, int, int, int]:
        points_px = [self.point(x, y) for x, y in points_mm]
        xs = [point[0] for point in points_px]
        ys = [point[1] for point in points_px]
        x0 = math.floor(min(xs)) - margin_px
        y0 = math.floor(min(ys)) - margin_px
        x1 = math.ceil(max(xs)) + margin_px
        y1 = math.ceil(max(ys)) + margin_px

        if x1 - x0 < min_size_px:
            pad = int(math.ceil((min_size_px - (x1 - x0)) / 2))
            x0 -= pad
            x1 += pad
        if y1 - y0 < min_size_px:
            pad = int(math.ceil((min_size_px - (y1 - y0)) / 2))
            y0 -= pad
            y1 += pad

        return _clip_bbox((x0, y0, x1 - x0, y1 - y0), self.image_width, self.image_height)


def _find_matching_paren(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1

    raise ValueError(f"Unbalanced S-expression beginning at byte offset {start}")


def _iter_blocks(text: str, symbol: str) -> list[tuple[int, int, str]]:
    pattern = re.compile(r"\(" + re.escape(symbol) + r"(?=\s)")
    blocks: list[tuple[int, int, str]] = []
    for match in pattern.finditer(text):
        start = match.start()
        end = _find_matching_paren(text, start)
        blocks.append((start, end, text[start:end]))
    return blocks


def _parse_first_at(block: str) -> tuple[float, float, float]:
    match = re.search(
        r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)(?:\s+([-+]?\d+(?:\.\d+)?))?\s*\)",
        block,
    )
    if not match:
        raise ValueError("Block has no (at x y [angle]) expression")
    angle = 0.0 if match.group(3) is None else float(match.group(3))
    return float(match.group(1)), float(match.group(2)), angle


def _parse_size(block: str) -> tuple[float, float] | None:
    match = re.search(
        r"\(size\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*\)",
        block,
    )
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _parse_layer(block: str) -> str:
    match = re.search(r"\(layer\s+\"([^\"]+)\"\s*\)", block)
    return "" if not match else match.group(1)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(image_width, int(x + width))
    y1 = min(image_height, int(y + height))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _natural_ref_key(reference: str) -> tuple[str, int, str]:
    match = re.match(r"([A-Za-z]+)(\d+)", reference)
    if not match:
        return reference, 0, reference
    return match.group(1).upper(), int(match.group(2)), reference


def _reference_prefix(reference: str) -> str:
    match = re.match(r"([A-Za-z]+)", reference)
    return reference.upper() if not match else match.group(1).upper()


def _component_rank(component: ComponentInfo) -> tuple[int, str, int, str]:
    prefix, number, full = _natural_ref_key(component.reference)
    preferred = {
        "R": 0,
        "C": 1,
        "D": 2,
        "LED": 3,
        "Q": 4,
        "U": 5,
    }.get(prefix, 9)
    return preferred, prefix, number, full


def _has_reasonable_pad_shape(component: ComponentInfo) -> bool:
    for pad in component.pads:
        if pad.width <= 0 or pad.height <= 0:
            return False
        aspect = max(pad.width, pad.height) / max(min(pad.width, pad.height), 1e-6)
        if aspect > 8.0:
            return False
    return True


def _component_mm_area(component: ComponentInfo) -> float:
    points: list[tuple[float, float]] = []
    for pad in component.pads:
        points.extend(_pad_world_corners(component, pad, extra_mm=0.45))
    if not points:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def _component_center_distance(first: ComponentInfo, second: ComponentInfo) -> float:
    return float(np.hypot(first.x - second.x, first.y - second.y))


def _rotate_point(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def _pad_world_corners(
    component: ComponentInfo,
    pad: PadInfo,
    extra_mm: float = 0.0,
) -> list[tuple[float, float]]:
    half_w = pad.width / 2.0 + extra_mm
    half_h = pad.height / 2.0 + extra_mm
    local_corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    points: list[tuple[float, float]] = []
    for corner_x, corner_y in local_corners:
        pad_dx, pad_dy = _rotate_point(corner_x, corner_y, pad.angle)
        local_x = pad.x + pad_dx
        local_y = pad.y + pad_dy
        world_dx, world_dy = _rotate_point(local_x, local_y, component.angle)
        points.append((component.x + world_dx, component.y + world_dy))
    return points


def _pad_world_center(component: ComponentInfo, pad: PadInfo) -> tuple[float, float]:
    world_dx, world_dy = _rotate_point(pad.x, pad.y, component.angle)
    return component.x + world_dx, component.y + world_dy


def _parse_pad_block(block: str) -> PadInfo | None:
    header = re.match(r"\(pad\s+(\"[^\"]*\"|\S+)\s+(\S+)\s+(\S+)", block)
    if not header:
        return None

    pad_name = _strip_quotes(header.group(1))
    pad_type = header.group(2)
    shape = header.group(3)
    if pad_type != "smd":
        return None
    if '"F.Cu"' not in block and "F.Cu" not in block:
        return None

    try:
        x, y, angle = _parse_first_at(block)
    except ValueError:
        x, y, angle = 0.0, 0.0, 0.0

    size = _parse_size(block)
    if size is None:
        return None

    return PadInfo(
        name=pad_name,
        pad_type=pad_type,
        shape=shape,
        x=x,
        y=y,
        angle=angle,
        width=size[0],
        height=size[1],
        block=block,
    )


def _parse_components(text: str) -> list[ComponentInfo]:
    components: list[ComponentInfo] = []

    for start, end, block in _iter_blocks(text, "footprint"):
        footprint_match = re.match(r"\(footprint\s+\"([^\"]+)\"", block)
        if not footprint_match:
            continue

        reference_match = re.search(r"\(property\s+\"Reference\"\s+\"([^\"]+)\"", block)
        if not reference_match:
            reference_match = re.search(r"\(fp_text\s+reference\s+\"([^\"]+)\"", block)
        if not reference_match:
            continue

        layer = _parse_layer(block)
        if layer and layer != "F.Cu":
            continue

        try:
            x, y, angle = _parse_first_at(block)
        except ValueError:
            continue

        pads = [
            pad
            for _, _, pad_block in _iter_blocks(block, "pad")
            if (pad := _parse_pad_block(pad_block)) is not None
        ]
        if not pads:
            continue

        components.append(
            ComponentInfo(
                reference=reference_match.group(1),
                footprint=footprint_match.group(1),
                layer=layer,
                x=x,
                y=y,
                angle=angle,
                pads=pads,
                span=(start, end),
                block=block,
            )
        )

    return components


def _parse_edge_bounds(text: str, components: list[ComponentInfo]) -> tuple[float, float, float, float]:
    points: list[tuple[float, float]] = []

    for symbol in ("gr_line", "gr_rect", "gr_arc", "gr_poly", "gr_curve"):
        for _, _, block in _iter_blocks(text, symbol):
            if '"Edge.Cuts"' not in block and "Edge.Cuts" not in block:
                continue
            for match in re.finditer(
                r"\((?:start|end|mid|center|xy)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)",
                block,
            ):
                points.append((float(match.group(1)), float(match.group(2))))

    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    fallback_points: list[tuple[float, float]] = []
    for component in components:
        fallback_points.append((component.x, component.y))
        for pad in component.pads:
            fallback_points.extend(_pad_world_corners(component, pad, extra_mm=0.5))
    if not fallback_points:
        raise ValueError("Could not infer board extents from Edge.Cuts or footprints")
    xs = [point[0] for point in fallback_points]
    ys = [point[1] for point in fallback_points]
    margin = 3.0
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def parse_board(board_path: Path) -> ParsedBoard:
    text = _read_text(board_path)
    components = _parse_components(text)
    edge_bounds = _parse_edge_bounds(text, components)
    return ParsedBoard(
        path=board_path,
        text=text,
        edge_bounds_mm=edge_bounds,
        components=components,
    )


def _load_rgba(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        image = np.dstack([image, alpha])
    elif image.shape[2] != 4:
        raise ValueError(f"Unsupported image channel count for {path}: {image.shape}")
    return image


def _board_bbox_from_alpha(transparent_image_path: Path) -> tuple[int, int, int, int]:
    rgba = _load_rgba(transparent_image_path)
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0 or len(ys) == 0:
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, rgba.shape[1], rgba.shape[0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to load image for size check: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _white_composite(transparent_path: Path, output_path: Path) -> None:
    rgba = _load_rgba(transparent_path)
    background = np.full(rgba.shape[:2] + (3,), 255, dtype=np.uint8)
    composite = composite_rgba_on_background(rgba, background)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)


def _write_composite(
    transparent_path: Path,
    output_path: Path,
    background_name: str,
    brightness: float = 1.0,
    blur_sigma: float = 0.0,
) -> None:
    rgba = _load_rgba(transparent_path)
    if background_name == "white":
        background = np.full(rgba.shape[:2] + (3,), 255, dtype=np.uint8)
    else:
        backgrounds = generate_backgrounds(rgba.shape[:2])
        if background_name not in backgrounds:
            raise KeyError(f"Unknown generated background: {background_name}")
        background = backgrounds[background_name]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_path),
        composite_rgba_on_background(
            rgba,
            background,
            brightness=brightness,
            blur_sigma=blur_sigma,
        ),
    )


def _as_float_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32)


def _apply_low_contrast(image: np.ndarray) -> np.ndarray:
    image_float = _as_float_image(image)
    adjusted = (image_float - 128.0) * 0.62 + 138.0
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_illumination_gradient(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x_gradient = np.linspace(0.72, 1.16, width, dtype=np.float32)
    y_gradient = np.linspace(0.96, 1.06, height, dtype=np.float32)[:, None]
    gain = x_gradient[None, :] * y_gradient
    adjusted = _as_float_image(image) * gain[:, :, None]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_gaussian_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 9.0, image.shape).astype(np.float32)
    noisy = _as_float_image(image) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_motion_blur(image: np.ndarray) -> np.ndarray:
    kernel_size = 13
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def _apply_color_cast(image: np.ndarray) -> np.ndarray:
    gains = np.array([1.08, 0.96, 0.88], dtype=np.float32)
    shifted = _as_float_image(image) * gains[None, None, :]
    shifted[:, :, 1] += 4.0
    return np.clip(shifted, 0, 255).astype(np.uint8)


def _apply_jpeg_compression(image: np.ndarray) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 42])
    if not success:
        return image.copy()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image.copy() if decoded is None else decoded


def _apply_periodic_noise(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)[:, None]
    stripes = 7.0 * np.sin(2.0 * np.pi * x[None, :] / 34.0)
    cross = 4.0 * np.sin(2.0 * np.pi * (x[None, :] + y) / 91.0)
    pattern = stripes + cross
    noisy = _as_float_image(image) + pattern[:, :, None]
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_enhancement_stress(
    image: np.ndarray,
    stress_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if stress_name == "low_contrast":
        return _apply_low_contrast(image)
    if stress_name == "illumination_gradient":
        return _apply_illumination_gradient(image)
    if stress_name == "gaussian_noise":
        return _apply_gaussian_noise(image, rng)
    if stress_name == "motion_blur":
        return _apply_motion_blur(image)
    if stress_name == "color_cast":
        return _apply_color_cast(image)
    if stress_name == "jpeg_compression":
        return _apply_jpeg_compression(image)
    if stress_name == "periodic_noise":
        return _apply_periodic_noise(image)
    raise ValueError(f"Unknown enhancement stress variant: {stress_name}")


def _write_enhancement_stress_cases(
    board_id: str,
    golden_path: Path,
    output_root: Path,
    ground_truth_dir: Path,
    manifest_cases: list[dict[str, Any]],
    common_case_paths: dict[str, str],
    rng: np.random.Generator,
) -> None:
    golden = cv2.imread(str(golden_path), cv2.IMREAD_COLOR)
    if golden is None:
        raise FileNotFoundError(f"Failed to load golden image for enhancement stress cases: {golden_path}")

    for variant in ENHANCEMENT_STRESS_VARIANTS:
        stress_name = str(variant["name"])
        case_id = f"{board_id}_enhance_{stress_name}"
        test_path = _case_path(output_root, "tests", f"{case_id}_top.png")
        gt_path = ground_truth_dir / f"{case_id}.json"

        stressed = _apply_enhancement_stress(golden, stress_name, rng)
        cv2.imwrite(str(test_path), stressed)
        ground_truth = _empty_ground_truth(case_id)
        ground_truth["metadata"] = {
            "variant_type": "enhancement_stress",
            "stress_name": stress_name,
            "course_topic": variant["course_topic"],
            "description": variant["description"],
        }
        _write_json(gt_path, ground_truth)
        _add_manifest_case(
            manifest_cases,
            case_id,
            board_id,
            test_path,
            gt_path,
            common_case_paths,
            "enhancement_stress",
            metadata={
                "stress_name": stress_name,
                "course_topic": variant["course_topic"],
                "description": variant["description"],
            },
        )


def _run_command(command: list[str], dry_run: bool = False) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(command))
        return
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stdout}"
        )


def _render_board(
    kicad_cli: Path,
    board_path: Path,
    output_path: Path,
    width: int,
    height: int,
    *,
    zoom: str = "1",
    pan: str | None = None,
    rotate: str | None = None,
    background: str = "transparent",
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(kicad_cli),
        "pcb",
        "render",
        str(board_path),
        "--output",
        str(output_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--side",
        "top",
        "--background",
        background,
        "--quality",
        "high",
        "--zoom",
        zoom,
    ]
    if pan:
        command.extend(["--pan", pan])
    if rotate:
        command.extend(["--rotate", rotate])
    _run_command(command, dry_run=dry_run)


def _export_svg(
    kicad_cli: Path,
    board_path: Path,
    output_path: Path,
    force: bool,
    dry_run: bool,
) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            str(kicad_cli),
            "pcb",
            "export",
            "svg",
            str(board_path),
            "--output",
            str(output_path),
            "--layers",
            "F.Cu,F.Mask,F.Silkscreen,Edge.Cuts",
            "--page-size-mode",
            "2",
            "--exclude-drawing-sheet",
            "--mode-single",
        ],
        dry_run=dry_run,
    )


def _export_pos(
    kicad_cli: Path,
    board_path: Path,
    output_path: Path,
    force: bool,
    dry_run: bool,
) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            str(kicad_cli),
            "pcb",
            "export",
            "pos",
            str(board_path),
            "--output",
            str(output_path),
            "--side",
            "front",
            "--format",
            "csv",
            "--units",
            "mm",
        ],
        dry_run=dry_run,
    )


def _select_components(parsed: ParsedBoard, count: int = 2) -> list[ComponentInfo]:
    allowed_prefixes = {"R", "C", "D", "LED", "F", "L"}
    candidates = [
        component
        for component in parsed.components
        if len({pad.name for pad in component.pads if pad.name}) == 2
        and _reference_prefix(component.reference) in allowed_prefixes
        and not component.reference.upper().startswith(("H", "TP", "MH"))
        and _component_mm_area(component) >= 12.0
        and _has_reasonable_pad_shape(component)
    ]
    if len(candidates) < count:
        candidates = [
            component
            for component in parsed.components
            if len({pad.name for pad in component.pads if pad.name}) == 2
            and _reference_prefix(component.reference) in allowed_prefixes
            and not component.reference.upper().startswith(("H", "TP", "MH"))
            and _has_reasonable_pad_shape(component)
        ]
    if len(candidates) < count:
        candidates = [
            component
            for component in parsed.components
            if len({pad.name for pad in component.pads if pad.name}) >= 2
            and _reference_prefix(component.reference) in allowed_prefixes
            and not component.reference.upper().startswith(("H", "TP", "MH"))
            and _has_reasonable_pad_shape(component)
        ]

    candidates.sort(key=lambda component: (-_component_mm_area(component), _component_rank(component)))
    selected: list[ComponentInfo] = []
    for component in candidates:
        if all(_component_center_distance(component, existing) >= 10.0 for existing in selected):
            selected.append(component)
        if len(selected) >= count:
            break

    if len(selected) < count:
        for component in candidates:
            if component not in selected:
                selected.append(component)
            if len(selected) >= count:
                break

    if len(selected) < count:
        raise ValueError(
            f"{parsed.path} has only {len(selected)} suitable top-side SMD components; "
            f"{count} are required"
        )
    return selected


def _component_roi(
    component: ComponentInfo,
    mapper: PixelMapper,
) -> tuple[int, int, int, int]:
    points: list[tuple[float, float]] = []
    for pad in component.pads:
        points.extend(_pad_world_corners(component, pad, extra_mm=0.45))
    return mapper.bbox_from_points(points, margin_px=14, min_size_px=18)


def _pad_roi(
    component: ComponentInfo,
    pad: PadInfo,
    mapper: PixelMapper,
) -> tuple[int, int, int, int]:
    points = _pad_world_corners(component, pad, extra_mm=0.28)
    return mapper.bbox_from_points(points, margin_px=7, min_size_px=12)


def _build_roi_json(
    board_id: str,
    parsed: ParsedBoard,
    selected_components: list[ComponentInfo],
    mapper: PixelMapper,
    output_path: Path,
) -> dict[str, Any]:
    components_json: list[dict[str, Any]] = []
    solder_json: list[dict[str, Any]] = []

    for component in selected_components:
        named_pads = [pad for pad in component.pads if pad.name]
        named_pads.sort(key=lambda pad: (pad.name, pad.x, pad.y))
        component_bbox = _component_roi(component, mapper)
        components_json.append(
            {
                "id": component.reference,
                "bbox": list(component_bbox),
                "type": component.footprint,
                "expected_angle": component.angle,
                "metadata": {
                    "source_board": board_id,
                    "center_mm": [component.x, component.y],
                    "pad_names": [pad.name for pad in named_pads],
                },
            }
        )

        used_ids: dict[str, int] = {}
        emitted_pad_names: set[str] = set()
        for pad in named_pads:
            if pad.name in emitted_pad_names:
                continue
            emitted_pad_names.add(pad.name)
            used_ids[pad.name] = used_ids.get(pad.name, 0) + 1
            suffix = "" if used_ids[pad.name] == 1 else f"_{used_ids[pad.name]}"
            pad_id = f"{component.reference}-P{pad.name}{suffix}"
            center = _pad_world_center(component, pad)
            solder_json.append(
                {
                    "id": pad_id,
                    "component_id": component.reference,
                    "bbox": list(_pad_roi(component, pad, mapper)),
                    "metadata": {
                        "pad_name": pad.name,
                        "center_mm": [center[0], center[1]],
                        "pad_size_mm": [pad.width, pad.height],
                    },
                }
            )

    roi_json = {
        "board_id": board_id,
        "source_pcb": str(parsed.path),
        "image_size": [mapper.image_width, mapper.image_height],
        "edge_bounds_mm": list(parsed.edge_bounds_mm),
        "board_bbox_px": list(mapper.board_bbox_px),
        "components": components_json,
        "solder_joints": solder_json,
    }
    _write_json(output_path, roi_json)
    return roi_json


def _replace_component_at(
    component: ComponentInfo,
    *,
    x: float | None = None,
    y: float | None = None,
    angle: float | None = None,
) -> str:
    match = re.search(
        r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)(?:\s+([-+]?\d+(?:\.\d+)?))?\s*\)",
        component.block,
    )
    if not match:
        raise ValueError(f"Could not find footprint-level (at ...) for {component.reference}")
    active_x = component.x if x is None else x
    active_y = component.y if y is None else y
    active_angle = component.angle if angle is None else angle
    replacement = f"(at {active_x:.6f} {active_y:.6f} {active_angle:.6f})"
    return component.block[: match.start()] + replacement + component.block[match.end() :]


def _write_missing_variant(parsed: ParsedBoard, component: ComponentInfo, output_path: Path) -> None:
    start, end = component.span
    variant_text = parsed.text[:start] + parsed.text[end:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(variant_text, encoding="utf-8")


def _write_repositioned_variant(
    parsed: ParsedBoard,
    component: ComponentInfo,
    output_path: Path,
    *,
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    d_angle: float = 0.0,
) -> None:
    start, end = component.span
    replacement = _replace_component_at(
        component,
        x=component.x + dx_mm,
        y=component.y + dy_mm,
        angle=(component.angle + d_angle) % 360.0,
    )
    variant_text = parsed.text[:start] + replacement + parsed.text[end:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(variant_text, encoding="utf-8")


def _write_visible_rotation_variant(
    parsed: ParsedBoard,
    component: ComponentInfo,
    component_bbox: list[int],
    variant_pcb: Path,
    variant_transparent: Path,
    test_path: Path,
    golden_image: np.ndarray | None,
    kicad_cli: Path,
    width: int,
    height: int,
    dry_run: bool,
) -> dict[str, Any]:
    candidate_angles = (90.0, 45.0, 180.0)
    visible_threshold = 0.035
    best_angle = candidate_angles[0]
    best_changed_fraction = -1.0
    best_mean_difference = -1.0

    if dry_run or golden_image is None:
        _write_repositioned_variant(parsed, component, variant_pcb, d_angle=best_angle)
        return {
            "d_angle": best_angle,
            "rotation_changed_fraction": 0.0,
            "rotation_mean_difference": 0.0,
            "rotation_visible": False,
        }

    for angle in candidate_angles:
        _write_repositioned_variant(parsed, component, variant_pcb, d_angle=angle)
        _render_board(
            kicad_cli,
            variant_pcb,
            variant_transparent,
            width,
            height,
            background="transparent",
            force=True,
            dry_run=False,
        )
        _white_composite(variant_transparent, test_path)
        test_image = cv2.imread(str(test_path), cv2.IMREAD_COLOR)
        if test_image is None:
            continue
        changed_fraction = _changed_fraction_in_bbox(golden_image, test_image, component_bbox)
        mean_difference = _mean_difference_in_bbox(golden_image, test_image, component_bbox)
        if changed_fraction > best_changed_fraction:
            best_angle = angle
            best_changed_fraction = changed_fraction
            best_mean_difference = mean_difference
        if changed_fraction >= visible_threshold:
            return {
                "d_angle": angle,
                "rotation_changed_fraction": changed_fraction,
                "rotation_mean_difference": mean_difference,
                "rotation_visible": True,
            }

    _write_repositioned_variant(parsed, component, variant_pcb, d_angle=best_angle)
    _render_board(
        kicad_cli,
        variant_pcb,
        variant_transparent,
        width,
        height,
        background="transparent",
        force=True,
        dry_run=False,
    )
    _white_composite(variant_transparent, test_path)
    return {
        "d_angle": best_angle,
        "rotation_changed_fraction": max(0.0, best_changed_fraction),
        "rotation_mean_difference": max(0.0, best_mean_difference),
        "rotation_visible": best_changed_fraction >= visible_threshold,
    }


def _case_path(output_root: Path, *parts: str) -> Path:
    path = output_root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _defect_json(
    case_id: str,
    defect_type: str,
    bbox: list[int],
    *,
    component_id: str | None = None,
    solder_joint_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defect: dict[str, Any] = {
        "id": "gt1",
        "defect_type": defect_type,
        "bbox": bbox,
    }
    if component_id is not None:
        defect["component_id"] = component_id
    if solder_joint_id is not None:
        defect["solder_joint_id"] = solder_joint_id
    if metadata:
        defect["metadata"] = metadata
    return {"case_id": case_id, "defects": [defect]}


def _empty_ground_truth(case_id: str) -> dict[str, Any]:
    return {"case_id": case_id, "defects": []}


def _roi_by_component(roi_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in roi_json["components"]}


def _solder_by_component(roi_json: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in roi_json["solder_joints"]:
        grouped.setdefault(item["component_id"], []).append(item)
    for joints in grouped.values():
        joints.sort(key=lambda item: item["id"])
    return grouped


def _bbox_center(bbox: list[int]) -> tuple[int, int]:
    x, y, width, height = bbox
    return x + width // 2, y + height // 2


def _union_bbox(bboxes: list[list[int]], image_shape: tuple[int, int, int], margin: int = 0) -> list[int]:
    x0 = min(bbox[0] for bbox in bboxes) - margin
    y0 = min(bbox[1] for bbox in bboxes) - margin
    x1 = max(bbox[0] + bbox[2] for bbox in bboxes) + margin
    y1 = max(bbox[1] + bbox[3] for bbox in bboxes) + margin
    height, width = image_shape[:2]
    return list(_clip_bbox((x0, y0, x1 - x0, y1 - y0), width, height))


def _median_ring_color(image: np.ndarray, bbox: list[int], ring: int = 8) -> tuple[int, int, int]:
    x, y, width, height = bbox
    x0 = max(0, x - ring)
    y0 = max(0, y - ring)
    x1 = min(image.shape[1], x + width + ring)
    y1 = min(image.shape[0], y + height + ring)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return 80, 120, 80
    mask = np.ones(patch.shape[:2], dtype=bool)
    inner_x0 = max(0, x - x0)
    inner_y0 = max(0, y - y0)
    inner_x1 = min(mask.shape[1], inner_x0 + width)
    inner_y1 = min(mask.shape[0], inner_y0 + height)
    mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
    pixels = patch[mask]
    if len(pixels) == 0:
        pixels = patch.reshape(-1, 3)
    color = np.median(pixels, axis=0)
    return int(color[0]), int(color[1]), int(color[2])


def _fill_bbox_with_local_color(image: np.ndarray, bbox: list[int], strength: float) -> None:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return
    color = np.array(_median_ring_color(image, bbox), dtype=np.float32)
    patch = image[y:y + height, x:x + width].astype(np.float32)
    patch = patch * (1.0 - strength) + color[None, None, :] * strength
    image[y:y + height, x:x + width] = np.clip(patch, 0, 255).astype(np.uint8)


def _fill_solder_absence(image: np.ndarray, bbox: list[int], strength: float = 1.0) -> None:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return
    color = np.array(_median_ring_color(image, bbox, ring=12), dtype=np.float32)
    patch = image[y:y + height, x:x + width].astype(np.float32)
    patch = patch * (1.0 - strength) + color[None, None, :] * strength
    image[y:y + height, x:x + width] = np.clip(patch, 0, 255).astype(np.uint8)


def _changed_fraction_in_bbox(reference: np.ndarray, test: np.ndarray, bbox: list[int], threshold: int = 18) -> float:
    x, y, width, height = _clip_bbox(tuple(bbox), reference.shape[1], reference.shape[0])
    if width <= 0 or height <= 0:
        return 0.0
    reference_roi = reference[y:y + height, x:x + width]
    test_roi = test[y:y + height, x:x + width]
    if reference_roi.shape != test_roi.shape:
        test_roi = cv2.resize(test_roi, (reference_roi.shape[1], reference_roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    difference = cv2.absdiff(reference_roi, test_roi)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY) if difference.ndim == 3 else difference
    return float(np.mean(gray > threshold))


def _mean_difference_in_bbox(reference: np.ndarray, test: np.ndarray, bbox: list[int]) -> float:
    x, y, width, height = _clip_bbox(tuple(bbox), reference.shape[1], reference.shape[0])
    if width <= 0 or height <= 0:
        return 0.0
    reference_roi = reference[y:y + height, x:x + width]
    test_roi = test[y:y + height, x:x + width]
    if reference_roi.shape != test_roi.shape:
        test_roi = cv2.resize(test_roi, (reference_roi.shape[1], reference_roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    difference = cv2.absdiff(reference_roi, test_roi)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY) if difference.ndim == 3 else difference
    return float(np.mean(gray) / 255.0)


def _draw_measurable_solder_absence(
    image: np.ndarray,
    golden: np.ndarray,
    bbox: list[int],
    *,
    target_changed_fraction: float = 0.14,
) -> list[int]:
    x, y, width, height = bbox
    attempts = (
        (x + width // 5, y + height // 5, max(1, width * 3 // 5), max(1, height * 3 // 5), 1.0),
        (x + width // 8, y + height // 8, max(1, width * 3 // 4), max(1, height * 3 // 4), 1.0),
        (x, y, width, height, 0.82),
    )
    applied_bbox = bbox
    for attempt in attempts:
        working = golden.copy()
        clipped = list(_clip_bbox(tuple(int(value) for value in attempt[:4]), image.shape[1], image.shape[0]))
        _fill_solder_absence(working, clipped, strength=float(attempt[4]))
        image[:, :] = working
        applied_bbox = clipped
        if _changed_fraction_in_bbox(golden, image, bbox) >= target_changed_fraction:
            break
    if _changed_fraction_in_bbox(golden, image, bbox) < target_changed_fraction:
        working = golden.copy()
        x0, y0, w0, h0 = _clip_bbox((x, y, width, height), image.shape[1], image.shape[0])
        if w0 > 0 and h0 > 0:
            patch = working[y0:y0 + h0, x0:x0 + w0].astype(np.float32)
            local_color = np.array(_median_ring_color(working, [x0, y0, w0, h0], ring=16), dtype=np.float32)
            darker = np.clip(local_color * 0.72, 0, 255)
            patch = patch * 0.20 + darker[None, None, :] * 0.80
            working[y0:y0 + h0, x0:x0 + w0] = np.clip(patch, 0, 255).astype(np.uint8)
            image[:, :] = working
            applied_bbox = [x0, y0, w0, h0]
    return applied_bbox


def _draw_solder_bridge(image: np.ndarray, first_bbox: list[int], second_bbox: list[int]) -> list[int]:
    first_center = _bbox_center(first_bbox)
    second_center = _bbox_center(second_bbox)
    thickness = max(10, min(first_bbox[2], first_bbox[3], second_bbox[2], second_bbox[3]) // 2)
    cv2.line(image, first_center, second_center, (225, 225, 220), thickness, cv2.LINE_AA)
    cv2.circle(image, first_center, max(3, thickness // 2), (235, 235, 230), -1, cv2.LINE_AA)
    cv2.circle(image, second_center, max(3, thickness // 2), (235, 235, 230), -1, cv2.LINE_AA)
    return _union_bbox([first_bbox, second_bbox], image.shape, margin=thickness)


def _draw_measurable_solder_bridge(
    image: np.ndarray,
    golden: np.ndarray,
    first_bbox: list[int],
    second_bbox: list[int],
) -> list[int]:
    first_center = _bbox_center(first_bbox)
    second_center = _bbox_center(second_bbox)
    base_thickness = max(10, min(first_bbox[2], first_bbox[3], second_bbox[2], second_bbox[3]) // 2)
    bridge_bbox = _union_bbox([first_bbox, second_bbox], image.shape, margin=base_thickness)
    for multiplier, color in ((1.0, (225, 225, 220)), (1.35, (238, 238, 232)), (1.70, (245, 245, 238))):
        working = golden.copy()
        thickness = max(8, int(round(base_thickness * multiplier)))
        cv2.line(working, first_center, second_center, color, thickness, cv2.LINE_AA)
        cv2.circle(working, first_center, max(3, thickness // 2), (250, 250, 245), -1, cv2.LINE_AA)
        cv2.circle(working, second_center, max(3, thickness // 2), (250, 250, 245), -1, cv2.LINE_AA)
        bridge_bbox = _union_bbox([first_bbox, second_bbox], image.shape, margin=thickness)
        image[:, :] = working
        if _changed_fraction_in_bbox(golden, image, bridge_bbox) >= 0.18:
            break
    return bridge_bbox


def _draw_excessive_solder(image: np.ndarray, bbox: list[int]) -> None:
    x, y, width, height = bbox
    center = (x + width // 2, y + height // 2)
    axes = (max(6, int(width * 0.55)), max(6, int(height * 0.55)))
    cv2.ellipse(image, center, axes, 0, 0, 360, (230, 230, 225), -1, cv2.LINE_AA)
    cv2.ellipse(image, center, (max(3, axes[0] // 2), max(3, axes[1] // 2)), 0, 0, 360, (250, 250, 245), -1, cv2.LINE_AA)


def _write_solder_png_cases(
    board_id: str,
    golden_path: Path,
    roi_json: dict[str, Any],
    selected_components: list[ComponentInfo],
    output_root: Path,
    ground_truth_dir: Path,
    manifest_cases: list[dict[str, Any]],
    common_case_paths: dict[str, str],
) -> None:
    golden = cv2.imread(str(golden_path), cv2.IMREAD_COLOR)
    if golden is None:
        raise FileNotFoundError(f"Failed to load golden image for solder variants: {golden_path}")

    component_joints = _solder_by_component(roi_json)

    for component in selected_components:
        joints = component_joints.get(component.reference, [])
        if len(joints) < 2:
            continue

        first_joint = joints[0]
        second_joint = joints[1]
        solder_variants = (
            ("solder_bridge", first_joint, second_joint),
            ("insufficient_solder", first_joint, None),
            ("excessive_solder", first_joint, None),
            ("missing_solder", second_joint, None),
        )

        for defect_type, first, second in solder_variants:
            case_id = f"{board_id}_{defect_type}_{component.reference}"
            test_path = _case_path(output_root, "tests", f"{case_id}_top.png")
            image = golden.copy()

            if defect_type == "solder_bridge" and second is not None:
                gt_bbox = _draw_measurable_solder_bridge(image, golden, first["bbox"], second["bbox"])
                metadata = {"solder_joint_ids": [first["id"], second["id"]]}
                solder_joint_id = first["id"]
            elif defect_type == "insufficient_solder":
                edited_bbox = _draw_measurable_solder_absence(
                    image,
                    golden,
                    first["bbox"],
                    target_changed_fraction=0.14,
                )
                gt_bbox = first["bbox"]
                metadata = {
                    "solder_joint_ids": [first["id"]],
                    "edited_bbox": edited_bbox,
                    "changed_fraction": _changed_fraction_in_bbox(golden, image, first["bbox"]),
                    "mean_difference": _mean_difference_in_bbox(golden, image, first["bbox"]),
                }
                solder_joint_id = first["id"]
            elif defect_type == "excessive_solder":
                _draw_excessive_solder(image, first["bbox"])
                gt_bbox = list(_clip_bbox(
                    (
                        first["bbox"][0] - 6,
                        first["bbox"][1] - 6,
                        first["bbox"][2] + 12,
                        first["bbox"][3] + 12,
                    ),
                    image.shape[1],
                    image.shape[0],
                ))
                metadata = {"solder_joint_ids": [first["id"]]}
                solder_joint_id = first["id"]
            else:
                _fill_solder_absence(image, first["bbox"], strength=1.0)
                gt_bbox = first["bbox"]
                metadata = {"solder_joint_ids": [first["id"]]}
                solder_joint_id = first["id"]

            cv2.imwrite(str(test_path), image)
            gt_path = ground_truth_dir / f"{case_id}.json"
            _write_json(
                gt_path,
                _defect_json(
                    case_id,
                    defect_type,
                    gt_bbox,
                    component_id=component.reference,
                    solder_joint_id=solder_joint_id,
                    metadata=metadata,
                ),
            )
            manifest_cases.append(
                {
                    "case_id": case_id,
                    "board_id": board_id,
                    **common_case_paths,
                    "test": _relative(test_path, Path.cwd()),
                    "ground_truth": _relative(gt_path, Path.cwd()),
                    "variant_type": defect_type,
                }
            )


def _add_manifest_case(
    manifest_cases: list[dict[str, Any]],
    case_id: str,
    board_id: str,
    test_path: Path,
    gt_path: Path,
    common_case_paths: dict[str, str],
    variant_type: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    case = {
        "case_id": case_id,
        "board_id": board_id,
        **common_case_paths,
        "test": _relative(test_path, Path.cwd()),
        "ground_truth": _relative(gt_path, Path.cwd()),
        "variant_type": variant_type,
    }
    if metadata:
        case["metadata"] = metadata
    manifest_cases.append(case)


def _generate_board_dataset(
    board_id: str,
    source_pcb: Path,
    output_root: Path,
    kicad_cli: Path,
    width: int,
    height: int,
    seed: int,
    force: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not source_pcb.exists():
        raise FileNotFoundError(f"Source PCB does not exist: {source_pcb}")

    parsed = parse_board(source_pcb)
    selected = _select_components(parsed, count=2)

    transparent_dir = output_root / "transparent"
    golden_dir = output_root / "golden"
    exports_dir = output_root / "exports"
    roi_dir = output_root / "roi"
    ground_truth_dir = output_root / "ground_truth"
    variant_dir = exports_dir / "variant_boards"

    top_transparent = transparent_dir / f"{board_id}_top_transparent.png"
    golden_path = golden_dir / f"{board_id}_top.png"

    _render_board(
        kicad_cli,
        source_pcb,
        top_transparent,
        width,
        height,
        background="transparent",
        force=force,
        dry_run=dry_run,
    )
    if not dry_run:
        _white_composite(top_transparent, golden_path)

    _export_svg(kicad_cli, source_pcb, exports_dir / f"{board_id}_top.svg", force, dry_run)
    _export_pos(kicad_cli, source_pcb, exports_dir / f"{board_id}_pos.csv", force, dry_run)

    actual_width, actual_height = (width, height) if dry_run else _image_size(top_transparent)
    board_bbox_px = (0, 0, actual_width, actual_height) if dry_run else _board_bbox_from_alpha(top_transparent)
    mapper = PixelMapper(
        edge_bounds_mm=parsed.edge_bounds_mm,
        board_bbox_px=board_bbox_px,
        image_width=actual_width,
        image_height=actual_height,
    )
    roi_path = roi_dir / f"{board_id}.json"
    roi_json = _build_roi_json(board_id, parsed, selected, mapper, roi_path)
    components_by_id = _roi_by_component(roi_json)

    common_case_paths = {
        "golden": _relative(golden_path, Path.cwd()),
        "roi": _relative(roi_path, Path.cwd()),
    }
    manifest_cases: list[dict[str, Any]] = []

    rng = np.random.default_rng(seed)

    for camera in CAMERA_VARIANTS:
        camera_name = camera["name"]
        camera_transparent = (
            top_transparent
            if camera_name == "identity"
            else transparent_dir / f"{board_id}_render_{camera_name}_top_transparent.png"
        )
        if camera_name != "identity":
            _render_board(
                kicad_cli,
                source_pcb,
                camera_transparent,
                width,
                height,
                zoom=camera["zoom"],
                pan=camera.get("pan"),
                rotate=camera.get("rotate"),
                background="transparent",
                force=force,
                dry_run=dry_run,
            )

        for background_index, background in enumerate(DATASET_BACKGROUNDS):
            case_id = f"{board_id}_registration_{camera_name}_{background}"
            test_path = output_root / "tests" / f"{case_id}_top.png"
            gt_path = ground_truth_dir / f"{case_id}.json"
            if not dry_run:
                brightness = 1.0 if background == "white" else 0.94 + 0.03 * background_index
                _write_composite(
                    camera_transparent,
                    test_path,
                    background,
                    brightness=1.0,
                    blur_sigma=0.0,
                )
            _write_json(gt_path, _empty_ground_truth(case_id))
            _add_manifest_case(
                manifest_cases,
                case_id,
                board_id,
                test_path,
                gt_path,
                common_case_paths,
                "registration_background",
            )

    if not dry_run:
        _write_enhancement_stress_cases(
            board_id,
            golden_path,
            output_root,
            ground_truth_dir,
            manifest_cases,
            common_case_paths,
            rng,
        )

    golden_image = None if dry_run else cv2.imread(str(golden_path), cv2.IMREAD_COLOR)

    for component_index, component in enumerate(selected):
        component_bbox = components_by_id[component.reference]["bbox"]
        assembly_variants = (
            ("missing_component", "missing", {}),
            ("shifted_component", "shifted", {"dx_mm": 0.65 + 0.10 * component_index, "dy_mm": -0.35}),
            ("rotated_component", "rotated", {}),
        )
        for defect_type, suffix, parameters in assembly_variants:
            case_id = f"{board_id}_{suffix}_{component.reference}"
            variant_pcb = variant_dir / f"{case_id}.kicad_pcb"
            variant_transparent = transparent_dir / f"{case_id}_transparent.png"
            test_path = output_root / "tests" / f"{case_id}_top.png"

            if defect_type == "missing_component":
                _write_missing_variant(parsed, component, variant_pcb)
                _render_board(
                    kicad_cli,
                    variant_pcb,
                    variant_transparent,
                    width,
                    height,
                    background="transparent",
                    force=force,
                    dry_run=dry_run,
                )
                if not dry_run:
                    _white_composite(variant_transparent, test_path)
            elif defect_type == "rotated_component":
                parameters = _write_visible_rotation_variant(
                    parsed,
                    component,
                    component_bbox,
                    variant_pcb,
                    variant_transparent,
                    test_path,
                    golden_image,
                    kicad_cli,
                    width,
                    height,
                    dry_run,
                )
            else:
                _write_repositioned_variant(parsed, component, variant_pcb, **parameters)
                _render_board(
                    kicad_cli,
                    variant_pcb,
                    variant_transparent,
                    width,
                    height,
                    background="transparent",
                    force=force,
                    dry_run=dry_run,
                )
                if not dry_run:
                    _white_composite(variant_transparent, test_path)

            gt_path = ground_truth_dir / f"{case_id}.json"
            _write_json(
                gt_path,
                _defect_json(
                    case_id,
                    defect_type,
                    component_bbox,
                    component_id=component.reference,
                    metadata=parameters,
                ),
            )
            _add_manifest_case(
                manifest_cases,
                case_id,
                board_id,
                test_path,
                gt_path,
                common_case_paths,
                defect_type,
            )

    if not dry_run:
        _write_solder_png_cases(
            board_id,
            golden_path,
            roi_json,
            selected,
            output_root,
            ground_truth_dir,
            manifest_cases,
            common_case_paths,
        )

    board_entry = {
        "board_id": board_id,
        "source_pcb": str(source_pcb),
        "golden": _relative(golden_path, Path.cwd()),
        "transparent": _relative(top_transparent, Path.cwd()),
        "roi": _relative(roi_path, Path.cwd()),
        "edge_bounds_mm": list(parsed.edge_bounds_mm),
        "board_bbox_px": list(board_bbox_px),
        "selected_components": [component.reference for component in selected],
        "component_count": len(parsed.components),
    }
    return board_entry, manifest_cases


def generate_dataset(args: argparse.Namespace) -> Path:
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    kicad_cli = Path(args.kicad_cli)
    if not args.dry_run and not kicad_cli.exists():
        raise FileNotFoundError(f"KiCad CLI does not exist: {kicad_cli}")

    board_ids = args.boards or list(DEFAULT_BOARD_SOURCES)
    unknown = [board_id for board_id in board_ids if board_id not in DEFAULT_BOARD_SOURCES]
    if unknown:
        raise ValueError(f"Unknown board id(s): {', '.join(unknown)}")

    manifest_boards: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    for board_index, board_id in enumerate(board_ids):
        board_entry, board_cases = _generate_board_dataset(
            board_id=board_id,
            source_pcb=DEFAULT_BOARD_SOURCES[board_id],
            output_root=output_root,
            kicad_cli=kicad_cli,
            width=args.width,
            height=args.height,
            seed=args.seed + board_index,
            force=args.force,
            dry_run=args.dry_run,
        )
        manifest_boards.append(board_entry)
        manifest_cases.extend(board_cases)

    manifest = {
        "dataset_version": 1,
        "created_by": "src.kicad_synth_dataset",
        "seed": args.seed,
        "requested_image_size": [args.width, args.height],
        "kicad_cli": str(kicad_cli),
        "backgrounds": list(DATASET_BACKGROUNDS),
        "camera_variants": CAMERA_VARIANTS,
        "enhancement_stress_variants": list(ENHANCEMENT_STRESS_VARIANTS),
        "boards": manifest_boards,
        "cases": manifest_cases,
        "classical_methods": [
            "KiCad synthetic rendering",
            "background randomization",
            "camera pan/rotation/zoom perturbation",
            "spatial-domain low-contrast and illumination-gradient stress testing",
            "image-restoration Gaussian-noise and motion-blur stress testing",
            "color-cast robustness stress testing",
            "JPEG compression artifact stress testing",
            "frequency-domain periodic-noise stress testing",
            "component-level S-expression variants",
            "solder defect PNG synthesis",
            "ROI and ground-truth annotation",
        ],
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Wrote {manifest_path} with {len(manifest_cases)} cases across {len(manifest_boards)} boards.")
    return manifest_path


def validate_dataset(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    missing: list[str] = []
    empty_defect_cases: list[str] = []
    root = Path.cwd()

    for board in manifest.get("boards", []):
        for key in ("golden", "transparent", "roi"):
            path = root / board[key]
            if not path.exists():
                missing.append(str(path))

    for case in manifest.get("cases", []):
        for key in ("golden", "test", "roi", "ground_truth"):
            path = root / case[key]
            if not path.exists():
                missing.append(str(path))
        gt_path = root / case["ground_truth"]
        if gt_path.exists():
            with gt_path.open("r", encoding="utf-8") as file:
                truth = json.load(file)
            if case["variant_type"] not in DEFECT_FREE_VARIANT_TYPES and not truth.get("defects"):
                empty_defect_cases.append(case["case_id"])

    if missing:
        raise FileNotFoundError("Missing dataset artifacts:\n" + "\n".join(missing[:40]))
    if empty_defect_cases:
        raise ValueError("Defect cases without ground-truth defects: " + ", ".join(empty_defect_cases))

    print(
        "Dataset validation passed: "
        f"{len(manifest.get('boards', []))} boards, {len(manifest.get('cases', []))} cases."
    )


def clean_generated_images(args: argparse.Namespace) -> None:
    output_root = Path(args.output)
    generated_dirs = [
        output_root / "golden",
        output_root / "transparent",
        output_root / "tests",
        output_root / "exports" / "variant_boards",
    ]
    for directory in generated_dirs:
        if directory.exists():
            shutil.rmtree(directory)
            print(f"Removed {directory}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and validate the KiCad synthetic PCBA dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate renders, variants, ROI JSON, ground truth, and manifest.")
    generate.add_argument("--output", default="data/kicad_synth", help="Dataset output root.")
    generate.add_argument("--seed", type=int, default=3130, help="Deterministic random seed.")
    generate.add_argument("--kicad-cli", default=str(DEFAULT_KICAD_CLI), help="Path to kicad-cli.exe.")
    generate.add_argument("--boards", nargs="*", choices=sorted(DEFAULT_BOARD_SOURCES), help="Board IDs to generate.")
    generate.add_argument("--width", type=int, default=IMAGE_WIDTH, help="Render width in pixels.")
    generate.add_argument("--height", type=int, default=IMAGE_HEIGHT, help="Render height in pixels.")
    generate.add_argument("--force", action="store_true", help="Re-render existing generated images.")
    generate.add_argument("--dry-run", action="store_true", help="Print KiCad commands and write JSON only.")
    generate.set_defaults(func=generate_dataset)

    validate = subparsers.add_parser("validate", help="Validate manifest paths and ground-truth coverage.")
    validate.add_argument("--manifest", default="data/kicad_synth/manifest.json", help="Dataset manifest path.")
    validate.set_defaults(func=validate_dataset)

    clean = subparsers.add_parser("clean-images", help="Remove large generated image/variant-board artifacts.")
    clean.add_argument("--output", default="data/kicad_synth", help="Dataset output root.")
    clean.set_defaults(func=clean_generated_images)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
