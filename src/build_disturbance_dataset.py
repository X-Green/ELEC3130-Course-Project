"""Build disturbance-only KiCad synthetic datasets for pipeline robustness tests.

The generated cases are defect-free by construction. They stress registration,
enhancement, segmentation, and false-positive behavior of the existing
reference pipeline without editing the inspection rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .background_augment import composite_rgba_on_background, generate_backgrounds
    from .kicad_synth_dataset import (
        DEFAULT_BOARD_SOURCES,
        DEFAULT_KICAD_CLI,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        PixelMapper,
        _board_bbox_from_alpha,
        _build_roi_json,
        _empty_ground_truth,
        _export_pos,
        _export_svg,
        _image_size,
        _relative,
        _render_board,
        _select_components,
        _white_composite,
        parse_board,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from background_augment import composite_rgba_on_background, generate_backgrounds
    from kicad_synth_dataset import (
        DEFAULT_BOARD_SOURCES,
        DEFAULT_KICAD_CLI,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        PixelMapper,
        _board_bbox_from_alpha,
        _build_roi_json,
        _empty_ground_truth,
        _export_pos,
        _export_svg,
        _image_size,
        _relative,
        _render_board,
        _select_components,
        _white_composite,
        parse_board,
    )


DISTURBANCE_BATCHES = {
    "SlightlyDisturbed": {
        "variant_type": "slightly_disturbed",
        "description": "Realistic handheld photo perturbations with mild pose, lighting, color, noise, and compression changes.",
        "cases": [
            {
                "name": "warm_desk_tilt",
                "angle": -3.0,
                "scale": 0.94,
                "tx": 38,
                "ty": -24,
                "background": "warm_desk",
                "brightness": 1.03,
                "contrast": 1.05,
                "color_gains": (1.06, 0.99, 0.93),
                "gradient": 0.10,
                "noise_sigma": 3.5,
                "blur": 0.25,
                "jpeg_quality": 82,
            },
            {
                "name": "blue_mat_cool",
                "angle": 2.5,
                "scale": 0.96,
                "tx": -42,
                "ty": 32,
                "background": "blue_mat",
                "brightness": 0.96,
                "contrast": 1.02,
                "color_gains": (1.12, 1.01, 0.88),
                "gradient": -0.08,
                "noise_sigma": 4.0,
                "blur": 0.35,
                "jpeg_quality": 78,
            },
            {
                "name": "gray_gradient_shadow",
                "angle": -4.0,
                "scale": 0.93,
                "tx": 62,
                "ty": 38,
                "background": "gray_gradient",
                "brightness": 0.91,
                "contrast": 1.08,
                "color_gains": (0.98, 1.00, 1.05),
                "gradient": 0.18,
                "noise_sigma": 4.5,
                "motion": 7,
                "jpeg_quality": 76,
            },
            {
                "name": "noisy_lab_low_light",
                "angle": 3.5,
                "scale": 0.95,
                "tx": -58,
                "ty": -30,
                "background": "noisy_lab",
                "brightness": 0.86,
                "contrast": 1.12,
                "color_gains": (1.04, 0.96, 0.91),
                "gradient": 0.16,
                "noise_sigma": 6.0,
                "blur": 0.45,
                "jpeg_quality": 72,
            },
            {
                "name": "white_overexposed",
                "angle": 1.5,
                "scale": 0.97,
                "tx": 25,
                "ty": -48,
                "background": "white",
                "brightness": 1.16,
                "contrast": 0.94,
                "color_gains": (1.00, 1.03, 1.05),
                "gradient": -0.11,
                "noise_sigma": 3.0,
                "blur": 0.2,
                "jpeg_quality": 80,
            },
            {
                "name": "black_table_cast",
                "angle": -2.0,
                "scale": 0.95,
                "tx": -24,
                "ty": 52,
                "background": "black",
                "brightness": 0.98,
                "contrast": 1.06,
                "color_gains": (1.18, 1.00, 0.82),
                "gradient": 0.09,
                "noise_sigma": 4.0,
                "jpeg_quality": 74,
            },
            {
                "name": "desk_motion_blur",
                "angle": 4.5,
                "scale": 0.92,
                "tx": 76,
                "ty": -18,
                "background": "warm_desk",
                "brightness": 0.94,
                "contrast": 1.09,
                "color_gains": (1.05, 0.98, 0.92),
                "gradient": 0.13,
                "noise_sigma": 5.0,
                "motion": 9,
                "jpeg_quality": 70,
            },
            {
                "name": "matte_soft_noise",
                "angle": -5.0,
                "scale": 0.91,
                "tx": -70,
                "ty": -44,
                "background": "blue_mat",
                "brightness": 0.90,
                "contrast": 1.14,
                "color_gains": (1.10, 0.97, 0.87),
                "gradient": -0.14,
                "noise_sigma": 6.0,
                "blur": 0.55,
                "jpeg_quality": 68,
            },
        ],
    },
    "BadlyDisturbed": {
        "variant_type": "badly_disturbed",
        "description": "Strong synthetic photo disturbances for stress-testing registration and enhancement margins.",
        "cases": [
            {
                "name": "hard_rotate_yellow_cast",
                "angle": -12.0,
                "scale": 0.78,
                "tx": 145,
                "ty": -105,
                "background": "warm_desk",
                "brightness": 0.88,
                "contrast": 1.34,
                "color_gains": (1.30, 1.04, 0.64),
                "gradient": 0.34,
                "noise_sigma": 14.0,
                "motion": 15,
                "jpeg_quality": 45,
                "periodic": 7.0,
                "vignette": 0.34,
            },
            {
                "name": "blue_mat_large_shift",
                "angle": 10.0,
                "scale": 0.80,
                "tx": -165,
                "ty": 128,
                "background": "blue_mat",
                "brightness": 0.82,
                "contrast": 1.42,
                "color_gains": (1.42, 0.98, 0.58),
                "gradient": -0.30,
                "noise_sigma": 16.0,
                "blur": 0.85,
                "jpeg_quality": 42,
                "periodic": 8.5,
                "vignette": 0.28,
            },
            {
                "name": "dark_table_low_light",
                "angle": -14.0,
                "scale": 0.76,
                "tx": 188,
                "ty": 115,
                "background": "black",
                "brightness": 0.68,
                "contrast": 1.55,
                "color_gains": (1.22, 0.94, 0.76),
                "gradient": 0.40,
                "noise_sigma": 20.0,
                "motion": 17,
                "jpeg_quality": 38,
                "periodic": 6.5,
                "vignette": 0.46,
            },
            {
                "name": "gray_shadow_noise",
                "angle": 13.0,
                "scale": 0.79,
                "tx": -130,
                "ty": -145,
                "background": "gray_gradient",
                "brightness": 0.78,
                "contrast": 1.48,
                "color_gains": (0.86, 1.03, 1.24),
                "gradient": -0.42,
                "noise_sigma": 18.0,
                "blur": 1.05,
                "jpeg_quality": 40,
                "periodic": 9.0,
                "vignette": 0.38,
            },
            {
                "name": "white_glare_compressed",
                "angle": 9.0,
                "scale": 0.82,
                "tx": 95,
                "ty": -152,
                "background": "white",
                "brightness": 1.28,
                "contrast": 1.18,
                "color_gains": (1.08, 1.08, 0.92),
                "gradient": 0.36,
                "noise_sigma": 12.0,
                "motion": 13,
                "jpeg_quality": 34,
                "periodic": 6.0,
                "vignette": 0.20,
            },
            {
                "name": "lab_green_cast",
                "angle": -9.0,
                "scale": 0.81,
                "tx": -112,
                "ty": 162,
                "background": "noisy_lab",
                "brightness": 0.80,
                "contrast": 1.36,
                "color_gains": (0.90, 1.20, 0.72),
                "gradient": -0.35,
                "noise_sigma": 17.0,
                "blur": 0.95,
                "jpeg_quality": 44,
                "periodic": 7.5,
                "vignette": 0.32,
            },
            {
                "name": "desk_motion_stripes",
                "angle": 15.0,
                "scale": 0.75,
                "tx": 176,
                "ty": 78,
                "background": "warm_desk",
                "brightness": 0.75,
                "contrast": 1.60,
                "color_gains": (1.35, 1.02, 0.66),
                "gradient": 0.46,
                "noise_sigma": 18.0,
                "motion": 21,
                "jpeg_quality": 36,
                "periodic": 11.0,
                "vignette": 0.42,
            },
            {
                "name": "blue_cold_blur",
                "angle": -16.0,
                "scale": 0.74,
                "tx": -182,
                "ty": -92,
                "background": "blue_mat",
                "brightness": 0.72,
                "contrast": 1.50,
                "color_gains": (1.48, 0.96, 0.58),
                "gradient": -0.44,
                "noise_sigma": 19.0,
                "motion": 19,
                "jpeg_quality": 35,
                "periodic": 10.0,
                "vignette": 0.44,
            },
        ],
    },
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


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
        raise ValueError(f"Unsupported channel count for {path}: {image.shape}")
    return image


def _board_entry_paths(output_root: Path, board_id: str) -> dict[str, Path]:
    return {
        "transparent": output_root / "transparent" / f"{board_id}_top_transparent.png",
        "golden": output_root / "golden" / f"{board_id}_top.png",
        "roi": output_root / "roi" / f"{board_id}.json",
        "svg": output_root / "exports" / f"{board_id}_top.svg",
        "pos": output_root / "exports" / f"{board_id}_pos.csv",
    }


def _ensure_reference_artifacts(
    *,
    board_id: str,
    source_pcb: Path,
    output_root: Path,
    kicad_cli: Path,
    width: int,
    height: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if not source_pcb.exists():
        raise FileNotFoundError(f"Source PCB does not exist: {source_pcb}")

    paths = _board_entry_paths(output_root, board_id)
    parsed = parse_board(source_pcb)
    selected = _select_components(parsed, count=2)

    _render_board(
        kicad_cli,
        source_pcb,
        paths["transparent"],
        width,
        height,
        background="transparent",
        force=force,
        dry_run=dry_run,
    )
    if not dry_run and (force or not paths["golden"].exists()):
        _white_composite(paths["transparent"], paths["golden"])

    _export_svg(kicad_cli, source_pcb, paths["svg"], force, dry_run)
    _export_pos(kicad_cli, source_pcb, paths["pos"], force, dry_run)

    if dry_run:
        actual_width, actual_height = width, height
        board_bbox_px = (0, 0, width, height)
    else:
        actual_width, actual_height = _image_size(paths["transparent"])
        board_bbox_px = _board_bbox_from_alpha(paths["transparent"])

    mapper = PixelMapper(
        edge_bounds_mm=parsed.edge_bounds_mm,
        board_bbox_px=board_bbox_px,
        image_width=actual_width,
        image_height=actual_height,
    )
    if force or not paths["roi"].exists():
        _build_roi_json(board_id, parsed, selected, mapper, paths["roi"])

    return {
        "board_id": board_id,
        "source_pcb": str(source_pcb),
        "golden": _relative(paths["golden"], Path.cwd()),
        "transparent": _relative(paths["transparent"], Path.cwd()),
        "roi": _relative(paths["roi"], Path.cwd()),
        "edge_bounds_mm": list(parsed.edge_bounds_mm),
        "board_bbox_px": list(board_bbox_px),
        "selected_components": [component.reference for component in selected],
        "component_count": len(parsed.components),
    }


def _transform_rgba(
    rgba: np.ndarray,
    *,
    angle: float,
    scale: float,
    tx: float,
    ty: float,
) -> np.ndarray:
    height, width = rgba.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(
        rgba,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _apply_gradient(image: np.ndarray, strength: float) -> np.ndarray:
    if abs(strength) < 1e-6:
        return image
    height, width = image.shape[:2]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(-0.65, 0.65, height, dtype=np.float32)[:, None]
    gain = 1.0 + strength * (0.75 * x + 0.25 * y)
    adjusted = image.astype(np.float32) * gain[:, :, None]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_color_and_tone(
    image: np.ndarray,
    *,
    brightness: float,
    contrast: float,
    color_gains: tuple[float, float, float],
) -> np.ndarray:
    gains = np.array(color_gains, dtype=np.float32)[None, None, :]
    adjusted = image.astype(np.float32) * gains
    adjusted = (adjusted - 128.0) * contrast + 128.0
    adjusted *= brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return image
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_motion_blur(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return image
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(image, -1, kernel)


def _apply_periodic_noise(image: np.ndarray, amplitude: float) -> np.ndarray:
    if amplitude <= 0:
        return image
    height, width = image.shape[:2]
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    pattern = amplitude * np.sin(2.0 * np.pi * x / 31.0)
    pattern = pattern + amplitude * 0.55 * np.sin(2.0 * np.pi * (x + y) / 83.0)
    adjusted = image.astype(np.float32) + pattern[:, :, None]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_vignette(image: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    height, width = image.shape[:2]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    radius = np.sqrt(x * x + y * y)
    gain = np.clip(1.0 - strength * radius, 0.45, 1.0)
    adjusted = image.astype(np.float32) * gain[:, :, None]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return image
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def _make_disturbed_image(
    transparent_path: Path,
    parameters: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    rgba = _load_rgba(transparent_path)
    warped = _transform_rgba(
        rgba,
        angle=float(parameters["angle"]),
        scale=float(parameters["scale"]),
        tx=float(parameters["tx"]),
        ty=float(parameters["ty"]),
    )
    backgrounds = generate_backgrounds(warped.shape[:2])
    background_name = str(parameters["background"])
    if background_name not in backgrounds:
        raise KeyError(f"Unknown background: {background_name}")

    composite = composite_rgba_on_background(
        warped,
        backgrounds[background_name],
        brightness=1.0,
        blur_sigma=float(parameters.get("composite_blur", 0.0)),
    )
    composite = _apply_gradient(composite, float(parameters.get("gradient", 0.0)))
    composite = _apply_color_and_tone(
        composite,
        brightness=float(parameters.get("brightness", 1.0)),
        contrast=float(parameters.get("contrast", 1.0)),
        color_gains=tuple(parameters.get("color_gains", (1.0, 1.0, 1.0))),
    )
    composite = _apply_periodic_noise(composite, float(parameters.get("periodic", 0.0)))
    composite = _apply_noise(composite, float(parameters.get("noise_sigma", 0.0)), rng)
    if parameters.get("motion"):
        composite = _apply_motion_blur(composite, int(parameters["motion"]))
    elif parameters.get("blur", 0.0) > 0:
        composite = cv2.GaussianBlur(composite, (0, 0), float(parameters["blur"]))
    composite = _apply_vignette(composite, float(parameters.get("vignette", 0.0)))
    return _apply_jpeg_roundtrip(composite, int(parameters.get("jpeg_quality", 100)))


def _manifest_case(
    *,
    case_id: str,
    board_id: str,
    board_entry: dict[str, Any],
    test_path: Path,
    gt_path: Path,
    batch_name: str,
    batch_config: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "board_id": board_id,
        "golden": board_entry["golden"],
        "roi": board_entry["roi"],
        "test": _relative(test_path, Path.cwd()),
        "ground_truth": _relative(gt_path, Path.cwd()),
        "variant_type": batch_config["variant_type"],
        "metadata": {
            "batch": batch_name,
            "stress_name": parameters["name"],
            "description": batch_config["description"],
            "camera": {
                "rotation_deg": parameters["angle"],
                "scale": parameters["scale"],
                "translation_px": [parameters["tx"], parameters["ty"]],
            },
            "background": parameters["background"],
            "color_gains_bgr": list(parameters.get("color_gains", (1.0, 1.0, 1.0))),
            "brightness": parameters.get("brightness", 1.0),
            "contrast": parameters.get("contrast", 1.0),
            "noise_sigma": parameters.get("noise_sigma", 0.0),
            "jpeg_quality": parameters.get("jpeg_quality", 100),
            "periodic_amplitude": parameters.get("periodic", 0.0),
        },
    }


def build_disturbance_dataset(args: argparse.Namespace) -> Path:
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    kicad_cli = Path(args.kicad_cli)
    if not args.dry_run and not kicad_cli.exists():
        raise FileNotFoundError(f"KiCad CLI does not exist: {kicad_cli}")

    board_ids = args.boards or list(DEFAULT_BOARD_SOURCES)
    unknown = [board_id for board_id in board_ids if board_id not in DEFAULT_BOARD_SOURCES]
    if unknown:
        raise ValueError(f"Unknown board id(s): {', '.join(unknown)}")

    batch_names = args.batches or list(DISTURBANCE_BATCHES)
    unknown_batches = [name for name in batch_names if name not in DISTURBANCE_BATCHES]
    if unknown_batches:
        raise ValueError(f"Unknown batch name(s): {', '.join(unknown_batches)}")

    manifest_boards: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)

    for board_id in board_ids:
        board_entry = _ensure_reference_artifacts(
            board_id=board_id,
            source_pcb=DEFAULT_BOARD_SOURCES[board_id],
            output_root=output_root,
            kicad_cli=kicad_cli,
            width=args.width,
            height=args.height,
            force=args.force_reference,
            dry_run=args.dry_run,
        )
        manifest_boards.append(board_entry)

        transparent_path = output_root / "transparent" / f"{board_id}_top_transparent.png"
        for batch_name in batch_names:
            batch_config = DISTURBANCE_BATCHES[batch_name]
            for case_index, parameters in enumerate(batch_config["cases"], start=1):
                case_id = f"{board_id}_{batch_name}_{case_index:02d}_{parameters['name']}"
                test_path = output_root / "tests" / batch_name / f"{case_id}_top.png"
                gt_path = output_root / "ground_truth" / batch_name / f"{case_id}.json"
                if not args.dry_run:
                    test_path.parent.mkdir(parents=True, exist_ok=True)
                    disturbed = _make_disturbed_image(transparent_path, parameters, rng)
                    cv2.imwrite(str(test_path), disturbed)
                ground_truth = _empty_ground_truth(case_id)
                ground_truth["metadata"] = {
                    "variant_type": batch_config["variant_type"],
                    "batch": batch_name,
                    "stress_name": parameters["name"],
                    "description": batch_config["description"],
                }
                _write_json(gt_path, ground_truth)
                manifest_cases.append(
                    _manifest_case(
                        case_id=case_id,
                        board_id=board_id,
                        board_entry=board_entry,
                        test_path=test_path,
                        gt_path=gt_path,
                        batch_name=batch_name,
                        batch_config=batch_config,
                        parameters=parameters,
                    )
                )

    manifest = {
        "dataset_version": 1,
        "created_by": "src.build_disturbance_dataset",
        "seed": args.seed,
        "requested_image_size": [args.width, args.height],
        "kicad_cli": str(kicad_cli),
        "boards": manifest_boards,
        "batches": {
            name: {
                "variant_type": DISTURBANCE_BATCHES[name]["variant_type"],
                "case_count_per_board": len(DISTURBANCE_BATCHES[name]["cases"]),
                "description": DISTURBANCE_BATCHES[name]["description"],
            }
            for name in batch_names
        },
        "cases": manifest_cases,
        "classical_methods": [
            "KiCad synthetic rendering",
            "camera rotation, scale, and translation perturbation",
            "desk, mat, gradient, dark-table, and noisy-lab backgrounds",
            "visible color casts and brightness/contrast shifts",
            "Gaussian sensor noise, motion blur, periodic interference, vignette, and JPEG artifacts",
            "defect-free robustness testing of the existing reference-based pipeline",
        ],
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    print(
        f"Wrote {manifest_path} with {len(manifest_cases)} cases "
        f"across {len(manifest_boards)} boards."
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build disturbance-only KiCad robustness datasets.")
    parser.add_argument("--output", default="data/kicad_disturbance", help="Output dataset root.")
    parser.add_argument("--seed", type=int, default=3130, help="Deterministic random seed.")
    parser.add_argument("--kicad-cli", default=str(DEFAULT_KICAD_CLI), help="Path to kicad-cli.exe.")
    parser.add_argument("--boards", nargs="*", choices=sorted(DEFAULT_BOARD_SOURCES), help="Board IDs to generate.")
    parser.add_argument(
        "--batches",
        nargs="*",
        choices=sorted(DISTURBANCE_BATCHES),
        help="Disturbance batches to generate.",
    )
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT, help="Render height in pixels.")
    parser.add_argument("--force-reference", action="store_true", help="Re-render golden/ROI reference artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print KiCad commands and write JSON only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_disturbance_dataset(args)


if __name__ == "__main__":
    main()
