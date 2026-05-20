"""Create background-augmented PCB images from transparent KiCad renders."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


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


def _solid_background(shape: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    height, width = shape
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:, :] = color
    return background


def _gradient_background(
    shape: tuple[int, int],
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
) -> np.ndarray:
    height, width = shape
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    start = np.array(start_color, dtype=np.float32)[None, None, :]
    end = np.array(end_color, dtype=np.float32)[None, None, :]
    gradient = start * (1.0 - t) + end * t
    return np.repeat(gradient, width, axis=1).astype(np.uint8)


def _noise_background(
    shape: tuple[int, int],
    seed: int,
    base_color: tuple[int, int, int],
    noise_sigma: float,
) -> np.ndarray:
    height, width = shape
    rng = np.random.default_rng(seed)
    base = _solid_background(shape, base_color).astype(np.float32)
    noise = rng.normal(0.0, noise_sigma, size=(height, width, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=9, sigmaY=9)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _desk_texture(shape: tuple[int, int], seed: int) -> np.ndarray:
    height, width = shape
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    grain = np.repeat(0.55 + 0.20 * np.sin(55 * x + rng.uniform(-1, 1)), height, axis=0)
    grain += 0.10 * np.sin(11 * y + 20 * x)
    grain += rng.normal(0, 0.03, size=(height, width))
    grain = cv2.GaussianBlur(grain.astype(np.float32), (0, 0), 2)
    grain = np.clip(grain, 0.0, 1.0)
    dark = np.array([75, 80, 84], dtype=np.float32)
    light = np.array([155, 150, 138], dtype=np.float32)
    texture = dark + (light - dark) * grain[:, :, None]
    return np.clip(texture, 0, 255).astype(np.uint8)


def composite_rgba_on_background(
    foreground_rgba: np.ndarray,
    background_bgr: np.ndarray,
    brightness: float = 1.0,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    foreground = foreground_rgba[:, :, :3].astype(np.float32)
    alpha = foreground_rgba[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    if foreground.shape[:2] != background_bgr.shape[:2]:
        background_bgr = cv2.resize(
            background_bgr,
            (foreground.shape[1], foreground.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    background = background_bgr.astype(np.float32)
    composite = foreground * alpha + background * (1.0 - alpha)
    composite = np.clip(composite * brightness, 0, 255).astype(np.uint8)

    if blur_sigma > 0:
        composite = cv2.GaussianBlur(composite, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    return composite


def generate_backgrounds(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "white": _solid_background(shape, (245, 245, 245)),
        "black": _solid_background(shape, (20, 20, 20)),
        "blue_mat": _solid_background(shape, (120, 80, 25)),
        "gray_gradient": _gradient_background(shape, (210, 210, 210), (70, 70, 70)),
        "warm_desk": _desk_texture(shape, seed=7),
        "noisy_lab": _noise_background(shape, seed=13, base_color=(125, 128, 120), noise_sigma=38.0),
    }


def write_augmented_set(
    transparent_image_path: Path,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    foreground = _load_rgba(transparent_image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = generate_backgrounds(foreground.shape[:2])
    written: list[Path] = []

    for index, (name, background) in enumerate(backgrounds.items()):
        brightness = 0.88 + 0.06 * (index % 5)
        blur_sigma = 0.0 if index % 3 else 0.35
        composite = composite_rgba_on_background(
            foreground,
            background,
            brightness=brightness,
            blur_sigma=blur_sigma,
        )
        output_path = output_dir / f"{prefix}_{name}.png"
        cv2.imwrite(str(output_path), composite)
        written.append(output_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite transparent KiCad PCB renders over varied backgrounds.")
    parser.add_argument("--input", required=True, help="Transparent KiCad PNG, usually rendered with --background transparent.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated composite images.")
    parser.add_argument("--prefix", default="board01_top_bg", help="Output filename prefix.")
    args = parser.parse_args()

    written = write_augmented_set(
        transparent_image_path=Path(args.input),
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
    )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
