"""Generate alignment-only debug visualizations for one image pair or dataset case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

try:
    from .alignment import register_board
    from .io_utils import load_image
    from .models import PipelineConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from alignment import register_board
    from io_utils import load_image
    from models import PipelineConfig


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_path(raw_path: str | Path, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    manifest_candidate = manifest_path.parent / path
    if manifest_candidate.exists():
        return manifest_candidate

    return cwd_path


def _case_from_manifest(manifest_path: Path, case_id: str) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    for case in manifest.get("cases", []):
        if case.get("case_id") == case_id:
            return case
    raise ValueError(f"Case id not found in manifest: {case_id}")


def _write_debug_outputs(args: argparse.Namespace) -> Path:
    if args.case_id:
        manifest_path = Path(args.manifest)
        case = _case_from_manifest(manifest_path, args.case_id)
        golden_path = _resolve_path(case["golden"], manifest_path)
        test_path = _resolve_path(case["test"], manifest_path)
        output_dir = Path(args.output_dir) / args.case_id
    else:
        if not args.golden or not args.test:
            raise ValueError("Use either --case-id or both --golden and --test.")
        golden_path = Path(args.golden)
        test_path = Path(args.test)
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = register_board(
        load_image(golden_path),
        load_image(test_path),
        PipelineConfig(debug=True),
    )

    cv2.imwrite(str(output_dir / "aligned_test.png"), result.aligned_test_image)
    if result.board_mask is not None:
        cv2.imwrite(str(output_dir / "alignment_board_mask.png"), result.board_mask)
    for name, image in result.debug_images.items():
        cv2.imwrite(str(output_dir / f"{name}.png"), image)

    with (output_dir / "alignment_report.json").open("w", encoding="utf-8") as file:
        json.dump(result.metadata, file, indent=2)

    print(f"Wrote alignment debug outputs to {output_dir}")
    print(json.dumps(result.metadata, indent=2))
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write alignment debug images and metadata.")
    parser.add_argument("--manifest", default="data/kicad_synth/manifest.json", help="Dataset manifest path.")
    parser.add_argument("--case-id", default=None, help="Dataset case id to debug.")
    parser.add_argument("--golden", default=None, help="Golden image path for manual pair debugging.")
    parser.add_argument("--test", default=None, help="Test image path for manual pair debugging.")
    parser.add_argument("--output-dir", default="outputs/alignment_debug", help="Debug output directory.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _write_debug_outputs(args)


if __name__ == "__main__":
    main()
