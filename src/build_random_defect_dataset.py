"""Build a balanced random-defect KiCad PCBA dataset.

This dataset matches the disturbance dataset scale: 8 boards x 16 cases. The
component-level defects are synthesized as KiCad board variants and re-rendered
with KiCad; solder-level defects are image-level solder edits on the golden
render, as in the main synthetic dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .build_disturbance_dataset import _ensure_reference_artifacts
    from .kicad_synth_dataset import (
        DEFAULT_BOARD_SOURCES,
        DEFAULT_KICAD_CLI,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        _changed_fraction_in_bbox,
        _component_mm_area,
        _defect_json,
        _draw_excessive_solder,
        _draw_measurable_solder_absence,
        _draw_measurable_solder_bridge,
        _fill_solder_absence,
        _has_reasonable_pad_shape,
        _mean_difference_in_bbox,
        _relative,
        _render_board,
        _roi_by_component,
        _select_components,
        _solder_by_component,
        _white_composite,
        _write_missing_variant,
        _write_repositioned_variant,
        _write_visible_rotation_variant,
        parse_board,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from build_disturbance_dataset import _ensure_reference_artifacts
    from kicad_synth_dataset import (
        DEFAULT_BOARD_SOURCES,
        DEFAULT_KICAD_CLI,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        _changed_fraction_in_bbox,
        _component_mm_area,
        _defect_json,
        _draw_excessive_solder,
        _draw_measurable_solder_absence,
        _draw_measurable_solder_bridge,
        _fill_solder_absence,
        _has_reasonable_pad_shape,
        _mean_difference_in_bbox,
        _relative,
        _render_board,
        _roi_by_component,
        _select_components,
        _solder_by_component,
        _white_composite,
        _write_missing_variant,
        _write_repositioned_variant,
        _write_visible_rotation_variant,
        parse_board,
    )


COMPONENT_DEFECT_TYPES = (
    "missing_component",
    "shifted_component",
    "rotated_component",
    "visual_component_mismatch",
)
SOLDER_DEFECT_TYPES = (
    "solder_bridge",
    "insufficient_solder",
    "excessive_solder",
    "missing_solder",
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _replace_first_at(block: str, x: float, y: float, angle: float) -> str:
    match = re.search(
        r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)(?:\s+([-+]?\d+(?:\.\d+)?))?\s*\)",
        block,
    )
    if not match:
        raise ValueError("Could not find footprint-level (at ...) expression")
    replacement = f"(at {x:.6f} {y:.6f} {angle:.6f})"
    return block[: match.start()] + replacement + block[match.end() :]


def _replace_reference_property(block: str, reference: str) -> str:
    replaced, count = re.subn(
        r'(\(property\s+"Reference"\s+")([^"]+)(")',
        rf"\g<1>{reference}\g<3>",
        block,
        count=1,
    )
    if count:
        return replaced
    return re.sub(
        r'(\(fp_text\s+reference\s+")([^"]+)(")',
        rf"\g<1>{reference}\g<3>",
        block,
        count=1,
    )


def _fresh_uuids(block: str) -> str:
    return re.sub(
        r'\(uuid\s+"[^"]+"\)',
        lambda _: f'(uuid "{uuid.uuid4()}")',
        block,
    )


def _wrong_component_candidates(parsed, target) -> list[Any]:
    target_area = max(_component_mm_area(target), 1e-6)
    candidates: list[Any] = []
    for component in parsed.components:
        if component.reference == target.reference:
            continue
        if component.footprint == target.footprint:
            continue
        if component.layer and component.layer != target.layer:
            continue
        if len(component.pads) < 2:
            continue
        if not _has_reasonable_pad_shape(component):
            continue
        area = _component_mm_area(component)
        if area <= 0:
            continue
        ratio = area / target_area
        if 0.25 <= ratio <= 3.50:
            candidates.append(component)

    if candidates:
        return candidates

    return [
        component
        for component in parsed.components
        if component.reference != target.reference
        and component.footprint != target.footprint
        and len(component.pads) >= 2
    ]


def _write_wrong_component_variant(parsed, target, donor, output_path: Path) -> None:
    replacement = donor.block
    replacement = _replace_first_at(replacement, target.x, target.y, target.angle)
    replacement = _replace_reference_property(replacement, target.reference)
    replacement = _fresh_uuids(replacement)

    start, end = target.span
    variant_text = parsed.text[:start] + replacement + parsed.text[end:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(variant_text, encoding="utf-8")


def _random_shift_parameters(rng: np.random.Generator) -> dict[str, float]:
    dx = float(rng.uniform(0.60, 1.05) * rng.choice([-1.0, 1.0]))
    dy = float(rng.uniform(0.30, 0.80) * rng.choice([-1.0, 1.0]))
    if abs(dx) < abs(dy):
        dx, dy = dy, dx
    return {"dx_mm": dx, "dy_mm": dy}


def _case_metadata(defect_type: str, component_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = {
        "dataset_role": "random_defect",
        "component_id": component_id,
        "synthesis": (
            "kicad_board_variant"
            if defect_type in COMPONENT_DEFECT_TYPES
            else "png_solder_edit"
        ),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _component_case(
    *,
    board_id: str,
    case_index: int,
    defect_type: str,
    component,
    parsed,
    component_bbox: list[int],
    golden_image: np.ndarray,
    output_root: Path,
    kicad_cli: Path,
    width: int,
    height: int,
    rng: np.random.Generator,
    dry_run: bool,
) -> tuple[str, Path, Path, dict[str, Any]]:
    suffix = defect_type.replace("visual_component_mismatch", "wrong_component")
    case_id = f"{board_id}_random_{case_index:02d}_{suffix}_{component.reference}"
    variant_pcb = output_root / "exports" / "variant_boards" / f"{case_id}.kicad_pcb"
    variant_transparent = output_root / "transparent" / f"{case_id}_transparent.png"
    test_path = output_root / "tests" / f"{case_id}_top.png"
    gt_path = output_root / "ground_truth" / f"{case_id}.json"
    metadata: dict[str, Any] = {}

    if defect_type == "missing_component":
        _write_missing_variant(parsed, component, variant_pcb)
        _render_board(
            kicad_cli,
            variant_pcb,
            variant_transparent,
            width,
            height,
            background="transparent",
            force=True,
            dry_run=dry_run,
        )
        if not dry_run:
            _white_composite(variant_transparent, test_path)
    elif defect_type == "shifted_component":
        parameters = _random_shift_parameters(rng)
        _write_repositioned_variant(parsed, component, variant_pcb, **parameters)
        _render_board(
            kicad_cli,
            variant_pcb,
            variant_transparent,
            width,
            height,
            background="transparent",
            force=True,
            dry_run=dry_run,
        )
        if not dry_run:
            _white_composite(variant_transparent, test_path)
        metadata.update(parameters)
    elif defect_type == "rotated_component":
        metadata.update(
            _write_visible_rotation_variant(
                parsed,
                component,
                component_bbox,
                variant_pcb,
                variant_transparent,
                test_path,
                None if dry_run else golden_image,
                kicad_cli,
                width,
                height,
                dry_run,
            )
        )
    else:
        candidates = _wrong_component_candidates(parsed, component)
        if not candidates:
            raise ValueError(f"No wrong-component donor found for {board_id} {component.reference}")
        donor = candidates[int(rng.integers(0, len(candidates)))]
        _write_wrong_component_variant(parsed, component, donor, variant_pcb)
        _render_board(
            kicad_cli,
            variant_pcb,
            variant_transparent,
            width,
            height,
            background="transparent",
            force=True,
            dry_run=dry_run,
        )
        if not dry_run:
            _white_composite(variant_transparent, test_path)
            wrong_image = cv2.imread(str(test_path), cv2.IMREAD_COLOR)
            metadata["changed_fraction"] = _changed_fraction_in_bbox(
                golden_image,
                wrong_image,
                component_bbox,
            )
            metadata["mean_difference"] = _mean_difference_in_bbox(
                golden_image,
                wrong_image,
                component_bbox,
            )
        metadata.update(
            {
                "inserted_reference": donor.reference,
                "inserted_footprint": donor.footprint,
                "expected_footprint": component.footprint,
            }
        )

    ground_truth = _defect_json(
        case_id,
        defect_type,
        component_bbox,
        component_id=component.reference,
        metadata=_case_metadata(defect_type, component.reference, metadata),
    )
    _write_json(gt_path, ground_truth)
    return case_id, test_path, gt_path, metadata


def _solder_case(
    *,
    board_id: str,
    case_index: int,
    defect_type: str,
    component,
    joints: list[dict[str, Any]],
    golden_image: np.ndarray,
    output_root: Path,
    rng: np.random.Generator,
) -> tuple[str, Path, Path, dict[str, Any]]:
    case_id = f"{board_id}_random_{case_index:02d}_{defect_type}_{component.reference}"
    test_path = output_root / "tests" / f"{case_id}_top.png"
    gt_path = output_root / "ground_truth" / f"{case_id}.json"
    image = golden_image.copy()
    metadata: dict[str, Any] = {}

    shuffled = list(joints)
    rng.shuffle(shuffled)
    first = shuffled[0]
    second = shuffled[1] if len(shuffled) > 1 else shuffled[0]

    if defect_type == "solder_bridge":
        gt_bbox = _draw_measurable_solder_bridge(image, golden_image, first["bbox"], second["bbox"])
        solder_joint_id = first["id"]
        metadata["solder_joint_ids"] = [first["id"], second["id"]]
    elif defect_type == "insufficient_solder":
        edited_bbox = _draw_measurable_solder_absence(
            image,
            golden_image,
            first["bbox"],
            target_changed_fraction=0.14,
        )
        gt_bbox = first["bbox"]
        solder_joint_id = first["id"]
        metadata.update(
            {
                "solder_joint_ids": [first["id"]],
                "edited_bbox": edited_bbox,
                "changed_fraction": _changed_fraction_in_bbox(golden_image, image, first["bbox"]),
                "mean_difference": _mean_difference_in_bbox(golden_image, image, first["bbox"]),
            }
        )
    elif defect_type == "excessive_solder":
        _draw_excessive_solder(image, first["bbox"])
        x, y, w, h = first["bbox"]
        gt_bbox = [
            max(0, x - 6),
            max(0, y - 6),
            min(image.shape[1] - max(0, x - 6), w + 12),
            min(image.shape[0] - max(0, y - 6), h + 12),
        ]
        solder_joint_id = first["id"]
        metadata["solder_joint_ids"] = [first["id"]]
    else:
        _fill_solder_absence(image, first["bbox"], strength=1.0)
        gt_bbox = first["bbox"]
        solder_joint_id = first["id"]
        metadata["solder_joint_ids"] = [first["id"]]

    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), image)
    ground_truth = _defect_json(
        case_id,
        defect_type,
        gt_bbox,
        component_id=component.reference,
        solder_joint_id=solder_joint_id,
        metadata=_case_metadata(defect_type, component.reference, metadata),
    )
    _write_json(gt_path, ground_truth)
    return case_id, test_path, gt_path, metadata


def _planned_cases(selected_components: list[Any], rng: np.random.Generator) -> list[tuple[str, Any]]:
    cases = [
        (defect_type, component)
        for component in selected_components
        for defect_type in (*COMPONENT_DEFECT_TYPES, *SOLDER_DEFECT_TYPES)
    ]
    rng.shuffle(cases)
    return cases


def build_random_defect_dataset(args: argparse.Namespace) -> Path:
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
    rng = np.random.default_rng(args.seed)

    for board_id in board_ids:
        source_pcb = DEFAULT_BOARD_SOURCES[board_id]
        board_entry = _ensure_reference_artifacts(
            board_id=board_id,
            source_pcb=source_pcb,
            output_root=output_root,
            kicad_cli=kicad_cli,
            width=args.width,
            height=args.height,
            force=args.force_reference,
            dry_run=args.dry_run,
        )
        manifest_boards.append(board_entry)

        parsed = parse_board(source_pcb)
        selected = _select_components(parsed, count=2)
        roi_json = _read_json(output_root / "roi" / f"{board_id}.json")
        components_by_id = _roi_by_component(roi_json)
        joints_by_component = _solder_by_component(roi_json)
        golden_path = output_root / "golden" / f"{board_id}_top.png"
        golden_image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        if not args.dry_run:
            loaded = cv2.imread(str(golden_path), cv2.IMREAD_COLOR)
            if loaded is None:
                raise FileNotFoundError(f"Failed to load golden image: {golden_path}")
            golden_image = loaded

        for case_index, (defect_type, component) in enumerate(_planned_cases(selected, rng), start=1):
            component_bbox = components_by_id[component.reference]["bbox"]
            if defect_type in COMPONENT_DEFECT_TYPES:
                case_id, test_path, gt_path, metadata = _component_case(
                    board_id=board_id,
                    case_index=case_index,
                    defect_type=defect_type,
                    component=component,
                    parsed=parsed,
                    component_bbox=component_bbox,
                    golden_image=golden_image,
                    output_root=output_root,
                    kicad_cli=kicad_cli,
                    width=args.width,
                    height=args.height,
                    rng=rng,
                    dry_run=args.dry_run,
                )
            else:
                joints = joints_by_component.get(component.reference, [])
                if len(joints) < 2:
                    raise ValueError(f"{board_id} {component.reference} has fewer than two solder joints")
                case_id, test_path, gt_path, metadata = _solder_case(
                    board_id=board_id,
                    case_index=case_index,
                    defect_type=defect_type,
                    component=component,
                    joints=joints,
                    golden_image=golden_image,
                    output_root=output_root,
                    rng=rng,
                )

            manifest_cases.append(
                {
                    "case_id": case_id,
                    "board_id": board_id,
                    "golden": board_entry["golden"],
                    "roi": board_entry["roi"],
                    "test": _relative(test_path, Path.cwd()),
                    "ground_truth": _relative(gt_path, Path.cwd()),
                    "variant_type": defect_type,
                    "metadata": {
                        "dataset_role": "random_defect",
                        "component_id": component.reference,
                        **metadata,
                    },
                }
            )

    manifest = {
        "dataset_version": 1,
        "created_by": "src.build_random_defect_dataset",
        "seed": args.seed,
        "requested_image_size": [args.width, args.height],
        "kicad_cli": str(kicad_cli),
        "boards": manifest_boards,
        "cases": manifest_cases,
        "defect_types": [*COMPONENT_DEFECT_TYPES, *SOLDER_DEFECT_TYPES],
        "cases_per_board": len(COMPONENT_DEFECT_TYPES) * 2 + len(SOLDER_DEFECT_TYPES) * 2,
        "notes": [
            "Missing, shifted, rotated, and wrong-component defects are KiCad board variants.",
            "Wrong-component cases are labeled visual_component_mismatch.",
            "Solder defects are PNG-level solder edits on the golden render.",
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
    parser = argparse.ArgumentParser(description="Build a balanced random-defect KiCad PCBA dataset.")
    parser.add_argument("--output", default="data/kicad_random_defects", help="Output dataset root.")
    parser.add_argument("--seed", type=int, default=3130, help="Deterministic random seed.")
    parser.add_argument("--kicad-cli", default=str(DEFAULT_KICAD_CLI), help="Path to kicad-cli.exe.")
    parser.add_argument("--boards", nargs="*", choices=sorted(DEFAULT_BOARD_SOURCES), help="Board IDs to generate.")
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT, help="Render height in pixels.")
    parser.add_argument("--force-reference", action="store_true", help="Re-render golden/ROI reference artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print KiCad commands and write JSON only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_random_defect_dataset(args)


if __name__ == "__main__":
    main()
