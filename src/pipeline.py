"""Central orchestrator for the PCBA inspection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:
    from .alignment import register_board
    from .component_inspection import inspect_components
    from .detection import fuse_and_classify_defects
    from .io_utils import load_image, load_roi_json
    from .models import PipelineConfig, PipelineResult
    from .preprocessing import enhance_and_segment
    from .solder_inspection import inspect_solder_joints
except ImportError:  # pragma: no cover - supports direct script execution
    from alignment import register_board
    from component_inspection import inspect_components
    from detection import fuse_and_classify_defects
    from io_utils import load_image, load_roi_json
    from models import PipelineConfig, PipelineResult
    from preprocessing import enhance_and_segment
    from solder_inspection import inspect_solder_joints


def candidate_to_dict(candidate) -> dict:
    return {
        "id": candidate.id,
        "defect_type": candidate.defect_type,
        "bbox": list(candidate.bbox),
        "source": candidate.source,
        "score": candidate.score,
        "features": _json_ready(candidate.features),
        "metadata": _json_ready(candidate.metadata),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_visual_outputs(result: PipelineResult, output_dir: str | Path) -> None:
    write_visual_outputs(result, output_dir)


def write_visual_outputs(result: PipelineResult, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path / "aligned_test.png"), result.alignment.aligned_test_image)
    for name, image in result.alignment.debug_images.items():
        cv2.imwrite(str(output_path / f"{name}.png"), image)

    if result.segmentation.board_mask is not None:
        cv2.imwrite(str(output_path / "board_mask.png"), result.segmentation.board_mask)
    if result.segmentation.component_mask is not None:
        cv2.imwrite(str(output_path / "component_mask.png"), result.segmentation.component_mask)
    if result.segmentation.solder_mask is not None:
        cv2.imwrite(str(output_path / "solder_mask.png"), result.segmentation.solder_mask)
    if result.segmentation.frequency_denoised_test_image is not None:
        cv2.imwrite(str(output_path / "frequency_denoised_test.png"), result.segmentation.frequency_denoised_test_image)
    if result.segmentation.frequency_notch_mask_image is not None:
        cv2.imwrite(str(output_path / "frequency_notch_mask.png"), result.segmentation.frequency_notch_mask_image)
    if result.segmentation.frequency_spectrum_image is not None:
        cv2.imwrite(str(output_path / "frequency_spectrum.png"), result.segmentation.frequency_spectrum_image)
    if result.segmentation.frequency_highpass_image is not None:
        cv2.imwrite(str(output_path / "frequency_highpass.png"), result.segmentation.frequency_highpass_image)
    if result.segmentation.frequency_bandpass_image is not None:
        cv2.imwrite(str(output_path / "frequency_bandpass.png"), result.segmentation.frequency_bandpass_image)
    if result.segmentation.frequency_bandpass_difference_image is not None:
        cv2.imwrite(
            str(output_path / "frequency_bandpass_difference.png"),
            result.segmentation.frequency_bandpass_difference_image,
        )

    overlay = result.alignment.aligned_test_image.copy()
    color_by_source = {
        "component_inspection": (0, 0, 255),
        "solder_inspection": (0, 165, 255),
        "segmentation": (255, 0, 0),
    }

    for defect in result.final_defects:
        x, y, width, height = defect.bbox
        color = color_by_source.get(defect.source, (0, 255, 255))
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            overlay,
            defect.defect_type,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path / "defect_overlay.png"), overlay)

    report = {
        "metadata": result.metadata,
        "alignment": result.alignment.metadata,
        "segmentation": result.segmentation.metadata,
        "component_defects": [candidate_to_dict(defect) for defect in result.component_defects],
        "solder_defects": [candidate_to_dict(defect) for defect in result.solder_defects],
        "defects": [candidate_to_dict(defect) for defect in result.final_defects],
    }
    with (output_path / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def run_pipeline(
    golden_image_path: str | Path,
    test_image_path: str | Path,
    roi_json_path: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the five-step reference-based PCBA inspection workflow."""

    active_config = config or PipelineConfig()

    golden_image = load_image(golden_image_path)
    test_image = load_image(test_image_path)
    roi_map = load_roi_json(roi_json_path)

    alignment = register_board(golden_image, test_image, active_config)
    segmentation = enhance_and_segment(
        golden_image,
        alignment.aligned_test_image,
        roi_map,
        active_config,
    )
    component_defects = inspect_components(
        segmentation.enhanced_golden_image,
        segmentation.enhanced_test_image,
        segmentation,
        roi_map,
        active_config,
    )
    solder_defects = inspect_solder_joints(
        golden_image,
        alignment.aligned_test_image,
        segmentation,
        roi_map,
        active_config,
    )
    if (
        alignment.metadata.get("mean_abs_difference_on_board", 0.0)
        > active_config.solder_global_difference_skip_threshold
        and not component_defects
    ):
        for defect in solder_defects:
            defect.metadata["suppressed_reason"] = "global_appearance_difference"
        solder_defects = []
    final_defects = fuse_and_classify_defects(
        component_defects,
        solder_defects,
        segmentation,
        roi_map,
        active_config,
    )

    return PipelineResult(
        roi_map=roi_map,
        alignment=alignment,
        segmentation=segmentation,
        component_defects=component_defects,
        solder_defects=solder_defects,
        final_defects=final_defects,
        metadata={
            "golden_image_path": str(golden_image_path),
            "test_image_path": str(test_image_path),
            "roi_json_path": str(roi_json_path) if roi_json_path else None,
            "status": "completed",
        },
    )


def main() -> None:
    """CLI entry point for the PCBA inspection pipeline."""

    parser = argparse.ArgumentParser(description="Run the PCBA inspection pipeline.")
    parser.add_argument("--golden", required=True, help="Path to the golden reference image.")
    parser.add_argument("--test", required=True, help="Path to the test PCBA image.")
    parser.add_argument("--roi", default=None, help="Optional path to ROI JSON file.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for masks, overlay, and JSON report.")
    parser.add_argument("--debug", action="store_true", help="Enable debug metadata.")
    args = parser.parse_args()

    result = run_pipeline(
        args.golden,
        args.test,
        args.roi,
        PipelineConfig(debug=args.debug),
    )

    if args.output_dir:
        _write_visual_outputs(result, args.output_dir)

    print(
        "Pipeline completed: "
        f"{len(result.final_defects)} final defects."
    )
    for defect in result.final_defects:
        score = "" if defect.score is None else f", score={defect.score:.3f}"
        print(f"- {defect.defect_type} at {defect.bbox} ({defect.source}{score})")


if __name__ == "__main__":
    main()
