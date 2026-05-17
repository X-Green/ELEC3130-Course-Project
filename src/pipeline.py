"""Central orchestrator for the PCBA inspection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

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
        golden_image,
        alignment.aligned_test_image,
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
            "status": "placeholder_algorithms",
        },
    )


def main() -> None:
    """CLI entry point for the skeleton pipeline."""

    parser = argparse.ArgumentParser(description="Run the PCBA inspection pipeline skeleton.")
    parser.add_argument("--golden", required=True, help="Path to the golden reference image.")
    parser.add_argument("--test", required=True, help="Path to the test PCBA image.")
    parser.add_argument("--roi", default=None, help="Optional path to ROI JSON file.")
    args = parser.parse_args()

    result = run_pipeline(args.golden, args.test, args.roi)
    print(
        "Pipeline completed with placeholder algorithms: "
        f"{len(result.final_defects)} final defects."
    )


if __name__ == "__main__":
    main()
