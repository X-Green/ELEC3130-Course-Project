"""Input helpers for images and optional ROI JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .models import BBox, ComponentROI, ImageArray, ROIMap, SolderJointROI
except ImportError:  # pragma: no cover - supports direct script execution
    from models import BBox, ComponentROI, ImageArray, ROIMap, SolderJointROI


def load_image(image_path: str | Path) -> ImageArray:
    """Load an image from disk.

    TODO: Decide whether the project should standardize on RGB or OpenCV BGR.
    For now this returns the raw OpenCV BGR image array.
    """

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError("OpenCV is required to load images. Install opencv-python.") from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def load_roi_json(roi_json_path: str | Path | None) -> ROIMap:
    """Load optional component and solder ROI definitions.

    Expected bbox format is [x, y, width, height] in golden-image coordinates.
    """

    if roi_json_path is None:
        return ROIMap()

    path = Path(roi_json_path)
    if not path.exists():
        raise FileNotFoundError(f"ROI JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    components = [
        ComponentROI(
            id=str(item["id"]),
            type=item.get("type"),
            bbox=_parse_bbox(item["bbox"]),
            expected_angle=float(item.get("expected_angle", 0.0)),
            polarity=item.get("polarity"),
            metadata=item.get("metadata", {}),
        )
        for item in data.get("components", [])
    ]

    solder_joints = [
        SolderJointROI(
            id=str(item["id"]),
            component_id=item.get("component_id"),
            bbox=_parse_bbox(item["bbox"]),
            metadata=item.get("metadata", {}),
        )
        for item in data.get("solder_joints", [])
    ]

    roi_map = ROIMap(components=components, solder_joints=solder_joints)
    validate_roi_map(roi_map)
    return roi_map


def validate_roi_map(roi_map: ROIMap) -> None:
    """Validate ROI values.

    TODO: Add image-bound checks once the final image coordinate convention is fixed.
    """

    for roi in [*roi_map.components, *roi_map.solder_joints]:
        x, y, width, height = roi.bbox
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid bbox for ROI {roi.id}: {roi.bbox}")


def _parse_bbox(value: Any) -> BBox:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"bbox must be [x, y, width, height], got: {value}")
    x, y, width, height = value
    return int(x), int(y), int(width), int(height)

