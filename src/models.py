"""Shared data models for the PCBA inspection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

ImageArray: TypeAlias = Any
BBox: TypeAlias = tuple[int, int, int, int]


@dataclass(slots=True)
class ComponentROI:
    """Expected component region in golden-image coordinates."""

    id: str
    bbox: BBox
    type: str | None = None
    expected_angle: float = 0.0
    polarity: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SolderJointROI:
    """Expected solder joint region in golden-image coordinates."""

    id: str
    bbox: BBox
    component_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ROIMap:
    """Manually or automatically defined component and solder regions."""

    components: list[ComponentROI] = field(default_factory=list)
    solder_joints: list[SolderJointROI] = field(default_factory=list)


@dataclass(slots=True)
class PipelineConfig:
    """Shared configuration values for all pipeline stages."""

    min_defect_area: int = 20
    component_match_threshold: float = 0.75
    solder_area_tolerance: float = 0.30
    debug: bool = False


@dataclass(slots=True)
class AlignmentResult:
    """Output from board localization and registration."""

    aligned_test_image: ImageArray
    transform_matrix: Any | None = None
    board_mask: ImageArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentationResult:
    """Output from enhancement and segmentation."""

    enhanced_golden_image: ImageArray
    enhanced_test_image: ImageArray
    board_mask: ImageArray | None = None
    component_mask: ImageArray | None = None
    solder_mask: ImageArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DefectCandidate:
    """Candidate or final defect produced by a task module."""

    id: str
    defect_type: str
    bbox: BBox
    source: str
    score: float | None = None
    mask: ImageArray | None = None
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    """Structured result returned by the full pipeline."""

    roi_map: ROIMap
    alignment: AlignmentResult
    segmentation: SegmentationResult
    component_defects: list[DefectCandidate]
    solder_defects: list[DefectCandidate]
    final_defects: list[DefectCandidate]
    metadata: dict[str, Any] = field(default_factory=dict)

