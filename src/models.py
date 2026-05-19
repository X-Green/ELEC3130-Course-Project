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
    registration_features: int = 2000
    registration_min_matches: int = 10
    registration_ransac_reproj_threshold: float = 5.0
    component_match_threshold: float = 0.75
    component_missing_threshold: float = 0.45
    component_shift_tolerance_px: float = 8.0
    component_rotation_tolerance_deg: float = 20.0
    component_search_margin_ratio: float = 0.45
    solder_area_tolerance: float = 0.30
    solder_bridge_overlap_rois: int = 2
    evaluation_iou_threshold: float = 0.50
    enable_frequency_debug: bool = True
    enable_frequency_notch_filter: bool = True
    frequency_notch_min_radius_ratio: float = 0.015
    frequency_notch_max_radius_ratio: float = 0.48
    frequency_notch_peak_sigma: float = 6.0
    frequency_notch_radius: int = 5
    frequency_notch_max_peaks: int = 24
    enable_segmentation_fallback: bool = False
    solder_global_difference_skip_threshold: float = 20.0
    debug: bool = False


@dataclass(slots=True)
class AlignmentResult:
    """Output from board localization and registration."""

    aligned_test_image: ImageArray
    transform_matrix: Any | None = None
    board_mask: ImageArray | None = None
    debug_images: dict[str, ImageArray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentationResult:
    """Output from enhancement and segmentation."""

    enhanced_golden_image: ImageArray
    enhanced_test_image: ImageArray
    board_mask: ImageArray | None = None
    component_mask: ImageArray | None = None
    solder_mask: ImageArray | None = None
    frequency_denoised_test_image: ImageArray | None = None
    frequency_notch_mask_image: ImageArray | None = None
    frequency_spectrum_image: ImageArray | None = None
    frequency_highpass_image: ImageArray | None = None
    frequency_bandpass_image: ImageArray | None = None
    frequency_bandpass_difference_image: ImageArray | None = None
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
