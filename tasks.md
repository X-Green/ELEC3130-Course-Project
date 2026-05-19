# Updated Project Tasks

## Project Direction

The project topic is updated from bare PCB defect detection to **reference-based PCBA assembly and solder defect inspection using classical image processing**.

The original idea focused mainly on detecting defects in bare PCB copper patterns, such as open circuits, short circuits, missing holes, and extra copper. This is a valid image processing problem, but the domain is relatively narrow and mainly concerns the printed copper layout before electronic components are mounted.

The updated direction expands the domain to **PCBA inspection**, where the board already contains electronic components and solder joints. This makes the project closer to real automated optical inspection (AOI) used in electronics manufacturing. The system will compare a test PCBA image against a golden reference image and detect defects such as missing components, shifted or rotated components, polarity issues, solder bridges, insufficient solder, and excessive solder.

The main techniques include edge detection, image registration, color processing, filtering, segmentation, morphology, connected component analysis, template matching, and rule-based classification.

## Proposed Project Title

**Classical Image Processing Based Reference PCBA Assembly and Solder Defect Inspection**

## Input and Output

### Input

- A golden reference PCBA image showing a correctly assembled board.
- A test PCBA image that may contain defects.
- Optional component and solder pad ROI information, either manually defined or generated from the reference image.

### Output

- Aligned test image.
- Component and solder segmentation masks.
- Detected defect regions.
- Bounding boxes around suspected defects.
- Rule-based defect labels.
- Final visual inspection report.

## Overall Workflow

```text
Golden PCBA image + Test PCBA image
        |
        v
1. Board localization and registration
        |
        v
2. Image enhancement and segmentation
        |
        +---------------------------+
        |                           |
        v                           v
3. Component inspection        4. Solder joint inspection
        |                           |
        +-------------+-------------+
                      v
5. Defect classification, fusion, visualization, and evaluation
```

## Current Implementation Status

The five tasks are now implemented as one reproducible classical image processing pipeline. The implementation is not just a proposal: it has a KiCad-generated dataset, ROI annotations, ground truth, per-case reports, and batch metrics.

| Task | Status | Main files/artifacts |
| --- | --- | --- |
| Task 1: Board localization and registration | Implemented | `src/alignment.py`, `aligned_test.png`, registration metadata |
| Task 2: Image enhancement and segmentation | Implemented | `src/preprocessing.py`, `src/freq_filter.py`, masks, FFT debug images |
| Task 3: Component inspection | Implemented | `src/component_inspection.py`, component defect candidates |
| Task 4: Solder joint inspection | Implemented, needs further tuning for bridge recall | `src/solder_inspection.py`, solder defect candidates |
| Task 5: Fusion, visualization, and evaluation | Implemented | `src/detection.py`, `src/pipeline.py`, `src/run_dataset_eval.py`, reports |

The dataset generator is implemented in `src/kicad_synth_dataset.py`. It renders four KiCad boards, generates golden/test images, writes component and solder-joint ROI files, creates assembly/solder defect cases, creates enhancement stress cases, and builds `data/kicad_synth/manifest.json`.

The latest full dataset evaluation uses:

```powershell
python -m src.run_dataset_eval --manifest data/kicad_synth/manifest.json --output-dir outputs/kicad_synth_eval --verbose --continue-on-error
```

The current acceptance status is:

- Overall F1 target is satisfied.
- Defect-free registration/background false-positive target is satisfied.
- Enhancement stress testing is included for low contrast, illumination gradients, Gaussian noise, motion blur, color cast, JPEG compression, and periodic noise.
- Missing component and shifted component are strong.
- Missing solder and excessive solder are strong.
- Rotated component and solder bridge are implemented but still weaker than the desired per-class recall target, so these should be presented as remaining limitations and possible future work.

## Dataset Coverage Against Course Topics

The dataset is designed to show breadth across the ELEC3130 syllabus while keeping the project focused on one application.

| Course/project topic | Dataset coverage | Purpose |
| --- | --- | --- |
| Image formation and quantification | KiCad renders at fixed resolution with pixel-coordinate ROI/ground truth | Provides controlled image data and measurable annotations |
| Transforms and registration | Camera pan, zoom, rotation, and shift-rotation cases | Tests homography/phase-correlation alignment |
| Morphological image processing | Component/solder masks and solder-shape cases | Tests opening, closing, connected components, and shape cleanup |
| Spatial-domain enhancement | Low-contrast and illumination-gradient stress cases | Tests CLAHE, histogram matching, and brightness normalization |
| Frequency-domain enhancement | Periodic-noise stress cases plus FFT debug images for every case | Tests high-pass, band-pass, spectrum, and band-pass difference images |
| Image restoration | Gaussian-noise and motion-blur stress cases | Tests denoising and robustness to degraded imaging |
| Color image processing | Color-cast stress cases and solder color segmentation | Tests Lab/HSV processing and color normalization |
| Image compression | JPEG-compression stress cases | Tests robustness to compression artifacts |
| Segmentation | Board, component, and solder masks for all cases | Tests Canny, Otsu/adaptive thresholding, HSV/Lab masks, morphology |
| Object recognition | Missing, shifted, and rotated component cases | Tests template matching, edge features, local subtraction, and rule logic |
| Rule-based classification | Solder and component ground-truth defect cases | Tests interpretable classical classification without deep learning |

The enhancement stress cases are intentionally defect-free. They verify that the enhancement and registration stages do not create false assembly or solder defects when the only change is imaging quality.

## Five Balanced Group Tasks

### Task 1: Board Localization and Image Registration

**Goal:**  
Align the test PCBA image with the golden reference image so that later comparison-based methods are reliable.

**Main responsibilities:**

- Detect the board region from the image background.
- Locate board edges, corners, or fiducial markers if available.
- Estimate translation, rotation, scaling, or perspective difference between the golden and test images.
- Warp the test image into the coordinate system of the golden image.
- Evaluate alignment quality using visual overlays and pixel-level difference maps.

**Possible methods:**

- Canny edge detection.
- Hough transform.
- Contour detection.
- Corner detection.
- Fiducial marker matching.
- Template matching.
- Phase correlation.
- Affine or homography transformation.

**Output to next tasks:**

- Registered test image.
- Transformation matrix.
- Board mask or board boundary.

### Task 2: Image Enhancement and Segmentation

**Goal:**  
Reduce the influence of lighting, color, noise, and camera variation, then segment useful regions such as board area, components, pads, and solder regions.

**Main responsibilities:**

- Normalize image brightness and color between the golden and test images.
- Remove noise while preserving component edges and solder boundaries.
- Enhance contrast between components, solder joints, pads, and PCB background.
- Generate binary or multi-class masks for later inspection.
- Compare spatial-domain, frequency-domain, and color-domain processing methods.

**Possible methods:**

- Histogram equalization or histogram matching.
- Gaussian filtering.
- Median filtering.
- Wiener filtering.
- Frequency-domain filtering.
- HSV or Lab color-space conversion.
- Otsu thresholding.
- Adaptive thresholding.
- Morphological opening and closing.

**Implemented spatial-domain enhancement:**

- Convert images to LAB color space.
- Apply CLAHE on the luminance channel to improve local contrast while preserving color.
- Apply bilateral filtering to suppress small noise while preserving component edges.
- Match the test-image color histogram to the golden reference image.
- Produce board, component, and solder masks using Otsu thresholding, Canny edges, HSV/Lab thresholding, and morphology.

**Implemented image enhancement in the frequency domain:**

The pipeline now explicitly demonstrates Week 5 course content. For every aligned test case, the preprocessing stage computes:

- FFT log-magnitude spectrum: shows the distribution of low-frequency background content and high-frequency edge/detail content.
- FFT high-pass image: suppresses smooth board/background illumination and emphasizes component edges, pad boundaries, silkscreen edges, and sharp solder transitions.
- FFT band-pass image: keeps mid-frequency structures such as component outlines, solder blobs, and local pad detail while suppressing both very smooth illumination and very fine noise.
- FFT band-pass golden/test difference image: applies the same band-pass filter to the golden and test images, then subtracts the filtered responses to highlight local detail changes caused by missing, shifted, rotated, or solder-modified regions.

These frequency-domain outputs are saved per case:

```text
outputs/kicad_synth_eval/cases/<case_id>/frequency_spectrum.png
outputs/kicad_synth_eval/cases/<case_id>/frequency_highpass.png
outputs/kicad_synth_eval/cases/<case_id>/frequency_bandpass.png
outputs/kicad_synth_eval/cases/<case_id>/frequency_bandpass_difference.png
```

The batch evaluator also records frequency-domain statistics in `summary.csv` and `metrics.json`, including mean band-pass difference, 95th percentile band-pass difference, changed-pixel fraction, and ground-truth-ROI frequency difference. This makes the frequency-domain step measurable rather than only visual.

The dataset includes `enhancement_stress` cases with `periodic_noise`, so the frequency-domain method is tested against a condition where sinusoidal interference should appear clearly in the FFT spectrum and band-pass difference image.

**Output to next tasks:**

- Enhanced golden and test images.
- Component candidate mask.
- Solder or metallic-region mask.
- Clean binary masks for local defect analysis.
- Frequency-domain debug and difference images for presentation/evidence.

### Task 3: Component Inspection

**Goal:**  
Detect assembly defects related to electronic components, such as missing components, shifted components, rotated components, and possible polarity errors.

**Main responsibilities:**

- Define or extract expected component ROIs from the golden reference image.
- Compare each component ROI in the golden and test images.
- Detect whether a component is missing, displaced, rotated, or visually inconsistent.
- Estimate component position, orientation, and size.
- Produce component-level defect candidates.

**Possible methods:**

- Template matching.
- Normalized cross-correlation.
- Structural similarity index measure.
- Edge difference analysis.
- Contour extraction.
- Bounding box and minimum-area rectangle analysis.
- Color histogram comparison.
- Local image subtraction after registration.

**Possible defect types:**

- Missing component.
- Shifted component.
- Rotated component.
- Wrong orientation or polarity, if visible marks are available.

**Output to final task:**

- Component defect candidates.
- Component bounding boxes.
- Component defect scores.
- Estimated displacement and rotation angle.

### Task 4: Solder Joint Inspection

**Goal:**  
Detect visible solder-related defects around component pins and pads.

**Main responsibilities:**

- Locate solder joint ROIs around expected pads or component terminals.
- Segment solder regions from the board and component background.
- Measure solder shape, area, brightness, and connectivity.
- Detect abnormal solder patterns.
- Produce solder-level defect candidates.

**Possible methods:**

- HSV or Lab color thresholding.
- Specular highlight detection.
- Local adaptive thresholding.
- Morphological filtering.
- Connected component analysis.
- Shape feature extraction.
- Skeleton or connectivity analysis.
- Local image subtraction from golden solder regions.

**Possible defect types:**

- Solder bridge.
- Insufficient solder.
- Excessive solder.
- Missing solder.
- Abnormal solder blob shape.

**Output to final task:**

- Solder defect candidates.
- Solder joint masks.
- Defect bounding boxes.
- Area, shape, and connectivity features.

### Task 5: Defect Classification, Fusion, Visualization, and Evaluation

**Goal:**  
Combine the outputs from component inspection and solder inspection into a final defect report with interpretable rule-based classification and quantitative evaluation.

**Main responsibilities:**

- Merge candidate defects from component and solder inspection.
- Remove duplicate detections and obvious false positives.
- Design rule-based decision logic for final defect labels.
- Generate visual results with masks, bounding boxes, and labels.
- Evaluate the system using available annotations or manually labeled test images.
- Prepare final presentation figures and result tables.

**Possible methods:**

- Connected component feature analysis.
- Rule-based classification.
- Graph or connectivity-based reasoning.
- Non-maximum suppression for overlapping boxes.
- Precision, recall, F1 score, and confusion matrix.
- Visual overlay generation.

**Final output:**

- Final defect list.
- Defect type for each detected region.
- Bounding box visualization.
- Evaluation metrics.
- Complete project demo pipeline.

## Why This Five-Step Split Is Balanced

Each group member owns one complete module with clear input, processing methods, and output. The modules are connected in a single workflow, so the project remains cohesive instead of becoming five unrelated experiments.

This split also avoids assigning one member a weak or isolated task. For example, frequency-domain filtering is not treated as a standalone project step because it is mainly an enhancement method. Instead, it is included in Task 2 together with color correction, denoising, and segmentation, making that task more substantial and better connected to the rest of the workflow.

## Scope

- Component missing detection.
- Component shift or rotation detection.
- Solder bridge detection.
- Insufficient or excessive solder detection.
- Final rule-based classification and visualization.

## Reproducible Commands

Generate or refresh the dataset:

```powershell
python -m src.kicad_synth_dataset generate --output data/kicad_synth --seed 3130 --force
```

Validate the dataset:

```powershell
python -m src.kicad_synth_dataset validate --manifest data/kicad_synth/manifest.json
```

Run one image-pair pipeline:

```powershell
python -m src.pipeline --golden data/kicad_synth/golden/board01_top.png --test data/kicad_synth/tests/board01_missing_D11_top.png --roi data/kicad_synth/roi/board01.json --output-dir outputs/manual_board01_missing_D11
```

Run the full dataset evaluation:

```powershell
python -m src.run_dataset_eval --manifest data/kicad_synth/manifest.json --output-dir outputs/kicad_synth_eval --verbose --continue-on-error
```

Expected dataset composition after generation:

- 4 golden board images.
- 4 ROI files.
- 48 registration/background cases.
- 28 enhancement stress cases.
- 24 assembly defect cases.
- 32 solder defect cases.
- 132 total manifest cases.

## Presentation Story

The project should be presented as a complete classical AOI system:

1. KiCad rendering gives a controlled, reproducible PCBA dataset with known labels.
2. Board registration maps every test image into the golden coordinate system.
3. Spatial and frequency-domain enhancement reduce lighting/background effects and emphasize useful structures.
4. Color segmentation and morphology extract board, component, and solder evidence.
5. Component ROIs use template matching, edge/difference features, displacement, and rotation estimates.
6. Solder ROIs use HSV/Lab metallic masks, local area/shape/connectivity features, and bridge rules.
7. Rule-based fusion produces interpretable labels and overlays.
8. Batch evaluation reports precision, recall, F1, false-positive rate, confusion information, and representative figures.
