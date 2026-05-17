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

**Output to next tasks:**

- Enhanced golden and test images.
- Component candidate mask.
- Solder or metallic-region mask.
- Clean binary masks for local defect analysis.

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
