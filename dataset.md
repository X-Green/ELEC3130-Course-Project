# Dataset Research Notes

## Best Fit for This PCBA Pipeline

### 1. SolDef_AI

- Link: https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection
- Paper: https://www.mdpi.com/2504-4494/8/3/117
- Why it fits: this is the closest public dataset to the updated project direction. It contains soldered SMT component images, defect-free and defective assemblies, component positioning defects, and solder quantity defects such as excessive or insufficient solder.
- Use in this repo: best for Task 3 component inspection and Task 4 solder inspection. Because the images are component-level/multi-view rather than full golden-test board pairs, create ROI-level golden references by choosing defect-free samples with the same component/view and compare defective samples against them.
- Limitation: not ideal for Task 1 board-level registration because it is not primarily organized as full-board golden/test image pairs.

## Best Fit for Reference-Based Registration and Evaluation

### 2. DeepPCB

- Link: https://github.com/tangsanli5201/DeepPCB
- Why it fits: it provides 1,500 aligned image pairs. Each pair contains a defect-free template image, a tested image, and bounding-box annotations.
- Use in this repo: best for Task 1 registration, Task 2 preprocessing, and Task 5 quantitative evaluation because the golden/test pair structure matches this pipeline directly.
- Limitation: it is a bare PCB copper-pattern dataset, not a PCBA assembly/solder dataset. It should be treated as an auxiliary benchmark, not the final topic dataset.

## Small Reference-Based PCBA-Style Option

### 3. ChangeChip / CD-PCB

- Paper: https://arxiv.org/abs/2109.05746
- Why it fits: the method is explicitly reference-based and compares an inspected PCB against a golden PCB. The paper also discusses soldering defects plus missing or misaligned electronic elements.
- Use in this repo: useful for demonstrating the same golden-vs-test logic and for presentation background.
- Limitation: CD-PCB is only a small synthesized evaluation set, so it is not enough by itself for strong evaluation.

## Auxiliary Industrial Dataset

### 4. MVTec AD

- Link: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- Why it fits only partially: it is a standard industrial anomaly detection benchmark and includes object categories such as transistor, but it is not PCB/PCBA-specific and does not provide golden PCBA/test PCBA pairs.
- Use in this repo: only for method comparison or explaining anomaly localization, not as the main dataset.

## Recommended Project Choice

Use **SolDef_AI as the main dataset** because the project is now about PCBA assembly and solder inspection. Use **DeepPCB as a secondary benchmark** when the group needs clean reference/template pairs to validate registration, subtraction, bounding boxes, and evaluation metrics.

Suggested workflow:

1. Download SolDef_AI manually from Kaggle.
2. Pick same-view defect-free component images as golden ROI references.
3. Use defective component images as test ROIs for missing/misaligned component and solder quantity experiments.
4. Write ROI JSON files using the existing schema:

```json
{
  "components": [
    {
      "id": "U1",
      "bbox": [90, 85, 65, 35],
      "type": "SMT_component"
    }
  ],
  "solder_joints": [
    {
      "id": "U1-P1",
      "component_id": "U1",
      "bbox": [70, 91, 25, 25]
    }
  ]
}
```

5. Run the implemented pipeline:

```powershell
python -m src.pipeline --golden path\to\golden.png --test path\to\test.png --roi path\to\roi.json --output-dir outputs\sample_01
```

The output directory contains `aligned_test.png`, masks, `defect_overlay.png`, and `report.json`.
