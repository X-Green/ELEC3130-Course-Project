# Final Dataset: KiCad Synthetic PCBA Inspection Dataset

This project uses a reproducible KiCad-based synthetic dataset for classical, reference-based PCBA assembly and solder inspection. The dataset is generated from real KiCad PCB projects, rendered with KiCad CLI, annotated with component and solder-joint ROIs, and evaluated with the existing non-deep-learning inspection pipeline.

The final dataset choice is `data/kicad_random_defects`. It is the best fit for the final ELEC3130 project because it covers all 8 available PCB designs, has balanced defect classes, and contains both component-level and solder-level defects.

## Why We Use A Synthetic KiCad Dataset

Public PCB datasets are useful references, but they do not match this project perfectly:

- Many public PCB datasets focus on bare PCB copper defects, not assembled PCBA defects.
- Many PCBA datasets are component-level crops rather than full golden/test board pairs.
- This project needs controlled golden references, known defect locations, and repeatable evaluation.

KiCad synthesis solves these problems. Each board starts from a real `.kicad_pcb` file. The generator renders a defect-free golden image, creates controlled test variants, and writes ground-truth JSON files in the same coordinate system as the golden image.

This also keeps the project aligned with the course requirement: the inspection method is classical image processing, not deep learning.

## Source Boards

The dataset uses 8 PCB designs listed in `src/kicad_synth_dataset.py`:

```text
board01: SmartTBBatteryChargerNewTest
board02: demo-rec-2004.00685-sensorboard
board03: Matrice4D-NewBattery-HolderPCB
board04: X-Tray-Dev-Board
board05: SuperCap2024V1.2R_Power
board06: SuperCap2024Control_V1.1R_F3_ISOPWR
board07: maglev2025_hardware_test1
board08: maglev2025_hardware_test2
```

KiCad CLI is expected at:

```text
C:\Program Files\KiCad\9.0\bin\kicad-cli.exe
```

## Final Main Dataset

Main dataset path:

```text
data/kicad_random_defects
```

Generator:

```text
src/build_random_defect_dataset.py
```

Composition:

| Item | Count |
| --- | ---: |
| Boards | 8 |
| Cases per board | 16 |
| Total test cases | 128 |
| Golden images | 8 |
| ROI files | 8 |
| Ground-truth files | 128 |
| KiCad component-variant boards | 64 |

Defect class balance:

| Defect type | Cases | Synthesis method |
| --- | ---: | --- |
| `missing_component` | 16 | KiCad board variant |
| `shifted_component` | 16 | KiCad board variant |
| `rotated_component` | 16 | KiCad board variant |
| `visual_component_mismatch` | 16 | KiCad board variant, wrong footprint inserted |
| `solder_bridge` | 16 | PNG-level solder edit |
| `insufficient_solder` | 16 | PNG-level solder edit |
| `excessive_solder` | 16 | PNG-level solder edit |
| `missing_solder` | 16 | PNG-level solder edit |

The component defect cases are generated from KiCad. This is important: missing components, shifted components, rotated components, and wrong-component cases are not just painted on the image. The `.kicad_pcb` file is edited, saved as a variant, and re-rendered by KiCad.

Solder defects are generated at PNG level because solder appearance is not represented as normal movable KiCad components in the same way. The solder edits are still ROI-controlled and measurable.

## Data Layout

```text
data/kicad_random_defects/
  manifest.json
  golden/
    board01_top.png
  transparent/
    board01_top_transparent.png
    board01_random_01_rotated_component_D12_transparent.png
  tests/
    board01_random_01_rotated_component_D12_top.png
    board01_random_04_missing_component_D11_top.png
  roi/
    board01.json
  ground_truth/
    board01_random_01_rotated_component_D12.json
  exports/
    board01_top.svg
    board01_pos.csv
    variant_boards/
      board01_random_01_rotated_component_D12.kicad_pcb
```

Each manifest case contains:

```json
{
  "case_id": "board01_random_04_missing_component_D11",
  "board_id": "board01",
  "golden": "data/kicad_random_defects/golden/board01_top.png",
  "roi": "data/kicad_random_defects/roi/board01.json",
  "test": "data/kicad_random_defects/tests/board01_random_04_missing_component_D11_top.png",
  "ground_truth": "data/kicad_random_defects/ground_truth/board01_random_04_missing_component_D11.json",
  "variant_type": "missing_component"
}
```

## ROI And Ground Truth

Each ROI JSON file stores component boxes and solder-joint boxes in golden-image pixel coordinates:

```json
{
  "board_id": "board01",
  "image_size": [2400, 1800],
  "components": [
    {
      "id": "D11",
      "bbox": [x, y, width, height],
      "type": "LED_SMD..."
    }
  ],
  "solder_joints": [
    {
      "id": "D11-P1",
      "component_id": "D11",
      "bbox": [x, y, width, height]
    }
  ]
}
```

Each ground-truth JSON file stores one defect:

```json
{
  "case_id": "board01_random_04_missing_component_D11",
  "defects": [
    {
      "id": "gt1",
      "defect_type": "missing_component",
      "bbox": [x, y, width, height],
      "component_id": "D11"
    }
  ]
}
```

## How To Generate The Dataset

From the repo root:

```powershell
python -m src.build_random_defect_dataset --output data\kicad_random_defects --seed 3130
```

Useful options:

```powershell
python -m src.build_random_defect_dataset --output data\kicad_random_defects --boards board01 board02
python -m src.build_random_defect_dataset --output data\kicad_random_defects --force-reference
python -m src.build_random_defect_dataset --output data\kicad_random_defects --kicad-cli "C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
```

## How To Validate The Dataset

```powershell
python -m src.kicad_synth_dataset validate --manifest data\kicad_random_defects\manifest.json
```

Current validation result:

```text
Dataset validation passed: 8 boards, 128 cases.
```

## How To Run One Case

```powershell
python -m src.pipeline `
  --golden data\kicad_random_defects\golden\board01_top.png `
  --test data\kicad_random_defects\tests\board01_random_04_missing_component_D11_top.png `
  --roi data\kicad_random_defects\roi\board01.json `
  --output-dir outputs\manual_random_missing_D11
```

The output directory contains:

```text
aligned_test.png
board_mask.png
component_mask.png
solder_mask.png
frequency_spectrum.png
frequency_highpass.png
frequency_bandpass.png
frequency_bandpass_difference.png
defect_overlay.png
report.json
```

## How To Run Full Evaluation

```powershell
python -m src.run_dataset_eval `
  --manifest data\kicad_random_defects\manifest.json `
  --output-dir outputs\kicad_random_defects_eval `
  --verbose `
  --continue-on-error
```

Evaluation outputs:

```text
outputs/kicad_random_defects_eval/
  summary.csv
  metrics.json
  summary.md
  cases/
    <case_id>/
      aligned_test.png
      defect_overlay.png
      report.json
      evaluation.json
      frequency_*.png
      *_mask.png
```

## Current Evaluation Result

Latest evaluation:

```text
outputs/kicad_random_defects_eval/summary.md
```

Overall result:

| Metric | Value |
| --- | ---: |
| Cases evaluated | 128 |
| Overall precision | 0.863 |
| Overall recall | 0.836 |
| Overall F1 score | 0.849 |
| Evaluation errors | 0 |

Per-class result:

| Defect type | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `excessive_solder` | 1.000 | 1.000 | 1.000 |
| `insufficient_solder` | 1.000 | 0.938 | 0.968 |
| `missing_component` | 0.875 | 0.875 | 0.875 |
| `missing_solder` | 1.000 | 0.875 | 0.933 |
| `rotated_component` | 0.640 | 1.000 | 0.780 |
| `shifted_component` | 0.762 | 1.000 | 0.865 |
| `solder_bridge` | 0.941 | 1.000 | 0.970 |
| `visual_component_mismatch` | 0.000 | 0.000 | 0.000 |

Important interpretation:

- The final random-defect dataset is suitable as the main benchmark because it is balanced, covers all 8 boards, and includes both component defects and solder defects.
- The existing classical pipeline performs strongly on missing, shifted, rotated, and solder-related defects.
- `visual_component_mismatch` is intentionally included as a wrong-component benchmark. The current pipeline often produces local component mismatch evidence, but its rule-based classifier does not reliably assign the exact `visual_component_mismatch` label. This should be presented as a known limitation and future work, not hidden from the report.

## Auxiliary Datasets

The project also keeps two auxiliary datasets.

### Original Synthetic Benchmark

Path:

```text
data/kicad_synth
```

Purpose:

- Earlier benchmark for verifying the full pipeline.
- Includes registration/background cases, enhancement stress cases, and labeled defect cases.
- Useful for showing strong baseline performance.

Composition:

| Item | Count |
| --- | ---: |
| Boards | 4 |
| Total cases | 132 |
| Registration/background cases | 48 |
| Enhancement stress cases | 28 |
| Defect cases | 56 |

Latest result:

| Metric | Value |
| --- | ---: |
| Overall precision | 0.930 |
| Overall recall | 0.946 |
| Overall F1 score | 0.938 |
| Defect-free false-positive rate | 0.026 |

### Disturbance Robustness Dataset

Path:

```text
data/kicad_disturbance
```

Purpose:

- Qualitative robustness and limitation testing.
- Simulates realistic photo disturbances: rotation, translation, scale changes, backgrounds, color cast, brightness/contrast change, Gaussian noise, motion blur, JPEG artifacts, periodic noise, and vignetting.
- These cases are defect-free, so this dataset is not the main defect benchmark.

Composition:

| Item | Count |
| --- | ---: |
| Boards | 8 |
| SlightlyDisturbed cases | 64 |
| BadlyDisturbed cases | 64 |
| Total cases | 128 |

Latest result:

| Metric | Value |
| --- | ---: |
| Defect-free false-positive rate | 0.945 |
| SlightlyDisturbed false-positive rate | 0.891 |
| BadlyDisturbed false-positive rate | 1.000 |

Interpretation:

- This dataset is useful for the qualitative webpage demo and for discussing robustness limits.
- It should not be used as the main quantitative defect benchmark because all ground-truth files are defect-free.

## Course Topic Coverage

The dataset supports the ELEC3130 digital image processing story:

| Course topic | Dataset evidence |
| --- | --- |
| Image formation | KiCad-generated fixed-resolution board renders |
| Geometric transforms | Board-level rendered variants and registration diagnostics |
| Spatial enhancement | Low contrast, illumination, and color-normalization examples |
| Frequency-domain processing | FFT spectrum, high-pass, band-pass, and band-pass difference images |
| Image restoration | Noise and motion-blur disturbance cases |
| Color image processing | Color-cast disturbance cases and HSV/Lab solder segmentation |
| Image compression | JPEG disturbance cases |
| Segmentation | Board, component, and solder masks for every evaluated case |
| Morphology | Mask cleanup and connected-component solder/component reasoning |
| Object recognition | Component template matching and local ROI comparison |
| Rule-based classification | Final defect labels, overlays, and evaluation metrics |

## Packaged Files For Teammates

Prepared zip files:

```text
outputs/dataset_packages/kicad_pcba_random_defects.zip
outputs/dataset_packages/kicad_pcba_datasets_full.zip
outputs/dataset_packages/kicad_pcba_report_assets.zip
```

Recommended usage:

- Use `kicad_pcba_random_defects.zip` for the final main dataset.
- Use `kicad_pcba_report_assets.zip` for writing module reports because it contains summaries and representative output images.
- Use `kicad_pcba_datasets_full.zip` only if a teammate needs every dataset together.
