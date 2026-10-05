# Physical trace analysis

This program is for the five repeated physical “あ” drawing trials.

## What it does

The default pipeline:

1. Treats the complete scan as the A4 paper plane.
2. Converts the page to a physical coordinate frame of 210 × 297 mm.
3. Finds the main robot-drawn ink component and defines a local region of interest.
4. Retains nearby disconnected ink pieces inside that region.
5. Skeletonizes the ink trace.
6. Converts skeleton pixels to millimetres.
7. Computes symmetric nearest-curve distances for all trial pairs.
8. If a target XY CSV is supplied, computes physical target-to-trace error.

The script intentionally does **not** translate or rotate one observed drawing to
match another. Therefore, if Trial 5 was physically drawn several millimetres to
the side on the paper, that displacement remains in the absolute repeatability
metric.

## Recommended input

For reproducible analysis, use scans acquired with identical printer settings.

Best:
- PNG, or
- a uniform 300 dpi PDF set.

TIFF is also supported when the TIFF can be decoded normally. Some
multifunction printers produce JPEG-compressed TIFFs that common scientific
libraries reject. If that happens, re-export the same scan as PNG or use PDF.
Do not use screenshots.

## Install dependencies

```bash
pip install numpy pandas pillow opencv-python scipy scikit-image matplotlib pymupdf
```

## Run repeatability analysis only

Windows PowerShell / Command Prompt:

```text
python physical_trace_analysis.py --images DiffusionData_Experiment1.PDF DiffusionData_Experiment2.PDF DiffusionData_Experiment3.PDF DiffusionData_Experiment4.PDF DiffusionData_Experiment5.PDF --output analysis_out
```

This can be run before the authoritative target XY path is available.

Important outputs:

- `analysis_out/extraction_summary.csv`
- `analysis_out/pairwise_repeatability.csv`
- `analysis_out/repeatability_summary.csv`
- `analysis_out/qc/*_extraction_qc.png`
- `analysis_out/observed_points/*_trace_mm.csv`
- `analysis_out/all_traces_overlay_qc.png`

**Inspect every QC image before using the statistics.**

## Add the target trajectory

The target should be a CSV with at least x and y columns. Example:

```csv
stroke,x,y
1,0.000,0.000
1,0.001,0.002
...
```

If the target is in metres:

```text
python physical_trace_analysis.py --images ... --target target_a.csv --target-unit m --target-stroke-column stroke --output analysis_out
```

If its column names are unusual:

```text
--target-x-column YOUR_X_COLUMN --target-y-column YOUR_Y_COLUMN
```

## Target-to-paper coordinate calibration

The observed scan uses:

- origin: bottom-left of the paper
- +x: right
- +y: up
- units: mm

If the target path is in a robot/world coordinate frame, the robot-to-paper
transform must be supplied before an **absolute** target error is scientifically
valid.

The script provides:

```text
--target-scale 1.0
--target-flip-x
--target-flip-y
--target-rotation-deg 0
--target-tx-mm 0
--target-ty-mm 0
```

These parameters transform the authoritative target into the paper frame.

The default is:

```text
--target-align none
```

This preserves absolute placement and is the mode intended for the paper once
the target-to-paper calibration is known.

Two optional diagnostic modes exist:

```text
--target-align translation
--target-align rigid
```

These deliberately remove placement error and therefore produce **shape-only**
metrics. Do not silently report them as absolute physical tracking accuracy.

## Metrics

For two point sets A and B, the program calculates nearest-curve distances in
both directions.

Symmetric mean:

```text
0.5 * [mean d(A→B) + mean d(B→A)]
```

Symmetric RMS:

```text
sqrt(0.5 * [mean d(A→B)^2 + mean d(B→A)^2])
```

Maximum:

```text
max(max d(A→B), max d(B→A))
```

The 95th-percentile distance is also saved as a QC diagnostic because a single
segmentation speck can make the mathematical maximum misleading.

## Intended paper outputs

Once the target calibration is finalized, the principal physical results are:

- mean ± SD RMS geometric target deviation [mm]
- mean maximum geometric target deviation [mm]
- mean pairwise trial-to-trial deviation [mm]

The program also saves intermediate masks, skeletons, physical point clouds, and
QC figures so that the analysis is reproducible and auditable.
