#!/usr/bin/env python3
"""
Physical ink-trace analysis for repeated robot drawing experiments.

Purpose
-------
1. Register each scan to an A4 paper coordinate frame by mapping the full scanned
   page to known paper dimensions.
2. Isolate the robot-drawn ink trace without using the drawing itself to
   translate/rotate the page.
3. Skeletonize the ink and convert it to millimetres.
4. Compute symmetric geometric distances between repeated trials.
5. Optionally compare each physical trace against an authoritative target XY path.

Important interpretation
------------------------
The default analysis preserves each drawing's absolute position on the paper.
It does NOT automatically translate/rotate a trial to make it match another
trial or the target. This is intentional: otherwise a genuine physical placement
error could be hidden.

If --target-align translation or --target-align rigid is used, those results are
shape-only diagnostics and should not replace the absolute physical result.

Dependencies
------------
numpy, pandas, pillow, opencv-python, scipy, scikit-image, matplotlib
Optional for PDF input: PyMuPDF (package name: pymupdf)

Example
-------
python physical_trace_analysis.py ^
  --images DiffusionData_Experiment1.PDF DiffusionData_Experiment2.PDF ^
           DiffusionData_Experiment3.PDF DiffusionData_Experiment4.PDF ^
           DiffusionData_Experiment5.PDF ^
  --output analysis_out

With a target path:
python physical_trace_analysis.py ^
  --images DiffusionData_Experiment1.PDF DiffusionData_Experiment2.PDF ^
           DiffusionData_Experiment3.PDF DiffusionData_Experiment4.PDF ^
           DiffusionData_Experiment5.PDF ^
  --target target_a.csv --target-unit m ^
  --output analysis_out
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


@dataclass
class TraceResult:
    name: str
    source: Path
    gray: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    points_mm: np.ndarray
    roi_px: tuple[int, int, int, int]
    threshold_used: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract physical robot ink traces and compute geometric repeatability."
    )
    p.add_argument("--images", nargs="+", required=True,
                   help="Scan files. Supported: PDF, PNG, TIFF/TIF, JPEG/JPG.")
    p.add_argument("--output", default="physical_trace_analysis",
                   help="Output directory.")
    p.add_argument("--paper-width-mm", type=float, default=210.0)
    p.add_argument("--paper-height-mm", type=float, default=297.0)
    p.add_argument("--pdf-dpi", type=float, default=300.0,
                   help="Rendering DPI for PDF input. Does not create new source detail.")
    p.add_argument("--threshold", default="200",
                   help='Ink threshold 0-255, or "otsu". Default: 200.')
    p.add_argument("--roi-padding-mm", type=float, default=7.0,
                   help="Padding around automatically detected main drawing.")
    p.add_argument("--min-component-mm2", type=float, default=0.04,
                   help="Remove tiny isolated dark specks below this area.")
    p.add_argument("--target", default=None,
                   help="Optional target CSV containing x/y columns.")
    p.add_argument("--target-unit", choices=["mm", "m"], default="mm")
    p.add_argument("--target-x-column", default=None)
    p.add_argument("--target-y-column", default=None)
    p.add_argument("--target-stroke-column", default=None,
                   help="Optional stroke/segment ID column used only for densification.")
    p.add_argument("--target-densify-mm", type=float, default=0.05)
    p.add_argument("--target-scale", type=float, default=1.0,
                   help="Additional scale after unit conversion.")
    p.add_argument("--target-flip-x", action="store_true")
    p.add_argument("--target-flip-y", action="store_true")
    p.add_argument("--target-rotation-deg", type=float, default=0.0)
    p.add_argument("--target-tx-mm", type=float, default=0.0)
    p.add_argument("--target-ty-mm", type=float, default=0.0)
    p.add_argument("--target-align", choices=["none", "translation", "rigid"],
                   default="none",
                   help="Default 'none' preserves absolute placement. Other modes are shape-only.")
    p.add_argument("--pairwise-align", choices=["none", "translation"], default="none",
                   help="Optional trial-to-trial alignment used only for repeatability analysis. "
                        "Use 'translation' to remove global sheet-placement differences.")
    p.add_argument("--rotate-90-cw", action="store_true",
                   help="Rotate the complete paper coordinate frame 90 degrees clockwise. "
                        "The same rigid transform is applied to every trial, so geometric "
                        "distance metrics are unchanged.")
    return p.parse_args()


def load_scan(path: Path, pdf_dpi: float) -> np.ndarray:
    """Return an RGB uint8 array. Reject obviously failed decodes."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RuntimeError(
                "PDF input requires PyMuPDF. Install with: pip install pymupdf"
            ) from e
        doc = fitz.open(path)
        if len(doc) < 1:
            raise RuntimeError(f"No page found in {path}")
        page = doc[0]
        scale = float(pdf_dpi) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = arr[:, :, :3]
        elif pix.n == 1:
            arr = np.repeat(arr, 3, axis=2)
        rgb = arr.copy()
        doc.close()
    else:
        try:
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB")).copy()
        except Exception as e:
            raise RuntimeError(
                f"Could not decode {path}. If this is a JPEG-compressed TIFF from a "
                "multifunction printer, re-export the same scan as PNG or use the PDF "
                "version; do not use a screenshot."
            ) from e

    # Scientific-safety check: some malformed TIFF decoders can return a flat image.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dynamic = int(gray.max()) - int(gray.min())
    if gray.std() < 2.0 or dynamic < 15:
        raise RuntimeError(
            f"{path} decoded to an almost uniform image (std={gray.std():.3f}, "
            f"range={dynamic}). Treat this as a decode failure and use PNG/PDF."
        )
    return rgb


def threshold_ink(gray: np.ndarray, threshold_arg: str) -> tuple[np.ndarray, float]:
    if threshold_arg.lower() == "otsu":
        t, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return (binary > 0).astype(np.uint8), float(t)

    t = float(threshold_arg)
    if not 0 <= t <= 255:
        raise ValueError("--threshold must be 0..255 or 'otsu'")
    return (gray < t).astype(np.uint8), t


def choose_main_component(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the principal drawing component.

    Labels/notes near the page edge are intentionally disfavoured.
    The component itself is used only to define an ROI; all legitimate dark
    components inside that ROI are retained later.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        raise RuntimeError("No dark connected components were detected.")

    H, W = mask.shape
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]

        if area < 20:
            continue
        # Avoid page-edge handwritten trial labels and scanner edge artifacts.
        if not (0.05 * W <= cx <= 0.95 * W and 0.15 * H <= cy <= 0.95 * H):
            continue
        # A drawing should not occupy most of the whole page.
        if w > 0.8 * W or h > 0.8 * H:
            continue

        candidates.append((area, x, y, w, h))

    if not candidates:
        # Fallback: largest non-background component.
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )

    _, x, y, w, h = max(candidates, key=lambda z: z[0])
    return x, y, w, h


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area_px:
            out[labels == i] = 1
    return out


def pixel_to_paper_mm(
    xx: np.ndarray,
    yy: np.ndarray,
    width_px: int,
    height_px: int,
    paper_width_mm: float,
    paper_height_mm: float,
) -> np.ndarray:
    """
    Paper frame:
      origin = bottom-left of the scanned page
      +x = right
      +y = up
    """
    x_mm = xx.astype(float) * paper_width_mm / max(width_px - 1, 1)
    y_mm = paper_height_mm - (
        yy.astype(float) * paper_height_mm / max(height_px - 1, 1)
    )
    return np.column_stack([x_mm, y_mm])


def extract_trace(
    source: Path,
    paper_width_mm: float,
    paper_height_mm: float,
    pdf_dpi: float,
    threshold_arg: str,
    roi_padding_mm: float,
    min_component_mm2: float,
) -> TraceResult:
    rgb = load_scan(source, pdf_dpi)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    raw_mask, threshold_used = threshold_ink(gray, threshold_arg)

    H, W = gray.shape
    x, y, w, h = choose_main_component(raw_mask)

    px_per_mm_x = W / paper_width_mm
    px_per_mm_y = H / paper_height_mm
    pad_px = int(round(roi_padding_mm * 0.5 * (px_per_mm_x + px_per_mm_y)))

    x0 = max(0, x - pad_px)
    y0 = max(0, y - pad_px)
    x1 = min(W, x + w + pad_px)
    y1 = min(H, y + h + pad_px)

    roi_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    roi_mask[y0:y1, x0:x1] = raw_mask[y0:y1, x0:x1]

    px_per_mm2 = px_per_mm_x * px_per_mm_y
    min_area_px = max(2, int(round(min_component_mm2 * px_per_mm2)))
    clean = remove_small_components(roi_mask, min_area_px)

    skel = skeletonize(clean.astype(bool))
    yy, xx = np.nonzero(skel)
    if len(xx) < 50:
        raise RuntimeError(
            f"Too few skeleton pixels were extracted from {source}: {len(xx)}"
        )

    points_mm = pixel_to_paper_mm(
        xx, yy, W, H, paper_width_mm, paper_height_mm
    )
    return TraceResult(
        name=source.stem,
        source=source,
        gray=gray,
        mask=clean,
        skeleton=skel,
        points_mm=points_mm,
        roi_px=(x0, y0, x1, y1),
        threshold_used=threshold_used,
    )


def symmetric_metrics(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    """
    Symmetric nearest-curve metrics.

    mean_mm:
        0.5 * [mean d(A->B) + mean d(B->A)]
    rms_mm:
        sqrt(0.5 * [mean d(A->B)^2 + mean d(B->A)^2])
    max_mm:
        maximum directed nearest-neighbour distance in either direction
    p95_mm:
        95th percentile of the pooled directed nearest distances (QC diagnostic)
    """
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Distance metric received an empty point set.")

    tree_A = cKDTree(A)
    tree_B = cKDTree(B)
    d_A_to_B = tree_B.query(A, k=1)[0]
    d_B_to_A = tree_A.query(B, k=1)[0]

    pooled = np.concatenate([d_A_to_B, d_B_to_A])
    return {
        "mean_mm": 0.5 * (float(d_A_to_B.mean()) + float(d_B_to_A.mean())),
        "rms_mm": math.sqrt(
            0.5 * (
                float(np.mean(d_A_to_B ** 2))
                + float(np.mean(d_B_to_A ** 2))
            )
        ),
        "max_mm": float(max(d_A_to_B.max(), d_B_to_A.max())),
        "p95_mm": float(np.percentile(pooled, 95)),
    }


def infer_xy_columns(df: pd.DataFrame, x_col: str | None, y_col: str | None):
    if x_col is not None and y_col is not None:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Requested target x/y columns were not found.")
        return x_col, y_col

    lower = {str(c).lower(): c for c in df.columns}
    x_candidates = ["x_mm", "x", "cart_x", "pos_x", "px"]
    y_candidates = ["y_mm", "y", "cart_y", "pos_y", "py"]

    xc = next((lower[c] for c in x_candidates if c in lower), None)
    yc = next((lower[c] for c in y_candidates if c in lower), None)
    if xc is None or yc is None:
        raise ValueError(
            "Could not infer target x/y columns. Use --target-x-column and "
            "--target-y-column."
        )
    return xc, yc


def densify_polyline(points: np.ndarray, spacing_mm: float) -> np.ndarray:
    if len(points) < 2 or spacing_mm <= 0:
        return points.copy()

    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        d = float(np.linalg.norm(b - a))
        if d == 0:
            continue
        n = max(1, int(math.ceil(d / spacing_mm)))
        ts = np.linspace(0.0, 1.0, n + 1)[1:]
        out.extend(a[None, :] + ts[:, None] * (b - a)[None, :])
    return np.asarray(out, dtype=float)


def transform_target(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    P = points.astype(float).copy()
    if args.target_unit == "m":
        P *= 1000.0
    P *= float(args.target_scale)

    if args.target_flip_x:
        P[:, 0] *= -1.0
    if args.target_flip_y:
        P[:, 1] *= -1.0

    theta = math.radians(float(args.target_rotation_deg))
    R = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ])
    P = P @ R.T
    P[:, 0] += float(args.target_tx_mm)
    P[:, 1] += float(args.target_ty_mm)
    return P


def load_target(args: argparse.Namespace) -> np.ndarray:
    df = pd.read_csv(args.target)
    x_col, y_col = infer_xy_columns(
        df, args.target_x_column, args.target_y_column
    )

    if args.target_stroke_column:
        if args.target_stroke_column not in df.columns:
            raise ValueError(
                f"Stroke column '{args.target_stroke_column}' was not found."
            )
        groups = []
        for _, g in df.groupby(args.target_stroke_column, sort=False):
            pts = g[[x_col, y_col]].to_numpy(float)
            pts = transform_target(pts, args)
            groups.append(densify_polyline(pts, args.target_densify_mm))
        target = np.vstack(groups)
    else:
        pts = df[[x_col, y_col]].to_numpy(float)
        pts = transform_target(pts, args)
        target = densify_polyline(pts, args.target_densify_mm)

    if not np.isfinite(target).all():
        raise ValueError("Target contains NaN or infinite coordinates.")
    return target


def rotate_90_cw(points: np.ndarray, paper_width_mm: float) -> np.ndarray:
    """
    Rotate a complete paper-coordinate point set 90 degrees clockwise.

    Original portrait-paper frame:
        0 <= x <= paper_width_mm
        0 <= y <= paper_height_mm
        +x right, +y up

    Rotated landscape-paper frame:
        x' = y
        y' = paper_width_mm - x

    The added translation keeps coordinates positive. Because this is one common
    rigid transform, pairwise Euclidean distances and RMS/maximum deviations are
    unchanged.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("rotate_90_cw expects an N x 2 array.")

    x_new = P[:, 1]
    y_new = float(paper_width_mm) - P[:, 0]
    return np.column_stack([x_new, y_new])


def best_translation(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Shape-only diagnostic: remove centroid translation."""
    return moving + (fixed.mean(axis=0) - moving.mean(axis=0))



def translation_icp(moving: np.ndarray, fixed: np.ndarray, max_iter: int = 100, tol: float = 1e-7):
    """
    Translation-only ICP.
    Returns the aligned point set and the applied translation vector.
    Intended for repeatability analysis when paper placement varies between trials.
    """
    X = best_translation(moving.copy(), fixed)
    total_t = fixed.mean(axis=0) - moving.mean(axis=0)
    prev = np.inf
    tree = cKDTree(fixed)
    for _ in range(max_iter):
        d, idx = tree.query(X, k=1)
        Y = fixed[idx]
        delta = Y.mean(axis=0) - X.mean(axis=0)
        X = X + delta
        total_t = total_t + delta
        err = float(np.mean(d))
        if abs(prev - err) < tol:
            break
        prev = err
    return X, total_t

def rigid_icp(
    moving: np.ndarray,
    fixed: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    2-D rigid ICP (rotation + translation, no scaling).
    Intended only for shape-only diagnostics.
    """
    X = best_translation(moving.copy(), fixed)
    prev = np.inf

    tree = cKDTree(fixed)
    for _ in range(max_iter):
        d, idx = tree.query(X, k=1)
        Y = fixed[idx]

        cx = X.mean(axis=0)
        cy = Y.mean(axis=0)
        Xc = X - cx
        Yc = Y - cy

        H = Xc.T @ Yc
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        X = (X - cx) @ R.T + cy
        err = float(np.mean(d))
        if abs(prev - err) < tol:
            break
        prev = err
    return X


def save_trace_outputs(result: TraceResult, outdir: Path) -> None:
    pts_dir = outdir / "observed_points"
    qc_dir = outdir / "qc"
    pts_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result.points_mm, columns=["x_mm", "y_mm"]).to_csv(
        pts_dir / f"{result.name}_trace_mm.csv", index=False
    )

    cv2.imwrite(
        str(qc_dir / f"{result.name}_mask.png"),
        (result.mask * 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(qc_dir / f"{result.name}_skeleton.png"),
        (result.skeleton.astype(np.uint8) * 255),
    )

    x0, y0, x1, y1 = result.roi_px
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(result.gray, cmap="gray")
    yy, xx = np.nonzero(result.skeleton)
    ax.scatter(xx, yy, s=0.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(result.name)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.tight_layout()
    fig.savefig(qc_dir / f"{result.name}_extraction_qc.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    sources = [Path(x) for x in args.images]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    traces: list[TraceResult] = []
    extraction_rows = []

    for src in sources:
        print(f"[extract] {src}")
        tr = extract_trace(
            src,
            paper_width_mm=args.paper_width_mm,
            paper_height_mm=args.paper_height_mm,
            pdf_dpi=args.pdf_dpi,
            threshold_arg=args.threshold,
            roi_padding_mm=args.roi_padding_mm,
            min_component_mm2=args.min_component_mm2,
        )
        if args.rotate_90_cw:
            tr.points_mm = rotate_90_cw(
                tr.points_mm,
                paper_width_mm=args.paper_width_mm,
            )

        traces.append(tr)
        save_trace_outputs(tr, outdir)

        H, W = tr.gray.shape
        extraction_rows.append({
            "trial": tr.name,
            "source": str(src),
            "width_px": W,
            "height_px": H,
            "paper_width_mm": args.paper_width_mm,
            "paper_height_mm": args.paper_height_mm,
            "threshold_used": tr.threshold_used,
            "skeleton_points": len(tr.points_mm),
            "roi_x0_px": tr.roi_px[0],
            "roi_y0_px": tr.roi_px[1],
            "roi_x1_px": tr.roi_px[2],
            "roi_y1_px": tr.roi_px[3],
            "trace_min_x_mm": float(tr.points_mm[:, 0].min()),
            "trace_max_x_mm": float(tr.points_mm[:, 0].max()),
            "trace_min_y_mm": float(tr.points_mm[:, 1].min()),
            "trace_max_y_mm": float(tr.points_mm[:, 1].max()),
        })

    pd.DataFrame(extraction_rows).to_csv(
        outdir / "extraction_summary.csv", index=False
    )

    # Absolute paper-frame repeatability.
    pair_rows = []
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            m = symmetric_metrics(traces[i].points_mm, traces[j].points_mm)
            pair_rows.append({
                "trial_a": traces[i].name,
                "trial_b": traces[j].name,
                **m,
            })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(outdir / "pairwise_repeatability.csv", index=False)

    if len(pair_df):
        repeatability_summary = pd.DataFrame([{
            "n_trials": len(traces),
            "n_pairs": len(pair_df),
            "mean_pairwise_mean_deviation_mm": float(pair_df["mean_mm"].mean()),
            "sd_pairwise_mean_deviation_mm": float(pair_df["mean_mm"].std(ddof=1))
                if len(pair_df) > 1 else np.nan,
            "mean_pairwise_rms_deviation_mm": float(pair_df["rms_mm"].mean()),
            "mean_pairwise_max_deviation_mm": float(pair_df["max_mm"].mean()),
            "mean_pairwise_p95_deviation_mm": float(pair_df["p95_mm"].mean()),
        }])
        repeatability_summary.to_csv(
            outdir / "repeatability_summary.csv", index=False
        )

    target = None

    # Optional translation-aligned repeatability (shape repeatability with sheet-placement removed).
    if args.pairwise_align == "translation" and len(traces) >= 2:
        reference = traces[0]
        aligned_points = {reference.name: reference.points_mm.copy()}
        alignment_rows = [{
            "trial": reference.name,
            "reference_trial": reference.name,
            "tx_mm": 0.0,
            "ty_mm": 0.0,
        }]
        for tr in traces[1:]:
            aligned_pts, tvec = translation_icp(tr.points_mm, reference.points_mm)
            aligned_points[tr.name] = aligned_pts
            alignment_rows.append({
                "trial": tr.name,
                "reference_trial": reference.name,
                "tx_mm": float(tvec[0]),
                "ty_mm": float(tvec[1]),
            })

        pd.DataFrame(alignment_rows).to_csv(
            outdir / "translation_alignment_summary.csv", index=False
        )

        pair_rows_aligned = []
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                A = aligned_points[traces[i].name]
                B = aligned_points[traces[j].name]
                m = symmetric_metrics(A, B)
                pair_rows_aligned.append({
                    "trial_a": traces[i].name,
                    "trial_b": traces[j].name,
                    **m,
                })
        pair_df_aligned = pd.DataFrame(pair_rows_aligned)
        pair_df_aligned.to_csv(
            outdir / "pairwise_repeatability_translation_aligned.csv", index=False
        )

        if len(pair_df_aligned):
            repeatability_summary_aligned = pd.DataFrame([{
                "n_trials": len(traces),
                "n_pairs": len(pair_df_aligned),
                "reference_trial": reference.name,
                "mean_pairwise_mean_deviation_mm": float(pair_df_aligned["mean_mm"].mean()),
                "sd_pairwise_mean_deviation_mm": float(pair_df_aligned["mean_mm"].std(ddof=1))
                    if len(pair_df_aligned) > 1 else np.nan,
                "mean_pairwise_rms_deviation_mm": float(pair_df_aligned["rms_mm"].mean()),
                "mean_pairwise_max_deviation_mm": float(pair_df_aligned["max_mm"].mean()),
                "mean_pairwise_p95_deviation_mm": float(pair_df_aligned["p95_mm"].mean()),
            }])
            repeatability_summary_aligned.to_csv(
                outdir / "repeatability_summary_translation_aligned.csv", index=False
            )

        fig, ax = plt.subplots(figsize=(7, 6))
        for tr in traces:
            pts = aligned_points[tr.name]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.8, label=tr.name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("aligned x [mm]")
        ax.set_ylabel("aligned y [mm]")
        ax.legend()
        ax.set_title(
            f"Translation-aligned physical traces (reference: {reference.name})"
        )
        fig.tight_layout()
        fig.savefig(outdir / "all_traces_overlay_translation_aligned.png", dpi=300)
        plt.close(fig)

    if args.target:
        target = load_target(args)
        if args.rotate_90_cw:
            target = rotate_90_cw(
                target,
                paper_width_mm=args.paper_width_mm,
            )
        pd.DataFrame(target, columns=["x_mm", "y_mm"]).to_csv(
            outdir / "target_transformed_mm.csv", index=False
        )

        target_rows = []
        for tr in traces:
            target_for_trial = target.copy()

            if args.target_align == "translation":
                target_for_trial = best_translation(
                    target_for_trial, tr.points_mm
                )
            elif args.target_align == "rigid":
                target_for_trial = rigid_icp(
                    target_for_trial, tr.points_mm
                )

            m = symmetric_metrics(tr.points_mm, target_for_trial)
            target_rows.append({
                "trial": tr.name,
                "alignment_mode": args.target_align,
                **m,
            })

        target_df = pd.DataFrame(target_rows)
        target_df.to_csv(outdir / "target_deviation_by_trial.csv", index=False)

        summary = pd.DataFrame([{
            "n_trials": len(target_df),
            "alignment_mode": args.target_align,
            "mean_rms_mm": float(target_df["rms_mm"].mean()),
            "sd_rms_mm": float(target_df["rms_mm"].std(ddof=1))
                if len(target_df) > 1 else np.nan,
            "mean_max_mm": float(target_df["max_mm"].mean()),
            "sd_max_mm": float(target_df["max_mm"].std(ddof=1))
                if len(target_df) > 1 else np.nan,
            "mean_symmetric_mean_mm": float(target_df["mean_mm"].mean()),
            "mean_p95_mm": float(target_df["p95_mm"].mean()),
        }])
        summary.to_csv(outdir / "target_deviation_summary.csv", index=False)

    # QC overlay. Scatter is used because skeleton points are not temporally ordered.
    fig, ax = plt.subplots(figsize=(7, 6))
    for tr in traces:
        ax.scatter(
            tr.points_mm[:, 0],
            tr.points_mm[:, 1],
            s=0.8,
            label=tr.name,
        )
    if target is not None:
        ax.scatter(
            target[:, 0],
            target[:, 1],
            s=1.0,
            label="target",
        )

    ax.set_aspect("equal", adjustable="box")
    if args.rotate_90_cw:
        ax.set_xlabel("rotated paper x [mm]")
        ax.set_ylabel("rotated paper y [mm]")
        ax.set_title("Extracted physical traces (90° clockwise paper frame)")
    else:
        ax.set_xlabel("paper x [mm]")
        ax.set_ylabel("paper y [mm]")
        ax.set_title("Extracted physical traces in the paper coordinate frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "all_traces_overlay_qc.png", dpi=300)
    plt.close(fig)

    notes = [
        (
            "Coordinate frame: 90 degrees clockwise from the original paper frame; "
            "x'=y and y'=paper_width-x."
            if args.rotate_90_cw
            else "Coordinate frame: origin at bottom-left of the scanned page; +x right; +y up."
        ),
        "The full scanned page is mapped to the specified paper dimensions.",
        "No trial-to-trial translation/rotation is applied for pairwise repeatability.",
        "Symmetric curve distances use nearest-point distances in both directions.",
        "Inspect every *_extraction_qc.png before accepting statistics.",
        "The reported maximum is sensitive to segmentation artifacts; p95 is saved as a QC diagnostic.",
    ]

    # Optional translation-aligned repeatability (shape repeatability with sheet-placement removed).
    if args.pairwise_align == "translation" and len(traces) >= 2:
        reference = traces[0]
        aligned_points = {reference.name: reference.points_mm.copy()}
        alignment_rows = [{
            "trial": reference.name,
            "reference_trial": reference.name,
            "tx_mm": 0.0,
            "ty_mm": 0.0,
        }]
        for tr in traces[1:]:
            aligned_pts, tvec = translation_icp(tr.points_mm, reference.points_mm)
            aligned_points[tr.name] = aligned_pts
            alignment_rows.append({
                "trial": tr.name,
                "reference_trial": reference.name,
                "tx_mm": float(tvec[0]),
                "ty_mm": float(tvec[1]),
            })

        pd.DataFrame(alignment_rows).to_csv(
            outdir / "translation_alignment_summary.csv", index=False
        )

        pair_rows_aligned = []
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                A = aligned_points[traces[i].name]
                B = aligned_points[traces[j].name]
                m = symmetric_metrics(A, B)
                pair_rows_aligned.append({
                    "trial_a": traces[i].name,
                    "trial_b": traces[j].name,
                    **m,
                })
        pair_df_aligned = pd.DataFrame(pair_rows_aligned)
        pair_df_aligned.to_csv(
            outdir / "pairwise_repeatability_translation_aligned.csv", index=False
        )

        if len(pair_df_aligned):
            repeatability_summary_aligned = pd.DataFrame([{
                "n_trials": len(traces),
                "n_pairs": len(pair_df_aligned),
                "reference_trial": reference.name,
                "mean_pairwise_mean_deviation_mm": float(pair_df_aligned["mean_mm"].mean()),
                "sd_pairwise_mean_deviation_mm": float(pair_df_aligned["mean_mm"].std(ddof=1))
                    if len(pair_df_aligned) > 1 else np.nan,
                "mean_pairwise_rms_deviation_mm": float(pair_df_aligned["rms_mm"].mean()),
                "mean_pairwise_max_deviation_mm": float(pair_df_aligned["max_mm"].mean()),
                "mean_pairwise_p95_deviation_mm": float(pair_df_aligned["p95_mm"].mean()),
            }])
            repeatability_summary_aligned.to_csv(
                outdir / "repeatability_summary_translation_aligned.csv", index=False
            )

        fig, ax = plt.subplots(figsize=(7, 6))
        for tr in traces:
            pts = aligned_points[tr.name]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.8, label=tr.name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("aligned x [mm]")
        ax.set_ylabel("aligned y [mm]")
        ax.legend()
        ax.set_title(
            f"Translation-aligned physical traces (reference: {reference.name})"
        )
        fig.tight_layout()
        fig.savefig(outdir / "all_traces_overlay_translation_aligned.png", dpi=300)
        plt.close(fig)

    if args.target:
        notes.append(
            f"Target alignment mode: {args.target_align}. "
            "Only 'none' preserves absolute target placement."
        )
    (outdir / "analysis_notes.txt").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )

    print(f"\nDone. Results written to: {outdir.resolve()}")
    print("Before using any metric in the paper, inspect the QC extraction images.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
