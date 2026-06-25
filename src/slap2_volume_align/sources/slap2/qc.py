"""QC plotting for SLAP2 DMD footprint and merge workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from slap2_volume_align.sources.slap2.geometry import (
    SampleGridSpec,
    infer_z_overlap_um,
    stack_corners_sample_xy,
)
from slap2_volume_align.sources.slap2.metadata import Slap2ReferenceStackSpec


def plot_dmd_footprints(
    specs: list[Slap2ReferenceStackSpec],
    *,
    labels: Optional[list[str]] = None,
    grid: Optional[SampleGridSpec] = None,
    out_path: Optional[str | Path] = None,
):
    """Plot sample-space XY footprints and z-ranges for DMD stacks."""

    if labels is None:
        labels = [f"DMD{i + 1}" for i in range(len(specs))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_xy, ax_z = axes

    for spec, label in zip(specs, labels):
        corners = stack_corners_sample_xy(spec)
        closed = np.vstack([corners, corners[0]])
        ax_xy.plot(closed[:, 0], closed[:, 1], marker="o", label=label)
        center = corners.mean(axis=0)
        ax_xy.text(center[0], center[1], label, ha="center", va="center")

        ax_z.plot([0, 1], [spec.z_min_um, spec.z_max_um], marker="o", label=label)

    if len(specs) >= 2:
        try:
            z0, z1 = infer_z_overlap_um(specs[0], specs[1])
            ax_z.axhspan(z0, z1, alpha=0.2, label="z overlap")
        except Exception:
            pass

    if grid is not None:
        gx = [grid.x_min_um, grid.x_max_um, grid.x_max_um, grid.x_min_um, grid.x_min_um]
        gy = [grid.y_min_um, grid.y_min_um, grid.y_max_um, grid.y_max_um, grid.y_min_um]
        ax_xy.plot(gx, gy, linestyle="--", linewidth=1.0, label="output grid")

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("sample x (µm)")
    ax_xy.set_ylabel("sample y (µm)")
    ax_xy.set_title("DMD footprints in sample coordinates")
    ax_xy.legend()
    ax_xy.grid(alpha=0.25)

    ax_z.set_xlim(-0.2, 1.2)
    ax_z.set_xticks([])
    ax_z.set_ylabel("z (µm)")
    ax_z.set_title("Axial coverage")
    ax_z.legend()
    ax_z.grid(alpha=0.25)

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
    return fig


def save_merge_qc_png(
    *,
    dmd1_projection: np.ndarray,
    dmd2_projection: np.ndarray,
    super_projection: np.ndarray,
    out_path: str | Path,
    residual_shift_yx_px: Optional[tuple[float, float]] = None,
) -> None:
    """Save a compact projection QC figure for a merged/preview volume."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [dmd1_projection, dmd2_projection, super_projection]
    titles = ["DMD1 warped", "DMD2 warped", "Merged / overlay"]

    for ax, img, title in zip(axes, images, titles):
        finite = np.isfinite(img)
        if finite.any():
            lo, hi = np.nanpercentile(img[finite], [1, 99.7])
        else:
            lo, hi = 0, 1
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")

    if residual_shift_yx_px is not None:
        fig.suptitle(
            f"DMD2 residual shift applied: y={residual_shift_yx_px[0]:.2f}px, x={residual_shift_yx_px[1]:.2f}px"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
