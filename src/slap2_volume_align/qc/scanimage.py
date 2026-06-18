"""QC helpers for ScanImage volume averaging."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_shift_csv(path: str | Path, rows: list[dict]) -> None:
    """Write per-repeat shift rows to CSV without pandas dependency."""

    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_volume_qc_png(
    path: str | Path,
    volumes_by_channel: dict[int, np.ndarray],
    *,
    max_planes: int = 12,
    max_display_size: int = 512,
) -> None:
    """Write a small mosaic of averaged planes for quick visual QC."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_channels = len(volumes_by_channel)
    if n_channels == 0:
        return

    first = next(iter(volumes_by_channel.values()))
    n_planes = first.shape[0]
    planes = np.linspace(0, n_planes - 1, min(max_planes, n_planes), dtype=int)

    ncols = len(planes)
    nrows = n_channels
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.4 * ncols, 2.4 * nrows),
        squeeze=False,
    )

    for row, (channel_index, volume) in enumerate(sorted(volumes_by_channel.items())):
        for col, z in enumerate(planes):
            ax = axes[row, col]
            img = volume[z]
            # Downsample for display only. The saved TIFF volume is full resolution.
            ds = max(1, int(np.ceil(max(img.shape) / max_display_size)))
            if ds > 1:
                img = img[::ds, ::ds]
            finite = np.isfinite(img)
            if finite.any():
                lo, hi = np.percentile(img[finite], [1, 99.8])
            else:
                lo, hi = 0, 1
            if hi <= lo:
                hi = lo + 1
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"ch {channel_index + 1}, z {z}", fontsize=8)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
