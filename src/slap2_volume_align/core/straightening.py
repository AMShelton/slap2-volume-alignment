"""Inter-plane z-stack straightening for averaged structural volumes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from scipy import ndimage
from scipy.signal import savgol_filter

from slap2_volume_align.core.registration import estimate_rigid_shift


def _safe_percentile_contrast(volume: np.ndarray) -> np.ndarray:
    contrast = np.empty(volume.shape[0], dtype=float)
    for z in range(volume.shape[0]):
        img = volume[z]
        finite = np.isfinite(img)
        if finite.any():
            vals = img[finite]
            contrast[z] = float(np.percentile(vals, 99.5) - np.percentile(vals, 5.0))
        else:
            contrast[z] = 0.0
    return contrast


def choose_anchor_z(volume_zyx: np.ndarray, *, margin: int = 5) -> int:
    """Choose a high-contrast interior z-plane as the straightening anchor."""

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX volume, got shape {volume.shape}")
    n_z = volume.shape[0]
    contrast = _safe_percentile_contrast(volume)
    if n_z <= 2 * margin:
        return int(np.argmax(contrast))
    valid = np.arange(margin, n_z - margin)
    return int(valid[np.argmax(contrast[valid])])


def _local_template(corrected_preview: list[np.ndarray | None], z_indices: Iterable[int]) -> np.ndarray:
    imgs = [corrected_preview[z] for z in z_indices if corrected_preview[z] is not None]
    if not imgs:
        raise ValueError("No corrected neighboring planes available for local template")
    return np.nanmedian(np.stack(imgs, axis=0), axis=0).astype(np.float32, copy=False)


def _smooth_vector(values: np.ndarray, *, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return values

    # Fill non-finite points by linear interpolation.
    finite = np.isfinite(values)
    if not finite.any():
        filled = np.zeros_like(values)
    elif finite.all():
        filled = values.copy()
    else:
        x = np.arange(n)
        filled = values.copy()
        filled[~finite] = np.interp(x[~finite], x[finite], values[finite])

    if window < 5 or n < 5:
        return filled
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < 5:
        return filled

    return savgol_filter(filled, window_length=window, polyorder=2, mode="interp")


def estimate_z_straightening_shifts(
    volume_zyx: np.ndarray,
    *,
    z_indices: Iterable[int] | None = None,
    anchor_z: int | None = None,
    template_radius: int = 2,
    binning: int = 4,
    upsample_factor: int = 10,
    max_step_shift_px: float = 12.0,
    highpass_sigma_px: float = 16.0,
    smooth_window: int = 9,
    interpolation_order: int = 1,
) -> list[dict]:
    """Estimate per-plane x/y shifts to straighten an averaged z-stack.

    The returned shifts are the y/x translations to apply to each plane. The
    algorithm anchors one high-contrast plane, walks outward in z, aligns each
    plane to a local template made from already-corrected neighboring planes, and
    then smooths the estimated shift trajectory over z.
    """

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX volume, got shape {volume.shape}")

    n_z = volume.shape[0]
    original_z = list(range(n_z)) if z_indices is None else list(z_indices)
    if len(original_z) != n_z:
        raise ValueError("z_indices must have one entry per plane in volume_zyx")

    if anchor_z is None:
        anchor_idx = choose_anchor_z(volume)
    else:
        # Accept either local volume index or original z index.
        if 0 <= anchor_z < n_z:
            anchor_idx = int(anchor_z)
        elif anchor_z in original_z:
            anchor_idx = int(original_z.index(anchor_z))
        else:
            raise ValueError(f"anchor_z={anchor_z} is neither a local nor original z index")

    corrected_preview: list[np.ndarray | None] = [None] * n_z
    corrected_preview[anchor_idx] = volume[anchor_idx].astype(np.float32, copy=False)

    raw_y = np.full(n_z, np.nan, dtype=float)
    raw_x = np.full(n_z, np.nan, dtype=float)
    error = np.full(n_z, np.nan, dtype=float)
    phase_diff = np.full(n_z, np.nan, dtype=float)
    accepted = np.zeros(n_z, dtype=bool)
    reason = [""] * n_z

    raw_y[anchor_idx] = 0.0
    raw_x[anchor_idx] = 0.0
    accepted[anchor_idx] = True
    reason[anchor_idx] = "anchor"

    def walk(z_iter: Iterable[int], neighbor_fn) -> None:
        for z in z_iter:
            template = _local_template(corrected_preview, neighbor_fn(z))
            result = estimate_rigid_shift(
                template,
                volume[z],
                upsample_factor=upsample_factor,
                max_shift_px=max_step_shift_px,
                binning=binning,
                highpass_sigma_px=highpass_sigma_px,
            )
            if result.accepted:
                dy, dx = result.shift_yx
                step_reason = result.reason
            else:
                dy, dx = 0.0, 0.0
                step_reason = result.reason or "rejected"

            raw_y[z] = dy
            raw_x[z] = dx
            error[z] = result.error
            phase_diff[z] = result.phase_difference
            accepted[z] = result.accepted
            reason[z] = step_reason
            corrected_preview[z] = ndimage.shift(
                volume[z].astype(np.float32, copy=False),
                shift=(dy, dx),
                order=interpolation_order,
                mode="nearest",
                prefilter=(interpolation_order > 1),
            ).astype(np.float32, copy=False)

    walk(
        range(anchor_idx + 1, n_z),
        lambda z: range(max(anchor_idx, z - template_radius), z),
    )
    walk(
        range(anchor_idx - 1, -1, -1),
        lambda z: range(z + 1, min(anchor_idx + 1, z + 1 + template_radius)),
    )

    smooth_y = _smooth_vector(raw_y, window=smooth_window)
    smooth_x = _smooth_vector(raw_x, window=smooth_window)

    rows: list[dict] = []
    for local_z in range(n_z):
        rows.append(
            {
                "volume_index": local_z,
                "z_index": int(original_z[local_z]),
                "anchor_volume_index": int(anchor_idx),
                "anchor_z_index": int(original_z[anchor_idx]),
                "raw_shift_y_px": float(raw_y[local_z]),
                "raw_shift_x_px": float(raw_x[local_z]),
                "smooth_shift_y_px": float(smooth_y[local_z]),
                "smooth_shift_x_px": float(smooth_x[local_z]),
                "accepted": bool(accepted[local_z]),
                "error": float(error[local_z]) if np.isfinite(error[local_z]) else "",
                "phase_difference": float(phase_diff[local_z]) if np.isfinite(phase_diff[local_z]) else "",
                "reason": reason[local_z],
            }
        )
    return rows


def apply_z_straightening(
    volume_zyx: np.ndarray,
    shift_rows: list[dict],
    *,
    use_smoothed: bool = True,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Apply estimated z-straightening shifts to a ``ZYX`` volume."""

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX volume, got shape {volume.shape}")
    if len(shift_rows) != volume.shape[0]:
        raise ValueError("shift_rows must have one row per z-plane")

    y_key = "smooth_shift_y_px" if use_smoothed else "raw_shift_y_px"
    x_key = "smooth_shift_x_px" if use_smoothed else "raw_shift_x_px"

    rows = sorted(shift_rows, key=lambda r: int(r["volume_index"]))
    out = np.empty(volume.shape, dtype=np.float32)
    for row in rows:
        z = int(row["volume_index"])
        dy = float(row[y_key])
        dx = float(row[x_key])
        out[z] = ndimage.shift(
            volume[z].astype(np.float32, copy=False),
            shift=(dy, dx),
            order=interpolation_order,
            mode="nearest",
            prefilter=(interpolation_order > 1),
        ).astype(np.float32, copy=False)
    return out


def write_straightening_csv(path: str | Path, rows: list[dict]) -> None:
    """Write z-straightening shift rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_straightening_csv(path: str | Path) -> list[dict]:
    """Read z-straightening shift rows from CSV."""

    with Path(path).open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def make_straightening_qc_png(path: str | Path, rows: list[dict]) -> None:
    """Write a compact diagnostic plot of z-straightening shifts."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    z = np.array([int(r["z_index"]) for r in rows])
    raw_y = np.array([float(r["raw_shift_y_px"]) for r in rows])
    raw_x = np.array([float(r["raw_shift_x_px"]) for r in rows])
    sm_y = np.array([float(r["smooth_shift_y_px"]) for r in rows])
    sm_x = np.array([float(r["smooth_shift_x_px"]) for r in rows])
    accepted = np.array([str(r["accepted"]).lower() == "true" for r in rows])
    anchor_z = int(rows[0]["anchor_z_index"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(z, raw_y, ".", alpha=0.35, label="raw y")
    axes[0].plot(z, sm_y, "-", label="smoothed y")
    axes[0].set_ylabel("Y shift (px)")
    axes[0].legend()

    axes[1].plot(z, raw_x, ".", alpha=0.35, label="raw x")
    axes[1].plot(z, sm_x, "-", label="smoothed x")
    axes[1].set_ylabel("X shift (px)")
    axes[1].legend()

    magnitude = np.hypot(sm_y, sm_x)
    axes[2].plot(z, magnitude, "-", label="smoothed magnitude")
    axes[2].plot(z[~accepted], magnitude[~accepted], "x", label="rejected/raw fallback")
    axes[2].set_ylabel("Shift magnitude (px)")
    axes[2].set_xlabel("z index")
    axes[2].legend()

    for ax in axes:
        ax.axvline(anchor_z, color="k", linestyle="--", alpha=0.4)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def straighten_volume_tiff(
    input_tif: str | Path,
    output_tif: str | Path | None = None,
    *,
    existing_shifts_csv: str | Path | None = None,
    shifts_csv: str | Path | None = None,
    qc_png: str | Path | None = None,
    summary_json: str | Path | None = None,
    anchor_z: int | None = None,
    template_radius: int = 2,
    binning: int = 4,
    upsample_factor: int = 10,
    max_step_shift_px: float = 12.0,
    highpass_sigma_px: float = 16.0,
    smooth_window: int = 9,
    use_smoothed: bool = True,
    interpolation_order: int = 1,
    output_dtype: str = "float32",
    output_compression: str | None = None,
) -> dict:
    """Estimate/apply inter-plane x/y straightening to an averaged volume TIFF."""

    input_tif = Path(input_tif)
    if output_tif is None:
        output_tif = input_tif.with_name(f"{input_tif.stem}_straightened.tif")
    output_tif = Path(output_tif)

    volume = tifffile.imread(input_tif)
    if volume.ndim != 3:
        raise ValueError(f"Expected a ZYX TIFF volume, got shape {volume.shape}")
    volume = volume.astype(np.float32, copy=False)

    if existing_shifts_csv is not None:
        rows = read_straightening_csv(existing_shifts_csv)
        shift_source = str(existing_shifts_csv)
    else:
        rows = estimate_z_straightening_shifts(
            volume,
            anchor_z=anchor_z,
            template_radius=template_radius,
            binning=binning,
            upsample_factor=upsample_factor,
            max_step_shift_px=max_step_shift_px,
            highpass_sigma_px=highpass_sigma_px,
            smooth_window=smooth_window,
            interpolation_order=interpolation_order,
        )
        shift_source = "estimated"

    corrected = apply_z_straightening(
        volume,
        rows,
        use_smoothed=use_smoothed,
        interpolation_order=interpolation_order,
    )

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    data = corrected.astype(output_dtype, copy=False) if output_dtype != "preserve" else corrected
    tifffile.imwrite(
        output_tif,
        data,
        bigtiff=True,
        photometric="minisblack",
        compression=output_compression,
        metadata={"axes": "ZYX"},
        description=f"Z-straightened volume from {input_tif.name}; axes=ZYX",
    )

    if shifts_csv is None and existing_shifts_csv is None:
        shifts_csv = output_tif.with_name(f"{input_tif.stem}_z_straightening_shifts.csv")
    if shifts_csv is not None and existing_shifts_csv is None:
        write_straightening_csv(shifts_csv, rows)

    if qc_png is None:
        qc_png = output_tif.with_name(f"{input_tif.stem}_z_straightening_qc.png")
    if qc_png is not None:
        make_straightening_qc_png(qc_png, rows)

    summary = {
        "input_tif": str(input_tif),
        "output_tif": str(output_tif),
        "shift_source": shift_source,
        "shifts_csv": str(shifts_csv) if shifts_csv is not None else str(existing_shifts_csv),
        "qc_png": str(qc_png) if qc_png is not None else None,
        "config": {
            "anchor_z": anchor_z,
            "template_radius": template_radius,
            "binning": binning,
            "upsample_factor": upsample_factor,
            "max_step_shift_px": max_step_shift_px,
            "highpass_sigma_px": highpass_sigma_px,
            "smooth_window": smooth_window,
            "use_smoothed": use_smoothed,
            "interpolation_order": interpolation_order,
            "output_dtype": output_dtype,
            "output_compression": output_compression,
        },
    }
    if summary_json is None:
        summary_json = output_tif.with_name(f"{input_tif.stem}_z_straightening_summary.json")
    if summary_json is not None:
        Path(summary_json).write_text(json.dumps(summary, indent=2))
        summary["summary_json"] = str(summary_json)
    return summary
