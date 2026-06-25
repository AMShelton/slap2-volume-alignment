"""Warp and merge SLAP2 GUI reference stacks from multiple DMDs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, map_coordinates, shift as ndi_shift
from skimage.registration import phase_cross_correlation

from slap2_volume_align.core.registration import bin_image, highpass_for_registration
from slap2_volume_align.sources.slap2.geometry import (
    SampleGridSpec,
    compute_output_grid,
    infer_z_overlap_um,
    sample_to_pixel_xy,
)
from slap2_volume_align.sources.slap2.metadata import (
    Slap2ReferenceStackSpec,
    read_reference_stack_spec,
)
from slap2_volume_align.sources.slap2.qc import plot_dmd_footprints, save_merge_qc_png
from slap2_volume_align.sources.slap2.reference_stack import read_z_interpolated_plane


@dataclass(frozen=True)
class Slap2MergeConfig:
    """Configuration for merging two SLAP2 reference stacks."""

    dmd1_tif: str | Path
    dmd2_tif: str | Path
    out_dir: str | Path
    channel: int = 1
    xy_resolution_um: Optional[float] = None
    z_resolution_um: Optional[float] = None
    z_grid: str = "first"
    padding_um: float = 2.0
    z_interp_method: str = "linear"
    output_dtype: str = "float32"
    output_compression: Optional[str] = None
    fine_register_overlap: bool = False
    residual_upsample_factor: int = 10
    residual_registration_binning: int = 2
    residual_highpass_sigma_px: float = 8.0
    residual_max_shift_px: float = 50.0
    xy_feather_px: float = 32.0
    z_feather: bool = True
    write_intermediates: bool = True
    write_qc_png: bool = True


def _jsonify(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_sample_mesh(grid: SampleGridSpec) -> tuple[np.ndarray, np.ndarray]:
    y_um = grid.y_coords_um
    x_um = grid.x_coords_um
    return np.meshgrid(x_um, y_um)


def _valid_weight_for_transform(
    transform: np.ndarray,
    image_shape: tuple[int, int],
    grid: SampleGridSpec,
    *,
    xy_feather_px: float = 32.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute inverse-map pixel coordinates and XY feather weights."""

    h, w = image_shape
    x_um, y_um = _make_sample_mesh(grid)
    x_px, y_px = sample_to_pixel_xy(transform, x_um, y_um)
    valid = (x_px >= 0) & (x_px <= w - 1) & (y_px >= 0) & (y_px <= h - 1)

    if xy_feather_px and xy_feather_px > 0:
        dist = distance_transform_edt(valid)
        weight = np.clip(dist / float(xy_feather_px), 0.0, 1.0).astype(np.float32)
    else:
        weight = valid.astype(np.float32)

    return y_px.astype(np.float32), x_px.astype(np.float32), weight


def _warp_plane_to_grid(
    image: np.ndarray,
    y_px: np.ndarray,
    x_px: np.ndarray,
    valid_weight: np.ndarray,
    *,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp one image plane to the output sample grid."""

    coords = np.vstack([y_px.ravel(), x_px.ravel()])
    warped = map_coordinates(
        image.astype(np.float32, copy=False),
        coords,
        order=order,
        mode="constant",
        cval=np.nan,
    ).reshape(y_px.shape)
    valid = np.isfinite(warped) & (valid_weight > 0)
    warped = np.where(valid, warped, 0.0).astype(np.float32)
    return warped, valid_weight.astype(np.float32)


def _z_weight(
    z_um: float,
    *,
    which: str,
    overlap: Optional[tuple[float, float]],
    z_feather: bool = True,
) -> float:
    if overlap is None or not z_feather:
        return 1.0
    z0, z1 = overlap
    if z1 <= z0 or z_um < z0 or z_um > z1:
        return 1.0
    frac = (z_um - z0) / (z1 - z0)
    if which == "dmd1":
        return float(1.0 - frac)
    if which == "dmd2":
        return float(frac)
    return 1.0


def warp_reference_stack_to_grid(
    tif_path: str | Path,
    spec: Slap2ReferenceStackSpec,
    grid: SampleGridSpec,
    *,
    channel: int = 1,
    z_interp_method: str = "linear",
    xy_feather_px: float = 32.0,
    z_weight_kind: str = "none",
    z_overlap_um: Optional[tuple[float, float]] = None,
    z_feather: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp one reference stack to a common sample-coordinate grid.

    Returns ``(sum_volume, weight_volume)`` as float32 arrays with shape ZYX.
    """

    y_px, x_px, xy_weight = _valid_weight_for_transform(
        spec.representative_transform,
        spec.image_shape,
        grid,
        xy_feather_px=xy_feather_px,
    )

    out = np.zeros((grid.z_size, grid.y_size, grid.x_size), dtype=np.float32)
    weights = np.zeros_like(out)

    for zi, z_um in enumerate(grid.z_coords_um):
        img = read_z_interpolated_plane(
            tif_path,
            spec,
            float(z_um),
            channel=channel,
            method=z_interp_method,
        )
        if img is None:
            continue

        warped, w_xy = _warp_plane_to_grid(img, y_px, x_px, xy_weight)
        wz = _z_weight(float(z_um), which=z_weight_kind, overlap=z_overlap_um, z_feather=z_feather)
        w = w_xy * float(wz)
        out[zi] += warped * w
        weights[zi] += w * (w_xy > 0)

    return out, weights


def normalize_sum_weight(sum_volume: np.ndarray, weight_volume: np.ndarray) -> np.ndarray:
    """Compute weighted average, returning NaN outside valid regions."""

    out = np.full(sum_volume.shape, np.nan, dtype=np.float32)
    mask = weight_volume > 0
    out[mask] = sum_volume[mask] / weight_volume[mask]
    return out


def _max_projection_for_registration(vol: np.ndarray) -> np.ndarray:
    valid = np.isfinite(vol)
    if not valid.any():
        return np.zeros(vol.shape[1:], dtype=np.float32)
    tmp = np.where(valid, vol, np.nan)
    return np.nanmax(tmp, axis=0).astype(np.float32)


def estimate_overlap_residual_shift_px(
    dmd1_warped: np.ndarray,
    dmd2_warped: np.ndarray,
    *,
    upsample_factor: int = 10,
    registration_binning: int = 2,
    highpass_sigma_px: float = 8.0,
    max_shift_px: float = 50.0,
) -> tuple[float, float, dict]:
    """Estimate residual YX pixel shift to apply to DMD2 after metadata warp."""

    p1 = _max_projection_for_registration(dmd1_warped)
    p2 = _max_projection_for_registration(dmd2_warped)

    # Restrict registration to the pixels where both projections are finite/nonzero.
    common = np.isfinite(p1) & np.isfinite(p2) & (p1 != 0) & (p2 != 0)
    if common.sum() < 100:
        return 0.0, 0.0, {"accepted": False, "reason": "insufficient common overlap pixels"}

    p1 = np.where(common, p1, 0.0)
    p2 = np.where(common, p2, 0.0)

    sigma_binned = max(highpass_sigma_px / max(registration_binning, 1), 0.0)
    fixed = highpass_for_registration(
        bin_image(p1, registration_binning),
        sigma_px=sigma_binned,
    )
    moving = highpass_for_registration(
        bin_image(p2, registration_binning),
        sigma_px=sigma_binned,
    )
    shift_binned, error, phase = phase_cross_correlation(
        fixed,
        moving,
        upsample_factor=upsample_factor,
    )

    dy = float(shift_binned[0] * registration_binning)
    dx = float(shift_binned[1] * registration_binning)
    mag = float(np.hypot(dy, dx))
    accepted = bool(np.isfinite(mag) and mag <= max_shift_px)
    if not accepted:
        dy, dx = 0.0, 0.0

    return dy, dx, {
        "accepted": accepted,
        "shift_y_px": dy,
        "shift_x_px": dx,
        "magnitude_px": mag,
        "error": float(error),
        "phase": float(phase),
        "common_pixels": int(common.sum()),
    }


def apply_residual_shift_to_sum_weight(
    sum_volume: np.ndarray,
    weight_volume: np.ndarray,
    shift_yx_px: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply residual YX shift to a warped DMD sum/weight volume."""

    dy, dx = shift_yx_px
    if np.isclose(dy, 0.0) and np.isclose(dx, 0.0):
        return sum_volume, weight_volume
    shifted_sum = ndi_shift(sum_volume, shift=(0, dy, dx), order=1, mode="constant", cval=0.0)
    shifted_w = ndi_shift(weight_volume, shift=(0, dy, dx), order=1, mode="constant", cval=0.0)
    return shifted_sum.astype(np.float32), shifted_w.astype(np.float32)


def write_imagej_tiff(path: str | Path, volume: np.ndarray, *, compression: Optional[str] = None) -> None:
    """Write a Fiji/ImageJ-compatible ZYX TIFF."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        volume.astype(np.float32, copy=False),
        imagej=True,
        bigtiff=True,
        metadata={"axes": "ZYX"},
        compression=compression,
    )


def make_slap2_footprint_summary(
    dmd1_tif: str | Path,
    dmd2_tif: str | Path,
    out_dir: str | Path,
    *,
    xy_resolution_um: Optional[float] = None,
    z_resolution_um: Optional[float] = None,
) -> dict:
    """Parse metadata, compute common grid, and write a footprint QC PNG."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec1 = read_reference_stack_spec(dmd1_tif)
    spec2 = read_reference_stack_spec(dmd2_tif)
    grid = compute_output_grid(
        [spec1, spec2],
        xy_resolution_um=xy_resolution_um,
        z_resolution_um=z_resolution_um,
        z_grid="first",
    )
    plot_dmd_footprints(
        [spec1, spec2],
        labels=["DMD1", "DMD2"],
        grid=grid,
        out_path=out_dir / "slap2_dmd_footprints_qc.png",
    )

    summary = {
        "dmd1": spec1.to_dict(),
        "dmd2": spec2.to_dict(),
        "z_overlap_um": list(infer_z_overlap_um(spec1, spec2)),
        "output_grid": grid.to_dict(),
        "footprint_qc_png": str(out_dir / "slap2_dmd_footprints_qc.png"),
    }
    (out_dir / "slap2_dmd_footprints_summary.json").write_text(json.dumps(summary, indent=2, default=_jsonify))
    return summary



def merge_dmd_reference_stack_specs(
    config: Slap2MergeConfig,
    spec1: Slap2ReferenceStackSpec,
    spec2: Slap2ReferenceStackSpec,
) -> dict:
    """Merge two SLAP2 reference stack specs.

    This variant is useful in notebooks for Fiji-created substacks whose JSON
    page metadata was stripped. In that case create specs with
    ``make_manual_reference_stack_spec`` and call this function directly.
    """

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dmd1_tif = Path(config.dmd1_tif)
    dmd2_tif = Path(config.dmd2_tif)

    grid = compute_output_grid(
        [spec1, spec2],
        xy_resolution_um=config.xy_resolution_um,
        z_resolution_um=config.z_resolution_um,
        padding_um=config.padding_um,
        z_grid=config.z_grid,
    )
    overlap = infer_z_overlap_um(spec1, spec2)

    # Warp separately so we can inspect and optionally residual-align before blending.
    dmd1_sum, dmd1_w = warp_reference_stack_to_grid(
        dmd1_tif,
        spec1,
        grid,
        channel=config.channel,
        z_interp_method=config.z_interp_method,
        xy_feather_px=config.xy_feather_px,
        z_weight_kind="dmd1",
        z_overlap_um=overlap,
        z_feather=config.z_feather,
    )
    dmd2_sum, dmd2_w = warp_reference_stack_to_grid(
        dmd2_tif,
        spec2,
        grid,
        channel=config.channel,
        z_interp_method=config.z_interp_method,
        xy_feather_px=config.xy_feather_px,
        z_weight_kind="dmd2",
        z_overlap_um=overlap,
        z_feather=config.z_feather,
    )

    residual = {"accepted": False, "shift_y_px": 0.0, "shift_x_px": 0.0, "reason": "not requested"}
    if config.fine_register_overlap:
        # Estimate only over z-overlap planes.
        z = grid.z_coords_um
        zmask = (z >= overlap[0]) & (z <= overlap[1])
        dmd1_avg_overlap = normalize_sum_weight(dmd1_sum[zmask], dmd1_w[zmask])
        dmd2_avg_overlap = normalize_sum_weight(dmd2_sum[zmask], dmd2_w[zmask])
        dy, dx, residual = estimate_overlap_residual_shift_px(
            dmd1_avg_overlap,
            dmd2_avg_overlap,
            upsample_factor=config.residual_upsample_factor,
            registration_binning=config.residual_registration_binning,
            highpass_sigma_px=config.residual_highpass_sigma_px,
            max_shift_px=config.residual_max_shift_px,
        )
        dmd2_sum, dmd2_w = apply_residual_shift_to_sum_weight(dmd2_sum, dmd2_w, (dy, dx))

    merged_sum = dmd1_sum + dmd2_sum
    merged_w = dmd1_w + dmd2_w
    merged = normalize_sum_weight(merged_sum, merged_w)
    # Fiji displays NaNs poorly; write zeros outside the valid footprint, but preserve weights sidecar.
    merged_for_tiff = np.nan_to_num(merged, nan=0.0).astype(config.output_dtype)

    out_tif = out_dir / f"{dmd1_tif.stem}_plus_{dmd2_tif.stem}_super_stack_ch{config.channel}.tif"
    write_imagej_tiff(out_tif, merged_for_tiff, compression=config.output_compression)

    dmd1_tif_out = None
    dmd2_tif_out = None
    weights_tif_out = None
    if config.write_intermediates:
        dmd1_tif_out = out_dir / f"{dmd1_tif.stem}_warped_ch{config.channel}.tif"
        dmd2_tif_out = out_dir / f"{dmd2_tif.stem}_warped_ch{config.channel}.tif"
        weights_tif_out = out_dir / "merge_weights.tif"
        write_imagej_tiff(dmd1_tif_out, np.nan_to_num(normalize_sum_weight(dmd1_sum, dmd1_w), nan=0.0), compression=config.output_compression)
        write_imagej_tiff(dmd2_tif_out, np.nan_to_num(normalize_sum_weight(dmd2_sum, dmd2_w), nan=0.0), compression=config.output_compression)
        write_imagej_tiff(weights_tif_out, merged_w.astype(np.float32), compression=config.output_compression)

    qc_png = None
    footprint_png = None
    if config.write_qc_png:
        footprint_png = out_dir / "slap2_dmd_footprints_qc.png"
        plot_dmd_footprints([spec1, spec2], labels=["DMD1", "DMD2"], grid=grid, out_path=footprint_png)
        qc_png = out_dir / "slap2_super_stack_merge_qc.png"
        save_merge_qc_png(
            dmd1_projection=_max_projection_for_registration(normalize_sum_weight(dmd1_sum, dmd1_w)),
            dmd2_projection=_max_projection_for_registration(normalize_sum_weight(dmd2_sum, dmd2_w)),
            super_projection=_max_projection_for_registration(merged),
            out_path=qc_png,
            residual_shift_yx_px=(float(residual.get("shift_y_px", 0.0)), float(residual.get("shift_x_px", 0.0))),
        )

    summary = {
        "config": asdict(config),
        "dmd1": spec1.to_dict(),
        "dmd2": spec2.to_dict(),
        "output_grid": grid.to_dict(),
        "z_overlap_um": list(overlap),
        "residual_registration": residual,
        "outputs": {
            "super_stack": str(out_tif),
            "dmd1_warped": str(dmd1_tif_out) if dmd1_tif_out else None,
            "dmd2_warped": str(dmd2_tif_out) if dmd2_tif_out else None,
            "weights": str(weights_tif_out) if weights_tif_out else None,
            "footprint_qc_png": str(footprint_png) if footprint_png else None,
            "merge_qc_png": str(qc_png) if qc_png else None,
        },
    }
    summary_path = out_dir / "slap2_super_stack_merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_jsonify))
    return summary


def merge_dmd_reference_stacks(config: Slap2MergeConfig) -> dict:
    """Merge two SLAP2 GUI reference stacks into one Fiji-compatible super stack."""

    spec1 = read_reference_stack_spec(config.dmd1_tif)
    spec2 = read_reference_stack_spec(config.dmd2_tif)
    return merge_dmd_reference_stack_specs(config, spec1, spec2)
