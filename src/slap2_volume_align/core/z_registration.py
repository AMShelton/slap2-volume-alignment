"""Crop-based inter-plane registration for ScanImage z volumes.

This module replaces the older global full-frame z-alignment path. The core
assumption is deliberately narrower and more realistic for sparse neuronal
structural stacks: use one or more informative local crops/landmarks to estimate
one rigid xy translation per z-plane, then apply the consensus shift to the full
plane. The implementation is inspired by the local normalized cross-correlation
style used in the SLAP2 preprocessing MATLAB code (`xcorr2_nans*` and
`multiRoiRegSLAP2.m`), while remaining pure Python and TIFF-friendly.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
from scipy import ndimage

from slap2_volume_align.core.registration import (
    ShiftResult,
    apply_rigid_shift,
    highpass_for_registration,
)
from slap2_volume_align.readers.tiff import write_volume_tiff

CropYX = tuple[int, int, int, int]


@dataclass(frozen=True)
class LocalNCCResult:
    """Local normalized cross-correlation shift result.

    ``shift_yx`` is the y/x shift, in full-resolution pixels, to apply to the
    moving image/crop so it aligns with the fixed template/crop.
    """

    shift_yx: tuple[float, float]
    corr: float
    accepted: bool
    reason: str = ""
    peak_at_edge: bool = False
    overlap_fraction: float = 0.0


def normalize_crop_yx(crop: Sequence[int], image_shape: tuple[int, int]) -> CropYX:
    """Clamp/validate one crop tuple ``(y0, y1, x0, x1)``."""

    if len(crop) != 4:
        raise ValueError(f"Crop must contain four integers (y0, y1, x0, x1), got {crop!r}")
    y0, y1, x0, x1 = [int(v) for v in crop]
    h, w = [int(v) for v in image_shape]
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Empty crop after clamping to image shape {image_shape}: {crop!r}")
    return y0, y1, x0, x1


def parse_crop_yx(value: str | Sequence[int] | None, image_shape: tuple[int, int]) -> CropYX | None:
    """Parse one crop from string/list input.

    Accepted string forms are ``"y0:y1,x0:x1"`` and ``"y0,y1,x0,x1"``.
    """

    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" in text:
            try:
                ypart, xpart = text.split(",")
                y0, y1 = [int(v) for v in ypart.split(":")]
                x0, x1 = [int(v) for v in xpart.split(":")]
            except Exception as exc:  # pragma: no cover - user input guard
                raise ValueError(
                    f"Could not parse crop {value!r}; expected 'y0:y1,x0:x1'"
                ) from exc
            return normalize_crop_yx((y0, y1, x0, x1), image_shape)
        parts = [int(v.strip()) for v in text.split(",") if v.strip()]
        return normalize_crop_yx(parts, image_shape)
    return normalize_crop_yx(value, image_shape)


def parse_crop_list_yx(
    crops: str | Sequence[Sequence[int]] | None,
    image_shape: tuple[int, int],
) -> list[CropYX] | None:
    """Parse a crop list from CLI/notebook-friendly forms.

    ``None``/empty means "auto-crops". A string can contain semicolon-delimited
    crops, e.g. ``"450:850,600:1000;900:1250,300:700"``.
    """

    if crops is None or crops == "":
        return None
    if isinstance(crops, str):
        out = []
        for chunk in crops.split(";"):
            parsed = parse_crop_yx(chunk, image_shape)
            if parsed is not None:
                out.append(parsed)
        return out or None
    return [normalize_crop_yx(crop, image_shape) for crop in crops]


def _safe_percentile_limits(img: np.ndarray, pcts: tuple[float, float] = (1.0, 99.8)) -> tuple[float, float]:
    finite = np.isfinite(img)
    if not finite.any():
        return 0.0, 1.0
    lo, hi = np.percentile(img[finite], pcts)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def transform_for_z_registration(
    image: np.ndarray,
    *,
    intensity_transform: str = "sqrt",
    highpass_sigma_px: float = 8.0,
) -> np.ndarray:
    """Prepare one image/crop for z registration.

    ``sqrt`` mirrors the stabilizing transform used in the SLAP2 registration
    code for sparse positive fluorescence images. ``log1p`` is an alternative
    when very bright puncta dominate. A broad high-pass improves robustness to
    illumination gradients.
    """

    img = np.asarray(image, dtype=np.float32)
    lo, hi = _safe_percentile_limits(img)
    x = np.clip(img, lo, hi) - lo
    mode = str(intensity_transform).lower()
    if mode == "sqrt":
        x = np.sqrt(np.maximum(x, 0.0))
    elif mode == "log1p":
        x = np.log1p(np.maximum(x, 0.0))
    elif mode in ("none", "linear", "identity"):
        pass
    else:
        raise ValueError(f"Unknown intensity_transform: {intensity_transform!r}")
    return highpass_for_registration(x, sigma_px=highpass_sigma_px)


def _crop_image(image: np.ndarray, crop: CropYX) -> np.ndarray:
    y0, y1, x0, x1 = crop
    return np.asarray(image[y0:y1, x0:x1])


def _bin_for_ncc(image: np.ndarray, binning: int) -> np.ndarray:
    if binning <= 1:
        return np.asarray(image, dtype=np.float32)
    h = (image.shape[0] // binning) * binning
    w = (image.shape[1] // binning) * binning
    if h <= 0 or w <= 0:
        raise ValueError(f"Crop {image.shape} is too small for binning={binning}")
    cropped = image[:h, :w].astype(np.float32, copy=False)
    reshaped = cropped.reshape(h // binning, binning, w // binning, binning)
    with np.errstate(invalid="ignore"):
        return np.nanmean(reshaped, axis=(1, 3)).astype(np.float32, copy=False)


def _overlap_slices(shape: tuple[int, int], dy: int, dx: int):
    h, w = shape
    y_fixed0 = max(0, dy)
    y_moving0 = max(0, -dy)
    y_n = h - abs(dy)
    x_fixed0 = max(0, dx)
    x_moving0 = max(0, -dx)
    x_n = w - abs(dx)
    if y_n <= 0 or x_n <= 0:
        return None
    return (
        slice(y_fixed0, y_fixed0 + y_n),
        slice(x_fixed0, x_fixed0 + x_n),
        slice(y_moving0, y_moving0 + y_n),
        slice(x_moving0, x_moving0 + x_n),
    )


def _pearson_for_shift(fixed: np.ndarray, moving: np.ndarray, dy: int, dx: int, min_overlap: int) -> tuple[float, int]:
    slices = _overlap_slices(fixed.shape, dy, dx)
    if slices is None:
        return float("nan"), 0
    yf, xf, ym, xm = slices
    a = fixed[yf, xf]
    b = moving[ym, xm]
    valid = np.isfinite(a) & np.isfinite(b)
    n = int(valid.sum())
    if n < min_overlap:
        return float("nan"), n
    av = a[valid].astype(np.float64, copy=False)
    bv = b[valid].astype(np.float64, copy=False)
    av = av - av.mean()
    bv = bv - bv.mean()
    den = np.sqrt(np.sum(av * av) * np.sum(bv * bv))
    if den <= 0 or not np.isfinite(den):
        return float("nan"), n
    return float(np.sum(av * bv) / den), n


def _subpixel_peak_offset(c_prev: float, c0: float, c_next: float) -> float:
    """Parabolic peak offset in [-1, 1] from three correlation values."""

    if not (np.isfinite(c_prev) and np.isfinite(c0) and np.isfinite(c_next)):
        return 0.0
    denom = c_prev - 2.0 * c0 + c_next
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (c_prev - c_next) / denom
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -1.0, 1.0))


def estimate_local_ncc_shift(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    max_shift_px: int = 20,
    min_corr: float = 0.08,
    min_overlap_fraction: float = 0.25,
    binning: int = 1,
) -> LocalNCCResult:
    """Estimate a local rigid shift using masked normalized cross-correlation.

    This searches integer shifts in ``[-max_shift_px, +max_shift_px]`` in binned
    pixels, then does a one-pixel parabolic refinement around the peak. It is
    slower than FFT phase correlation but much more explicit, crop-local, and
    robust to NaNs/edges for the small crops used in z-plane registration.
    """

    if fixed.shape != moving.shape:
        raise ValueError(f"fixed and moving crops must have the same shape, got {fixed.shape} and {moving.shape}")
    if max_shift_px < 0:
        raise ValueError("max_shift_px must be nonnegative")

    fixed_b = _bin_for_ncc(fixed, binning)
    moving_b = _bin_for_ncc(moving, binning)
    d = int(round(max_shift_px / max(binning, 1)))
    d = max(0, d)
    min_overlap = max(8, int(np.prod(fixed_b.shape) * float(min_overlap_fraction)))
    shifts = np.arange(-d, d + 1, dtype=int)
    corr = np.full((len(shifts), len(shifts)), np.nan, dtype=np.float32)
    overlaps = np.zeros_like(corr, dtype=np.int32)

    for iy, dy in enumerate(shifts):
        for ix, dx in enumerate(shifts):
            corr[iy, ix], overlaps[iy, ix] = _pearson_for_shift(fixed_b, moving_b, int(dy), int(dx), min_overlap)

    if not np.isfinite(corr).any():
        return LocalNCCResult((0.0, 0.0), float("nan"), False, "no_finite_correlations", False, 0.0)

    peak_flat = int(np.nanargmax(corr))
    iy, ix = np.unravel_index(peak_flat, corr.shape)
    peak_corr = float(corr[iy, ix])
    peak_at_edge = bool(iy == 0 or ix == 0 or iy == corr.shape[0] - 1 or ix == corr.shape[1] - 1)
    dy = float(shifts[iy])
    dx = float(shifts[ix])
    if not peak_at_edge:
        dy += _subpixel_peak_offset(float(corr[iy - 1, ix]), peak_corr, float(corr[iy + 1, ix]))
        dx += _subpixel_peak_offset(float(corr[iy, ix - 1]), peak_corr, float(corr[iy, ix + 1]))

    overlap_fraction = float(overlaps[iy, ix] / max(1, np.prod(fixed_b.shape)))
    shift_full = (dy * max(binning, 1), dx * max(binning, 1))

    if peak_at_edge:
        return LocalNCCResult(shift_full, peak_corr, False, "peak_at_search_edge", True, overlap_fraction)
    if not np.isfinite(peak_corr) or peak_corr < min_corr:
        return LocalNCCResult(shift_full, peak_corr, False, f"corr_below_min:{peak_corr:.3f}<{min_corr}", False, overlap_fraction)
    return LocalNCCResult(shift_full, peak_corr, True, "", False, overlap_fraction)


def infer_registration_crops(
    volume_zyx: np.ndarray,
    *,
    n_crops: int = 1,
    crop_size_px: int = 384,
    min_distance_px: int | None = None,
    percentile: float = 99.7,
    border_px: int | None = None,
) -> list[CropYX]:
    """Infer bright, spatially separated crops from a z max-projection.

    Manual crops are still preferred for final data products. This auto mode is
    intended to make smoke tests and first-pass notebook runs easy.
    """

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX volume, got shape {volume.shape}")
    n_crops = int(n_crops)
    if n_crops <= 0:
        raise ValueError("n_crops must be positive")
    h, w = volume.shape[1:]
    crop_size_px = int(crop_size_px)
    half = max(8, crop_size_px // 2)
    if border_px is None:
        border_px = half
    if min_distance_px is None:
        min_distance_px = crop_size_px

    proj = np.nanmax(volume.astype(np.float32, copy=False), axis=0)
    lo, hi = _safe_percentile_limits(proj, (1.0, percentile))
    score = np.clip(proj, lo, hi) - lo
    score = ndimage.gaussian_filter(score, sigma=max(2, crop_size_px / 64))
    score[:border_px, :] = -np.inf
    score[-border_px:, :] = -np.inf
    score[:, :border_px] = -np.inf
    score[:, -border_px:] = -np.inf

    crops: list[CropYX] = []
    centers: list[tuple[int, int]] = []
    work = score.copy()
    for _ in range(n_crops):
        if not np.isfinite(work).any():
            break
        flat = int(np.nanargmax(work))
        cy, cx = np.unravel_index(flat, work.shape)
        if not np.isfinite(work[cy, cx]):
            break
        crop = normalize_crop_yx((cy - half, cy + half, cx - half, cx + half), (h, w))
        crops.append(crop)
        centers.append((int(cy), int(cx)))
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= int(min_distance_px) ** 2
        work[mask] = -np.inf

    if not crops:
        # Conservative central crop fallback.
        crops = [normalize_crop_yx((h // 2 - half, h // 2 + half, w // 2 - half, w // 2 + half), (h, w))]
    return crops


def choose_anchor_z_from_crops(volume_zyx: np.ndarray, crops_yx: Sequence[CropYX], *, margin: int = 3) -> int:
    """Choose a high-contrast interior plane within the registration crops."""

    volume = np.asarray(volume_zyx)
    n_z = volume.shape[0]
    scores = np.zeros(n_z, dtype=float)
    for z in range(n_z):
        vals = []
        for crop in crops_yx:
            img = _crop_image(volume[z], crop)
            finite = np.isfinite(img)
            if finite.any():
                vals.append(float(np.percentile(img[finite], 99.5) - np.percentile(img[finite], 5.0)))
        scores[z] = float(np.nanmean(vals)) if vals else 0.0
    if n_z > 2 * margin:
        idxs = np.arange(margin, n_z - margin)
        return int(idxs[np.nanargmax(scores[idxs])])
    return int(np.nanargmax(scores))


def _weighted_average_shift(shifts: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    shifts = np.asarray(shifts, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        return (float(np.nanmean(shifts[:, 0])), float(np.nanmean(shifts[:, 1])))
    avg = np.sum(shifts * weights[:, None], axis=0) / weights.sum()
    return float(avg[0]), float(avg[1])


def _smooth_short_gaps(values: np.ndarray, *, window: int = 0, max_gap: int = 1, anchor_idx: int | None = None) -> np.ndarray:
    """Optionally interpolate short finite gaps and median-smooth contiguous runs."""

    values = np.asarray(values, dtype=float)
    out = values.copy()
    finite = np.isfinite(out)
    if finite.sum() >= 2 and max_gap >= 1:
        good = np.flatnonzero(finite)
        x = np.arange(out.size)
        interp = np.interp(x, good, out[good])
        for left, right in zip(good[:-1], good[1:]):
            gap = int(right - left - 1)
            if 0 < gap <= max_gap:
                out[left + 1 : right] = interp[left + 1 : right]
    if window and window >= 3:
        if window % 2 == 0:
            window += 1
        finite = np.isfinite(out)
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
        for start, stop in zip(edges[::2], edges[1::2]):
            run_len = stop - start
            if run_len >= window:
                out[start:stop] = ndimage.median_filter(out[start:stop], size=window, mode="nearest")
    if anchor_idx is not None and 0 <= anchor_idx < len(out) and np.isfinite(out[anchor_idx]):
        out = out - out[anchor_idx]
    return out


def _estimate_single_crop_walk(
    volume: np.ndarray,
    crop: CropYX,
    *,
    anchor_idx: int,
    max_shift_px: int,
    binning: int,
    highpass_sigma_px: float,
    intensity_transform: str,
    min_corr: float,
    min_overlap_fraction: float,
    interpolation_order: int,
    template_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    n_z = volume.shape[0]
    raw_y = np.full(n_z, np.nan, dtype=float)
    raw_x = np.full(n_z, np.nan, dtype=float)
    corr = np.full(n_z, np.nan, dtype=float)
    accepted = np.zeros(n_z, dtype=bool)
    reasons = [""] * n_z
    aligned_crops: list[np.ndarray | None] = [None] * n_z

    anchor_crop = transform_for_z_registration(
        _crop_image(volume[anchor_idx], crop),
        intensity_transform=intensity_transform,
        highpass_sigma_px=highpass_sigma_px,
    )
    aligned_crops[anchor_idx] = anchor_crop
    raw_y[anchor_idx] = 0.0
    raw_x[anchor_idx] = 0.0
    corr[anchor_idx] = 1.0
    accepted[anchor_idx] = True
    reasons[anchor_idx] = "anchor"

    def local_template(indices: Iterable[int]) -> np.ndarray | None:
        imgs = [aligned_crops[i] for i in indices if 0 <= i < n_z and aligned_crops[i] is not None]
        if not imgs:
            return None
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(np.stack(imgs, axis=0), axis=0).astype(np.float32, copy=False)

    def walk(z_iter: Iterable[int], neighbor_fn) -> None:
        for z in z_iter:
            template = local_template(neighbor_fn(z))
            if template is None:
                reasons[z] = "no_local_template"
                continue
            moving = transform_for_z_registration(
                _crop_image(volume[z], crop),
                intensity_transform=intensity_transform,
                highpass_sigma_px=highpass_sigma_px,
            )
            result = estimate_local_ncc_shift(
                template,
                moving,
                max_shift_px=max_shift_px,
                min_corr=min_corr,
                min_overlap_fraction=min_overlap_fraction,
                binning=binning,
            )
            raw_y[z], raw_x[z] = result.shift_yx
            corr[z] = result.corr
            accepted[z] = bool(result.accepted)
            reasons[z] = result.reason
            if result.accepted:
                aligned_crops[z] = apply_rigid_shift(
                    moving,
                    result.shift_yx,
                    order=interpolation_order,
                    cval=np.nan,
                )

    template_radius = max(1, int(template_radius))
    walk(
        range(anchor_idx + 1, n_z),
        lambda z: range(max(anchor_idx, z - template_radius), z),
    )
    walk(
        range(anchor_idx - 1, -1, -1),
        lambda z: range(z + 1, min(anchor_idx + 1, z + 1 + template_radius)),
    )
    return raw_y, raw_x, corr, reasons, accepted


def estimate_crop_z_registration(
    volume_zyx: np.ndarray,
    *,
    z_indices: Iterable[int] | None = None,
    crops_yx: Sequence[CropYX] | None = None,
    anchor_z: int | None = None,
    auto_n_crops: int = 1,
    auto_crop_size_px: int = 384,
    template_radius: int = 2,
    max_shift_px: int = 20,
    binning: int = 4,
    highpass_sigma_px: float = 8.0,
    intensity_transform: str = "sqrt",
    min_corr: float = 0.08,
    min_overlap_fraction: float = 0.25,
    min_accepted_crops: int = 1,
    smooth_window: int = 0,
    max_interpolation_gap: int = 1,
    interpolation_order: int = 1,
) -> tuple[list[dict], list[dict], list[CropYX]]:
    """Estimate crop-based rigid xy shifts for each z-plane in a volume.

    Returns ``(plane_rows, crop_rows, crops_yx)``. Plane rows contain the final
    consensus shift to apply to the full-resolution plane. Crop rows contain the
    per-crop evidence used to produce that consensus.
    """

    volume = np.asarray(volume_zyx, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX volume, got shape {volume.shape}")
    n_z = volume.shape[0]
    original_z = list(range(n_z)) if z_indices is None else list(z_indices)
    if len(original_z) != n_z:
        raise ValueError("z_indices must have one entry per z-plane")

    if crops_yx is None or len(crops_yx) == 0:
        crops = infer_registration_crops(
            volume,
            n_crops=auto_n_crops,
            crop_size_px=auto_crop_size_px,
        )
    else:
        crops = [normalize_crop_yx(c, volume.shape[1:]) for c in crops_yx]

    if anchor_z is None:
        anchor_idx = choose_anchor_z_from_crops(volume, crops)
    else:
        if 0 <= int(anchor_z) < n_z:
            anchor_idx = int(anchor_z)
        elif int(anchor_z) in original_z:
            anchor_idx = int(original_z.index(int(anchor_z)))
        else:
            raise ValueError(f"anchor_z={anchor_z} is neither a local nor original z index")

    all_y = []
    all_x = []
    all_corr = []
    all_acc = []
    per_crop_reasons: list[list[str]] = []

    for crop in crops:
        raw_y, raw_x, corr, reasons, accepted = _estimate_single_crop_walk(
            volume,
            crop,
            anchor_idx=anchor_idx,
            max_shift_px=max_shift_px,
            binning=binning,
            highpass_sigma_px=highpass_sigma_px,
            intensity_transform=intensity_transform,
            min_corr=min_corr,
            min_overlap_fraction=min_overlap_fraction,
            interpolation_order=interpolation_order,
            template_radius=template_radius,
        )
        all_y.append(raw_y)
        all_x.append(raw_x)
        all_corr.append(corr)
        all_acc.append(accepted)
        per_crop_reasons.append(reasons)

    all_y_arr = np.vstack(all_y)
    all_x_arr = np.vstack(all_x)
    all_corr_arr = np.vstack(all_corr)
    all_acc_arr = np.vstack(all_acc)

    final_y = np.full(n_z, np.nan, dtype=float)
    final_x = np.full(n_z, np.nan, dtype=float)
    n_accepted = np.zeros(n_z, dtype=int)
    mean_corr = np.full(n_z, np.nan, dtype=float)

    for z in range(n_z):
        ok = all_acc_arr[:, z] & np.isfinite(all_y_arr[:, z]) & np.isfinite(all_x_arr[:, z])
        n_accepted[z] = int(ok.sum())
        if n_accepted[z] >= int(min_accepted_crops):
            weights = np.maximum(all_corr_arr[ok, z], 0.0) ** 2
            final_y[z], final_x[z] = _weighted_average_shift(
                np.column_stack([all_y_arr[ok, z], all_x_arr[ok, z]]), weights
            )
            mean_corr[z] = float(np.nanmean(all_corr_arr[ok, z]))

    # Ensure the anchor is exactly zero and optionally fill only short gaps.
    final_y[anchor_idx] = 0.0
    final_x[anchor_idx] = 0.0
    final_y = _smooth_short_gaps(final_y, window=smooth_window, max_gap=max_interpolation_gap, anchor_idx=anchor_idx)
    final_x = _smooth_short_gaps(final_x, window=smooth_window, max_gap=max_interpolation_gap, anchor_idx=anchor_idx)

    plane_rows: list[dict] = []
    for z in range(n_z):
        accepted_plane = bool(np.isfinite(final_y[z]) and np.isfinite(final_x[z]))
        plane_rows.append(
            {
                "volume_index": int(z),
                "z_index": int(original_z[z]),
                "anchor_volume_index": int(anchor_idx),
                "anchor_z_index": int(original_z[anchor_idx]),
                "shift_y_px": float(final_y[z]) if np.isfinite(final_y[z]) else float("nan"),
                "shift_x_px": float(final_x[z]) if np.isfinite(final_x[z]) else float("nan"),
                "accepted": accepted_plane,
                "n_accepted_crops": int(n_accepted[z]),
                "mean_corr": float(mean_corr[z]) if np.isfinite(mean_corr[z]) else "",
                "reason": "" if accepted_plane else "insufficient_crop_support",
            }
        )

    crop_rows: list[dict] = []
    for crop_idx, crop in enumerate(crops):
        y0, y1, x0, x1 = crop
        for z in range(n_z):
            crop_rows.append(
                {
                    "crop_index": int(crop_idx),
                    "crop_y0": int(y0),
                    "crop_y1": int(y1),
                    "crop_x0": int(x0),
                    "crop_x1": int(x1),
                    "volume_index": int(z),
                    "z_index": int(original_z[z]),
                    "anchor_volume_index": int(anchor_idx),
                    "anchor_z_index": int(original_z[anchor_idx]),
                    "shift_y_px": float(all_y_arr[crop_idx, z]) if np.isfinite(all_y_arr[crop_idx, z]) else float("nan"),
                    "shift_x_px": float(all_x_arr[crop_idx, z]) if np.isfinite(all_x_arr[crop_idx, z]) else float("nan"),
                    "corr": float(all_corr_arr[crop_idx, z]) if np.isfinite(all_corr_arr[crop_idx, z]) else "",
                    "accepted": bool(all_acc_arr[crop_idx, z]),
                    "reason": per_crop_reasons[crop_idx][z],
                }
            )
    return plane_rows, crop_rows, crops


def apply_z_registration(
    volume_zyx: np.ndarray,
    plane_rows: list[dict],
    *,
    interpolation_order: int = 1,
    fill_mode: str = "nearest",
) -> np.ndarray:
    """Apply crop-based z registration shifts to a ZYX volume.

    Planes with non-finite shifts are left unshifted; this is an intentional
    safety behavior to avoid extrapolated, unsupported plane motion.
    """

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX volume, got shape {volume.shape}")
    if len(plane_rows) != volume.shape[0]:
        raise ValueError("plane_rows must have one row per z-plane")
    rows = sorted(plane_rows, key=lambda r: int(r["volume_index"]))
    out = np.empty(volume.shape, dtype=np.float32)
    mode = "nearest" if fill_mode == "nearest" else "constant"
    cval = np.nan if fill_mode == "nan" else 0.0
    for row in rows:
        z = int(row["volume_index"])
        dy = _safe_float(row.get("shift_y_px"))
        dx = _safe_float(row.get("shift_x_px"))
        if not np.isfinite(dy) or not np.isfinite(dx) or abs(dy) + abs(dx) <= 1e-6:
            out[z] = volume[z].astype(np.float32, copy=False)
        else:
            out[z] = ndimage.shift(
                volume[z].astype(np.float32, copy=False),
                shift=(dy, dx),
                order=interpolation_order,
                mode=mode,
                cval=cval,
                prefilter=(interpolation_order > 1),
            ).astype(np.float32, copy=False)
    return out


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_z_registration_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_z_registration_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def make_z_registration_qc_png(
    path: str | Path,
    plane_rows: list[dict],
    crop_rows: list[dict],
    crops_yx: Sequence[CropYX],
    *,
    projection_image: np.ndarray | None = None,
) -> None:
    """Write a compact QC plot for crop-based z registration."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not plane_rows:
        return

    z = np.array([int(r["z_index"]) for r in plane_rows])
    y = np.array([_safe_float(r.get("shift_y_px")) for r in plane_rows])
    x = np.array([_safe_float(r.get("shift_x_px")) for r in plane_rows])
    nacc = np.array([int(r.get("n_accepted_crops", 0)) for r in plane_rows])
    mean_corr = np.array([_safe_float(r.get("mean_corr")) for r in plane_rows])
    accepted = np.array([str(r.get("accepted", False)).lower() == "true" for r in plane_rows])
    anchor_z = int(plane_rows[0]["anchor_z_index"])

    fig, axes = plt.subplots(4 if projection_image is not None else 3, 1, figsize=(11, 10), sharex=False)
    if projection_image is not None:
        ax0 = axes[0]
        img = projection_image
        lo, hi = _safe_percentile_limits(img)
        ax0.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        for i, crop in enumerate(crops_yx):
            y0, y1, x0, x1 = crop
            ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=1.5))
            ax0.text(x0, y0, f"crop {i}", color="white", fontsize=8, va="bottom")
        ax0.set_title("Registration crops on max projection")
        ax0.axis("off")
        plot_axes = axes[1:]
    else:
        plot_axes = axes

    ax = plot_axes[0]
    ax.plot(z[accepted], y[accepted], ".", label="y accepted")
    ax.plot(z[~accepted], y[~accepted], "x", label="y unsupported")
    ax.plot(z, y, "-", alpha=0.5)
    ax.set_ylabel("Y shift (px)")
    ax.legend()

    ax = plot_axes[1]
    ax.plot(z[accepted], x[accepted], ".", label="x accepted")
    ax.plot(z[~accepted], x[~accepted], "x", label="x unsupported")
    ax.plot(z, x, "-", alpha=0.5)
    ax.set_ylabel("X shift (px)")
    ax.legend()

    ax = plot_axes[2]
    ax.plot(z, nacc, ".-", label="accepted crops")
    ax2 = ax.twinx()
    ax2.plot(z, mean_corr, ".-", alpha=0.6, label="mean corr")
    ax.set_ylabel("Accepted crops")
    ax2.set_ylabel("Mean corr")
    ax.set_xlabel("z index")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    for ax in plot_axes:
        ax.axvline(anchor_z, color="k", linestyle="--", alpha=0.4)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def register_volume_z_tiff(
    input_tif: str | Path,
    output_tif: str | Path | None = None,
    *,
    crops_yx: Sequence[CropYX] | None = None,
    crop_string: str | None = None,
    anchor_z: int | None = None,
    auto_n_crops: int = 1,
    auto_crop_size_px: int = 384,
    template_radius: int = 2,
    max_shift_px: int = 20,
    binning: int = 4,
    highpass_sigma_px: float = 8.0,
    intensity_transform: str = "sqrt",
    min_corr: float = 0.08,
    min_overlap_fraction: float = 0.25,
    min_accepted_crops: int = 1,
    smooth_window: int = 0,
    max_interpolation_gap: int = 1,
    interpolation_order: int = 1,
    output_dtype: str = "float32",
    output_compression: str | None = None,
    shifts_csv: str | Path | None = None,
    crop_shifts_csv: str | Path | None = None,
    qc_png: str | Path | None = None,
    summary_json: str | Path | None = None,
) -> dict:
    """Posthoc crop-based z registration for an averaged ZYX TIFF."""

    input_tif = Path(input_tif)
    if output_tif is None:
        output_tif = input_tif.with_name(f"{input_tif.stem}_zregistered.tif")
    output_tif = Path(output_tif)
    volume = tifffile.imread(input_tif).astype(np.float32, copy=False)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX TIFF volume, got shape {volume.shape}")
    if crop_string:
        crops_yx = parse_crop_list_yx(crop_string, volume.shape[1:])

    plane_rows, crop_rows, crops = estimate_crop_z_registration(
        volume,
        crops_yx=crops_yx,
        anchor_z=anchor_z,
        auto_n_crops=auto_n_crops,
        auto_crop_size_px=auto_crop_size_px,
        template_radius=template_radius,
        max_shift_px=max_shift_px,
        binning=binning,
        highpass_sigma_px=highpass_sigma_px,
        intensity_transform=intensity_transform,
        min_corr=min_corr,
        min_overlap_fraction=min_overlap_fraction,
        min_accepted_crops=min_accepted_crops,
        smooth_window=smooth_window,
        max_interpolation_gap=max_interpolation_gap,
        interpolation_order=interpolation_order,
    )
    corrected = apply_z_registration(volume, plane_rows, interpolation_order=interpolation_order)
    write_volume_tiff(
        output_tif,
        corrected,
        dtype=output_dtype,
        compression=output_compression,
        description=f"Crop-based z-registered volume from {input_tif.name}; axes=ZYX",
    )
    if shifts_csv is None:
        shifts_csv = output_tif.with_name(f"{input_tif.stem}_z_registration_shifts.csv")
    if crop_shifts_csv is None:
        crop_shifts_csv = output_tif.with_name(f"{input_tif.stem}_z_registration_crop_shifts.csv")
    write_z_registration_csv(shifts_csv, plane_rows)
    write_z_registration_csv(crop_shifts_csv, crop_rows)

    if qc_png is None:
        qc_png = output_tif.with_name(f"{input_tif.stem}_z_registration_qc.png")
    projection = np.nanmax(volume, axis=0)
    make_z_registration_qc_png(qc_png, plane_rows, crop_rows, crops, projection_image=projection)

    summary = {
        "input_tif": str(input_tif),
        "output_tif": str(output_tif),
        "shifts_csv": str(shifts_csv),
        "crop_shifts_csv": str(crop_shifts_csv),
        "qc_png": str(qc_png),
        "crops_yx": [list(c) for c in crops],
        "config": {
            "anchor_z": anchor_z,
            "auto_n_crops": auto_n_crops,
            "auto_crop_size_px": auto_crop_size_px,
            "template_radius": template_radius,
            "max_shift_px": max_shift_px,
            "binning": binning,
            "highpass_sigma_px": highpass_sigma_px,
            "intensity_transform": intensity_transform,
            "min_corr": min_corr,
            "min_overlap_fraction": min_overlap_fraction,
            "min_accepted_crops": min_accepted_crops,
            "smooth_window": smooth_window,
            "max_interpolation_gap": max_interpolation_gap,
            "interpolation_order": interpolation_order,
            "output_dtype": output_dtype,
            "output_compression": output_compression,
        },
    }
    if summary_json is None:
        summary_json = output_tif.with_name(f"{input_tif.stem}_z_registration_summary.json")
    Path(summary_json).write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(summary_json)
    return summary
