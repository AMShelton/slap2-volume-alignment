"""Rigid registration utilities for sparse structural reference images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce
from skimage.registration import phase_cross_correlation


@dataclass(frozen=True)
class ShiftResult:
    """Rigid shift estimate for one moving image relative to a template.

    ``shift_yx`` is the (y, x) shift, in full-resolution pixels, to apply to
    the moving image so that it aligns to the template.
    """

    shift_yx: tuple[float, float]
    error: float
    phase_difference: float
    accepted: bool
    reason: str = ""


def as_float_image(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 without modifying shape."""

    if image.dtype == np.float32:
        return image
    return image.astype(np.float32, copy=False)


def robust_rescale_for_registration(
    image: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
) -> np.ndarray:
    """Clip extreme intensities and subtract a robust center for registration."""

    img = as_float_image(image)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)

    lo, hi = np.percentile(img[finite], [low_percentile, high_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = img - np.nanmedian(img)
        return np.nan_to_num(out, copy=False).astype(np.float32)

    out = np.clip(img, lo, hi)
    out = out - np.median(out[finite])
    return np.nan_to_num(out, copy=False).astype(np.float32)


def highpass_for_registration(
    image: np.ndarray,
    *,
    sigma_px: float = 8.0,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
) -> np.ndarray:
    """Prepare an image for phase-correlation registration.

    The default is tuned for sparse two-photon structural/reference images:
    percentile clipping reduces domination by a few bright pixels, while a broad
    Gaussian subtraction removes slow illumination gradients.
    """

    img = robust_rescale_for_registration(
        image,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    if sigma_px and sigma_px > 0:
        img = img - ndimage.gaussian_filter(img, sigma=sigma_px)
    return np.nan_to_num(img, copy=False).astype(np.float32)


def bin_image(image: np.ndarray, binning: int) -> np.ndarray:
    """Mean-bin a 2-D image by an integer factor."""

    if binning <= 1:
        return image
    y = (image.shape[0] // binning) * binning
    x = (image.shape[1] // binning) * binning
    cropped = image[:y, :x]
    return block_reduce(cropped, block_size=(binning, binning), func=np.mean).astype(
        np.float32, copy=False
    )


def estimate_rigid_shift(
    template: np.ndarray,
    moving: np.ndarray,
    *,
    upsample_factor: int = 10,
    max_shift_px: float = 100.0,
    binning: int = 2,
    highpass_sigma_px: float = 8.0,
    normalization: str | None = "phase",
) -> ShiftResult:
    """Estimate full-resolution y/x shift to align moving image to template."""

    # Scale the high-pass sigma into binned-pixel units.
    sigma_binned = max(highpass_sigma_px / max(binning, 1), 0.0)

    fixed = highpass_for_registration(
        bin_image(template, binning), sigma_px=sigma_binned
    )
    mov = highpass_for_registration(bin_image(moving, binning), sigma_px=sigma_binned)

    try:
        shift_binned, error, phase_difference = phase_cross_correlation(
            fixed,
            mov,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
    except Exception as exc:  # pragma: no cover - defensive for bad frames
        return ShiftResult((0.0, 0.0), float("nan"), float("nan"), False, str(exc))

    shift_full = np.asarray(shift_binned, dtype=float) * max(binning, 1)
    shift_norm = float(np.linalg.norm(shift_full))
    if not np.all(np.isfinite(shift_full)):
        return ShiftResult((0.0, 0.0), float(error), float(phase_difference), False, "nonfinite_shift")
    if shift_norm > max_shift_px:
        return ShiftResult(
            (0.0, 0.0),
            float(error),
            float(phase_difference),
            False,
            f"shift_exceeds_max:{shift_norm:.2f}>{max_shift_px}",
        )

    return ShiftResult(
        (float(shift_full[0]), float(shift_full[1])),
        float(error),
        float(phase_difference),
        True,
        "",
    )


def apply_rigid_shift(
    image: np.ndarray,
    shift_yx: tuple[float, float],
    *,
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    """Apply a y/x translation to a 2-D image and return float32."""

    return ndimage.shift(
        as_float_image(image),
        shift=shift_yx,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=(order > 1),
    ).astype(np.float32, copy=False)


def nanmean_stack(images: list[np.ndarray]) -> np.ndarray:
    """NaN-aware mean of a small list of 2-D float images."""

    if not images:
        raise ValueError("No images provided")
    stack = np.stack(images, axis=0).astype(np.float32, copy=False)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(stack, axis=0)
    return out.astype(np.float32, copy=False)


def make_template(frames: list[np.ndarray], *, method: str = "median") -> np.ndarray:
    """Build a registration template from repeated frames."""

    if not frames:
        raise ValueError("No frames provided")
    stack = np.stack([as_float_image(f) for f in frames], axis=0)
    if method == "median":
        return np.median(stack, axis=0).astype(np.float32, copy=False)
    if method == "mean":
        return np.mean(stack, axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown template method: {method}")


def mean_shifted_frames(
    frames: list[np.ndarray],
    shifts_yx: list[tuple[float, float]],
    *,
    order: int = 1,
    zero_shift_tol_px: float = 0.05,
) -> np.ndarray:
    """Apply shifts and return a NaN-aware mean without stacking all frames.

    This is more memory efficient than materializing a full corrected movie for
    each plane. Frames with near-zero shifts are accumulated directly.
    """

    if len(frames) != len(shifts_yx):
        raise ValueError("frames and shifts_yx must have the same length")
    if not frames:
        raise ValueError("No frames provided")

    sum_img = np.zeros(frames[0].shape, dtype=np.float32)
    count_img = np.zeros(frames[0].shape, dtype=np.float32)

    for frame, shift in zip(frames, shifts_yx):
        if float(np.hypot(shift[0], shift[1])) <= zero_shift_tol_px:
            shifted = as_float_image(frame)
            finite = np.isfinite(shifted)
            sum_img[finite] += shifted[finite]
            count_img[finite] += 1.0
        else:
            shifted = apply_rigid_shift(frame, shift, order=order, cval=np.nan)
            finite = np.isfinite(shifted)
            sum_img[finite] += shifted[finite]
            count_img[finite] += 1.0

    out = np.full(frames[0].shape, np.nan, dtype=np.float32)
    valid = count_img > 0
    out[valid] = sum_img[valid] / count_img[valid]
    return out
