"""Bidirectional resonant-scanning line-phase correction utilities.

These helpers implement alternating-line phase correction for ScanImage-style
bidirectional/resonant scans.  The correction is intentionally applied as an
explicit preprocessing step before rigid registration/averaging so the motion
estimator sees geometrically consistent frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import tifffile
from scipy import ndimage

LineParity = Literal["odd", "even"]
FillMode = Literal["preserve", "nearest", "zero"]


def _selected_rows(line_parity: LineParity) -> slice:
    if line_parity == "odd":
        return slice(1, None, 2)
    if line_parity == "even":
        return slice(0, None, 2)
    raise ValueError(f"line_parity must be 'odd' or 'even', got {line_parity!r}")


def apply_bidirectional_phase_2d(
    image: np.ndarray,
    bidiphase: float,
    *,
    line_parity: LineParity = "odd",
    fill_mode: FillMode = "nearest",
    copy: bool = True,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Shift alternating scan lines along x by an integer or fractional phase.

    Parameters
    ----------
    image:
        Two-dimensional ``Y, X`` image.
    bidiphase:
        X-shift applied to selected rows. Positive values move selected rows to
        the right; negative values move selected rows to the left. Fractional
        values are supported and are often preferable for ScanImage line phase.
        ``0`` returns the input unchanged, optionally copied.
    line_parity:
        Which rows to shift. ``"odd"`` matches the common Suite2p convention,
        but some ScanImage files require shifting even rows or the opposite sign.
    fill_mode:
        How to handle newly exposed edge columns. ``"nearest"`` is recommended
        for registration. ``"preserve"`` preserves the original values in the
        newly exposed edge columns. ``"zero"`` fills them with zero.
    copy:
        If True, do not modify the input array.
    interpolation_order:
        Spline interpolation order for fractional shifts. ``1`` is a good
        default; ``0`` is nearest-neighbor; ``3`` is slower and smoother.

    Returns
    -------
    np.ndarray
        Corrected image with the same shape and dtype as the input.
    """

    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")

    phase = float(bidiphase)
    out = image.copy() if copy else image
    if phase == 0:
        return out

    rows = _selected_rows(line_parity)
    width = out.shape[1]
    abs_phase = abs(phase)
    if abs_phase >= width:
        raise ValueError(f"abs(bidiphase) must be smaller than image width; got {phase}")

    original = image if copy else out.copy()

    # Keep the exact integer implementation for speed and backward compatibility.
    is_integer = float(phase).is_integer()
    if is_integer:
        p = int(phase)
        if p > 0:
            out[rows, p:] = original[rows, :-p]
            if fill_mode == "nearest":
                out[rows, :p] = original[rows, :1]
            elif fill_mode == "zero":
                out[rows, :p] = 0
            elif fill_mode == "preserve":
                pass
            else:
                raise ValueError(f"Unknown fill_mode: {fill_mode}")
        else:
            p = -p
            out[rows, :-p] = original[rows, p:]
            if fill_mode == "nearest":
                out[rows, -p:] = original[rows, -1:]
            elif fill_mode == "zero":
                out[rows, -p:] = 0
            elif fill_mode == "preserve":
                pass
            else:
                raise ValueError(f"Unknown fill_mode: {fill_mode}")
        return out

    # Fractional row shifts.  Use nearest-edge extension during interpolation,
    # then optionally overwrite the exposed edge columns to match fill_mode.
    selected = original[rows].astype(np.float32, copy=False)
    shifted = ndimage.shift(
        selected,
        shift=(0.0, phase),
        order=interpolation_order,
        mode="nearest",
        prefilter=(interpolation_order > 1),
    )
    if np.issubdtype(out.dtype, np.integer):
        info = np.iinfo(out.dtype)
        shifted = np.clip(np.rint(shifted), info.min, info.max).astype(out.dtype)
    else:
        shifted = shifted.astype(out.dtype, copy=False)
    out[rows] = shifted

    edge = int(np.ceil(abs_phase))
    if edge > 0:
        if fill_mode == "preserve":
            if phase > 0:
                out[rows, :edge] = original[rows, :edge]
            else:
                out[rows, -edge:] = original[rows, -edge:]
        elif fill_mode == "zero":
            if phase > 0:
                out[rows, :edge] = 0
            else:
                out[rows, -edge:] = 0
        elif fill_mode == "nearest":
            pass
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

    return out


def apply_bidirectional_phase(
    data: np.ndarray,
    bidiphase: float,
    *,
    line_parity: LineParity = "odd",
    fill_mode: FillMode = "nearest",
    copy: bool = True,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Apply bidirectional line-phase correction to 2-D or stack-like arrays.

    The last two dimensions are interpreted as ``Y, X``. Leading dimensions are
    treated as frame/channel/z dimensions and are processed independently.
    """

    if data.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {data.shape}")

    phase = float(bidiphase)
    out = data.copy() if copy else data
    if phase == 0:
        return out

    if out.ndim == 2:
        return apply_bidirectional_phase_2d(
            out,
            phase,
            line_parity=line_parity,
            fill_mode=fill_mode,
            copy=False,
            interpolation_order=interpolation_order,
        )

    flat = out.reshape((-1, *out.shape[-2:]))
    for idx in range(flat.shape[0]):
        apply_bidirectional_phase_2d(
            flat[idx],
            phase,
            line_parity=line_parity,
            fill_mode=fill_mode,
            copy=False,
            interpolation_order=interpolation_order,
        )
    return out


def _robust_registration_image(
    image: np.ndarray,
    *,
    crop_yx: tuple[int, int] = (96, 96),
    x_stride: int = 1,
    highpass_sigma_px: float = 8.0,
) -> np.ndarray:
    """Prepare an image for odd/even line-phase scoring."""

    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {img.shape}")

    cy, cx = crop_yx
    if cy > 0 and img.shape[0] > 2 * cy:
        img = img[cy:-cy, :]
    if cx > 0 and img.shape[1] > 2 * cx:
        img = img[:, cx:-cx]
    if x_stride > 1:
        img = img[:, ::x_stride]

    finite = np.isfinite(img)
    if finite.any():
        lo, hi = np.percentile(img[finite], [1.0, 99.8])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            img = np.clip(img, lo, hi)

    if highpass_sigma_px and highpass_sigma_px > 0:
        # Preserve y row parity; only x is decimated above.
        sigma = (highpass_sigma_px, highpass_sigma_px / max(x_stride, 1))
        img = img - ndimage.gaussian_filter(img, sigma=sigma)

    # Remove row offsets so the score emphasizes horizontal staggering of
    # anatomy rather than line-to-line intensity gain differences.
    img = img - np.nanmedian(img, axis=1, keepdims=True)
    return np.nan_to_num(img, copy=False).astype(np.float32, copy=False)


def bidirectional_phase_score_2d(
    image: np.ndarray,
    bidiphase: float,
    *,
    line_parity: LineParity = "odd",
    fill_mode: FillMode = "nearest",
    edge_margin_px: int = 16,
) -> float:
    """Return a lower-is-better odd/even neighbor mismatch score.

    The selected rows are phase-shifted and compared with the average of their
    immediate neighboring rows. This metric is deliberately simple and works best
    on raw frames or raw-frame medians before motion correction.
    """

    corrected = apply_bidirectional_phase_2d(
        image,
        bidiphase,
        line_parity=line_parity,
        fill_mode=fill_mode,
        copy=True,
        interpolation_order=1,
    )

    if line_parity == "odd":
        rows = np.arange(1, corrected.shape[0] - 1, 2)
    else:
        # Skip row 0 so every scored row has both adjacent odd neighbors.
        rows = np.arange(2, corrected.shape[0] - 1, 2)

    if rows.size == 0:
        return float("nan")

    selected = corrected[rows]
    neighbors = 0.5 * (corrected[rows - 1] + corrected[rows + 1])
    diff = selected - neighbors
    margin = max(int(edge_margin_px), int(np.ceil(abs(float(bidiphase)))) + 2)
    if diff.shape[1] > 2 * margin:
        selected = selected[:, margin:-margin]
        neighbors = neighbors[:, margin:-margin]
        diff = diff[:, margin:-margin]

    # Sparse structural stacks are dominated by dark background.  Score only the
    # informative pixels so the optimum is not washed out by zeros/noise.
    signal = np.maximum(np.abs(selected), np.abs(neighbors))
    finite = np.isfinite(diff) & np.isfinite(signal)
    if not finite.any():
        return float("nan")
    threshold = np.nanpercentile(signal[finite], 75.0)
    informative = finite & (signal >= threshold)
    if informative.sum() < 100:
        informative = finite
    return float(np.nanmean(np.abs(diff[informative])))


def estimate_bidirectional_phase_2d(
    image: np.ndarray,
    *,
    phase_candidates: Sequence[float] | None = None,
    line_parity_candidates: Sequence[LineParity] = ("odd", "even"),
    crop_yx: tuple[int, int] = (96, 96),
    x_stride: int = 1,
    highpass_sigma_px: float = 8.0,
) -> dict:
    """Estimate the best alternating-line phase for one image.

    Returns a dictionary with ``best_phase``, ``best_line_parity``, the raw score
    at zero phase, and the percent improvement over zero phase. The phase is in
    full-resolution pixels, even when ``x_stride`` is used internally.
    """

    if phase_candidates is None:
        phase_candidates = np.arange(-8.0, 8.0001, 0.5)
    phase_candidates = [float(p) for p in phase_candidates]

    # Internally decimate x for speed.  Candidate phases are full-resolution px,
    # so divide by x_stride before scoring and multiply the winner back.
    prepared = _robust_registration_image(
        image,
        crop_yx=crop_yx,
        x_stride=x_stride,
        highpass_sigma_px=highpass_sigma_px,
    )

    rows = []
    for parity in line_parity_candidates:
        for phase_full in phase_candidates:
            score = bidirectional_phase_score_2d(
                prepared,
                phase_full / max(x_stride, 1),
                line_parity=parity,
                fill_mode="nearest",
            )
            rows.append(
                {
                    "line_parity": parity,
                    "phase": float(phase_full),
                    "score": score,
                }
            )

    valid = [r for r in rows if np.isfinite(r["score"])]
    if not valid:
        raise ValueError("Could not compute any finite bidirectional phase scores")

    best = min(valid, key=lambda r: r["score"])
    zero_scores = [r["score"] for r in valid if r["phase"] == 0 and r["line_parity"] == best["line_parity"]]
    zero_score = zero_scores[0] if zero_scores else float("nan")
    improvement = (
        100.0 * (zero_score - best["score"]) / zero_score
        if np.isfinite(zero_score) and zero_score != 0
        else float("nan")
    )
    return {
        "best_phase": float(best["phase"]),
        "best_line_parity": best["line_parity"],
        "best_score": float(best["score"]),
        "zero_phase_score_same_parity": float(zero_score),
        "improvement_pct": float(improvement),
        "scores": rows,
    }


def estimate_bidirectional_phase_stack(
    frames: Iterable[np.ndarray],
    *,
    phase_candidates: Sequence[float] | None = None,
    line_parity_candidates: Sequence[LineParity] = ("odd", "even"),
    crop_yx: tuple[int, int] = (96, 96),
    x_stride: int = 1,
    highpass_sigma_px: float = 8.0,
) -> dict:
    """Estimate bidirectional phase from several raw frames.

    Scores are aggregated by median across frames for each candidate phase and
    parity. This is the recommended estimator to run on a small sample of raw
    planes/repeats before calling ``average_scanimage_volume``.
    """

    frame_list = [np.asarray(frame) for frame in frames]
    if not frame_list:
        raise ValueError("No frames provided")
    if phase_candidates is None:
        phase_candidates = np.arange(-8.0, 8.0001, 0.5)
    phase_candidates = [float(p) for p in phase_candidates]

    per_frame_scores: list[dict] = []
    for frame_index, frame in enumerate(frame_list):
        estimate = estimate_bidirectional_phase_2d(
            frame,
            phase_candidates=phase_candidates,
            line_parity_candidates=line_parity_candidates,
            crop_yx=crop_yx,
            x_stride=x_stride,
            highpass_sigma_px=highpass_sigma_px,
        )
        for row in estimate["scores"]:
            per_frame_scores.append({"frame_index": frame_index, **row})

    aggregate_rows = []
    for parity in line_parity_candidates:
        for phase in phase_candidates:
            scores = [
                r["score"]
                for r in per_frame_scores
                if r["line_parity"] == parity and r["phase"] == phase and np.isfinite(r["score"])
            ]
            if scores:
                aggregate_rows.append(
                    {
                        "line_parity": parity,
                        "phase": float(phase),
                        "median_score": float(np.median(scores)),
                        "mean_score": float(np.mean(scores)),
                        "n_frames": int(len(scores)),
                    }
                )

    if not aggregate_rows:
        raise ValueError("Could not compute any finite aggregate bidirectional phase scores")

    best = min(aggregate_rows, key=lambda r: r["median_score"])
    zero_scores = [
        r["median_score"]
        for r in aggregate_rows
        if r["phase"] == 0 and r["line_parity"] == best["line_parity"]
    ]
    zero_score = zero_scores[0] if zero_scores else float("nan")
    improvement = (
        100.0 * (zero_score - best["median_score"]) / zero_score
        if np.isfinite(zero_score) and zero_score != 0
        else float("nan")
    )
    return {
        "best_phase": float(best["phase"]),
        "best_line_parity": best["line_parity"],
        "best_median_score": float(best["median_score"]),
        "zero_phase_median_score_same_parity": float(zero_score),
        "improvement_pct": float(improvement),
        "aggregate_scores": aggregate_rows,
        "per_frame_scores": per_frame_scores,
    }


def apply_bidirectional_phase_tiff(
    input_tif: str | Path,
    output_tif: str | Path,
    *,
    bidiphase: float,
    line_parity: LineParity = "odd",
    fill_mode: FillMode = "nearest",
    compression: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Stream a TIFF page-by-page and write a bidirectional-corrected copy.

    This avoids loading a large ScanImage stack into memory. Per-page
    ImageDescription strings are preserved where possible.
    """

    input_tif = Path(input_tif)
    output_tif = Path(output_tif)
    if output_tif.exists() and not overwrite:
        raise FileExistsError(f"Output exists and overwrite=False: {output_tif}")

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(input_tif) as tif, tifffile.TiffWriter(output_tif, bigtiff=True) as writer:
        for page_index, page in enumerate(tif.pages):
            image = page.asarray()
            corrected = apply_bidirectional_phase_2d(
                image,
                bidiphase,
                line_parity=line_parity,
                fill_mode=fill_mode,
                copy=True,
            )
            writer.write(
                corrected,
                photometric="minisblack",
                compression=compression,
                description=page.description,
                metadata=None,
            )
            if (page_index + 1) % 100 == 0:
                print(f"bidi corrected {page_index + 1}/{len(tif.pages)} pages")

        n_pages = len(tif.pages)

    return {
        "input_tif": str(input_tif),
        "output_tif": str(output_tif),
        "bidiphase": float(bidiphase),
        "line_parity": line_parity,
        "fill_mode": fill_mode,
        "compression": compression,
        "n_pages": n_pages,
    }
