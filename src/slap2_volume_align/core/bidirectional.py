"""Bidirectional resonant-scanning line-phase correction utilities.

These helpers implement the same basic odd-line integer pixel correction used in
many Suite2p preprocessing recipes, but in reusable, non-destructive functions.
They are intended as an optional preprocessing step before rigid registration and
averaging when bidirectional line phase is visible in raw ScanImage frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tifffile

LineParity = Literal["odd", "even"]
FillMode = Literal["preserve", "nearest", "zero"]


def apply_bidirectional_phase_2d(
    image: np.ndarray,
    bidiphase: int,
    *,
    line_parity: LineParity = "odd",
    fill_mode: FillMode = "preserve",
    copy: bool = True,
) -> np.ndarray:
    """Shift alternating scan lines by an integer number of pixels.

    Parameters
    ----------
    image:
        Two-dimensional ``Y, X`` image.
    bidiphase:
        Integer x-shift applied to selected rows. Positive values move selected
        rows to the right by copying pixels from the left; negative values move
        selected rows to the left by copying pixels from the right. ``0`` returns
        the input unchanged, optionally copied.
    line_parity:
        Which rows to shift. ``"odd"`` matches the common Suite2p convention and
        the colleague script supplied with this project.
    fill_mode:
        How to handle newly exposed edge columns. ``"preserve"`` leaves them as
        the original values, matching the supplied script. ``"nearest"`` fills
        from the nearest shifted edge. ``"zero"`` fills with zero.
    copy:
        If True, do not modify the input array.

    Returns
    -------
    np.ndarray
        Corrected image with the same shape and dtype as the input.
    """

    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")

    out = image.copy() if copy else image
    phase = int(bidiphase)
    if phase == 0:
        return out

    rows = slice(1, None, 2) if line_parity == "odd" else slice(0, None, 2)
    width = out.shape[1]
    abs_phase = abs(phase)
    if abs_phase >= width:
        raise ValueError(f"abs(bidiphase) must be smaller than image width; got {phase}")

    original = image if copy else out.copy()

    if phase > 0:
        out[rows, phase:] = original[rows, :-phase]
        if fill_mode == "nearest":
            out[rows, :phase] = original[rows, :1]
        elif fill_mode == "zero":
            out[rows, :phase] = 0
        elif fill_mode != "preserve":
            raise ValueError(f"Unknown fill_mode: {fill_mode}")
    else:
        p = -phase
        out[rows, :-p] = original[rows, p:]
        if fill_mode == "nearest":
            out[rows, -p:] = original[rows, -1:]
        elif fill_mode == "zero":
            out[rows, -p:] = 0
        elif fill_mode != "preserve":
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

    return out


def apply_bidirectional_phase(
    data: np.ndarray,
    bidiphase: int,
    *,
    line_parity: LineParity = "odd",
    fill_mode: FillMode = "preserve",
    copy: bool = True,
) -> np.ndarray:
    """Apply bidirectional line-phase correction to 2-D or stack-like arrays.

    The last two dimensions are interpreted as ``Y, X``. Leading dimensions are
    treated as frame/channel/z dimensions and are processed independently.
    """

    if data.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {data.shape}")

    phase = int(bidiphase)
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
        )

    flat = out.reshape((-1, *out.shape[-2:]))
    for idx in range(flat.shape[0]):
        apply_bidirectional_phase_2d(
            flat[idx],
            phase,
            line_parity=line_parity,
            fill_mode=fill_mode,
            copy=False,
        )
    return out


def apply_bidirectional_phase_tiff(
    input_tif: str | Path,
    output_tif: str | Path,
    *,
    bidiphase: int,
    line_parity: LineParity = "odd",
    fill_mode: FillMode = "preserve",
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
        "bidiphase": int(bidiphase),
        "line_parity": line_parity,
        "fill_mode": fill_mode,
        "compression": compression,
        "n_pages": n_pages,
    }
