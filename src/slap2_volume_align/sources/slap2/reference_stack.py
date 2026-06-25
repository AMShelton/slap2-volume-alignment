"""Lazy readers for SLAP2 GUI ``*-REFERENCE.tif`` stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from slap2_volume_align.sources.slap2.metadata import (
    Slap2ReferencePageInfo,
    Slap2ReferenceStackSpec,
    read_reference_stack_spec,
)


class Slap2ReferenceStackNotImplementedError(NotImplementedError):
    """Raised for raw SLAP2 reference-stack averaging calls before implementation."""


def average_slap2_reference_stack(*args, **kwargs):
    """Placeholder for future raw SLAP2 reference-stack averaging.

    Current SLAP2 support starts from GUI-generated ``*-REFERENCE.tif`` files.
    """

    raise Slap2ReferenceStackNotImplementedError(
        "SLAP2 raw reference-stack averaging is planned but not implemented yet. "
        "Use GUI-generated *-REFERENCE.tif files with the slap2 merge tools."
    )


def read_reference_plane(path: str | Path, page_index: int) -> np.ndarray:
    """Read one page from a SLAP2 reference TIFF."""

    with tifffile.TiffFile(path) as tif:
        return tif.pages[int(page_index)].asarray()


def read_reference_volume(path: str | Path, *, dtype: Optional[str] = None) -> np.ndarray:
    """Read the entire reference stack as ``Z x Y x X`` for one-channel stacks."""

    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.asarray(arr)


def iter_reference_planes(
    path: str | Path,
    *,
    spec: Optional[Slap2ReferenceStackSpec] = None,
    channel: int = 1,
):
    """Yield ``(page_info, image)`` sorted by z for a channel."""

    if spec is None:
        spec = read_reference_stack_spec(path)

    pages = spec.pages_for_channel(channel)
    with tifffile.TiffFile(path) as tif:
        for page_info in pages:
            yield page_info, tif.pages[page_info.page_index].asarray()


def get_channel_pages(spec: Slap2ReferenceStackSpec, channel: int = 1) -> list[Slap2ReferencePageInfo]:
    """Return page metadata sorted by z for a given channel."""

    pages = spec.pages_for_channel(channel)
    if not pages:
        raise ValueError(f"No pages found for channel {channel} in {spec.path}.")
    return pages


def read_z_interpolated_plane(
    path: str | Path,
    spec: Slap2ReferenceStackSpec,
    z_um: float,
    *,
    channel: int = 1,
    method: str = "linear",
) -> Optional[np.ndarray]:
    """Read/interpolate one image plane at a requested z position.

    Returns ``None`` if the requested z is outside the stack range.
    """

    pages = get_channel_pages(spec, channel=channel)
    z = np.asarray([p.z_um for p in pages], dtype=float)

    if z_um < z.min() or z_um > z.max():
        return None

    # Exact or nearest sampling.
    if method == "nearest" or z.size == 1:
        idx = int(np.argmin(np.abs(z - z_um)))
        return read_reference_plane(path, pages[idx].page_index).astype(np.float32, copy=False)

    # Linear interpolation between neighboring planes.
    upper = int(np.searchsorted(z, z_um, side="left"))
    if upper == 0:
        return read_reference_plane(path, pages[0].page_index).astype(np.float32, copy=False)
    if upper >= len(z):
        return read_reference_plane(path, pages[-1].page_index).astype(np.float32, copy=False)

    lower = upper - 1
    z0, z1 = z[lower], z[upper]
    if np.isclose(z0, z1):
        w = 0.0
    else:
        w = float((z_um - z0) / (z1 - z0))

    img0 = read_reference_plane(path, pages[lower].page_index).astype(np.float32, copy=False)
    img1 = read_reference_plane(path, pages[upper].page_index).astype(np.float32, copy=False)
    return (1.0 - w) * img0 + w * img1
