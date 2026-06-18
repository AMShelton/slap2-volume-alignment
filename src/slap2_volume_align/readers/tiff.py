"""TIFF IO helpers for large ScanImage/Bruker structural stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

from slap2_volume_align.sources.scanimage.metadata import ScanImageStackSpec, infer_scanimage_plane_count


def read_tiff_stack_spec(
    path: str | Path,
    *,
    n_planes: int | None = None,
    repeats_per_plane: int | None = None,
    n_channels: int = 1,
    order: str = "slice_blocks",
    infer_from_descriptions: bool = True,
) -> ScanImageStackSpec:
    """Read enough metadata to index a large TIFF without loading image data."""

    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        n_pages = len(tif.pages)
        image_shape = tuple(int(v) for v in tif.pages[0].shape)
        dtype = str(tif.pages[0].dtype)

        inferred_planes = None
        inferred_repeats = None
        if infer_from_descriptions and (n_planes is None or repeats_per_plane is None):
            descriptions = [page.description for page in tif.pages]
            inferred_planes, inferred_repeats = infer_scanimage_plane_count(
                descriptions, n_channels=n_channels
            )

    if n_planes is None:
        if inferred_planes is None:
            raise ValueError("n_planes was not provided and could not be inferred")
        n_planes = inferred_planes
    if repeats_per_plane is None:
        if inferred_repeats is None:
            raise ValueError("repeats_per_plane was not provided and could not be inferred")
        repeats_per_plane = inferred_repeats

    expected_pages = n_planes * repeats_per_plane * n_channels
    if expected_pages != n_pages:
        raise ValueError(
            f"Layout implies {expected_pages} pages, but TIFF contains {n_pages}. "
            "Check n_planes, repeats_per_plane, n_channels, and order."
        )

    return ScanImageStackSpec(
        n_pages=n_pages,
        image_shape=image_shape,  # type: ignore[arg-type]
        dtype=dtype,
        n_planes=n_planes,
        repeats_per_plane=repeats_per_plane,
        n_channels=n_channels,
        order=order,
    )


def scanimage_page_index(
    *,
    z_index: int,
    repeat_index: int,
    channel_index: int,
    spec: ScanImageStackSpec,
) -> int:
    """Map zero-based z/repeat/channel indices to a TIFF page index."""

    if spec.order == "slice_blocks":
        return (
            z_index * spec.repeats_per_plane * spec.n_channels
            + repeat_index * spec.n_channels
            + channel_index
        )
    if spec.order == "volume_interleaved":
        return (
            repeat_index * spec.n_planes * spec.n_channels
            + z_index * spec.n_channels
            + channel_index
        )
    raise ValueError(f"Unknown order: {spec.order}")


def selected_scanimage_pages(
    *,
    plane_indices: Iterable[int],
    spec: ScanImageStackSpec,
) -> list[int]:
    """List all pages corresponding to selected planes."""

    pages: list[int] = []
    for z in plane_indices:
        for r in range(spec.repeats_per_plane):
            for c in range(spec.n_channels):
                pages.append(
                    scanimage_page_index(
                        z_index=z, repeat_index=r, channel_index=c, spec=spec
                    )
                )
    return pages


def read_plane_channel_frames(
    tif: tifffile.TiffFile,
    *,
    z_index: int,
    channel_index: int,
    spec: ScanImageStackSpec,
) -> list[np.ndarray]:
    """Read all repeats for one z-plane and one channel."""

    frames = []
    for repeat_index in range(spec.repeats_per_plane):
        page_index = scanimage_page_index(
            z_index=z_index,
            repeat_index=repeat_index,
            channel_index=channel_index,
            spec=spec,
        )
        frames.append(tif.pages[page_index].asarray())
    return frames


def write_volume_tiff(
    path: str | Path,
    volume_zyx: np.ndarray,
    *,
    dtype: str = "float32",
    compression: str | None = "zlib",
    description: str | None = None,
) -> None:
    """Write a z/y/x volume TIFF."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = volume_zyx
    if dtype != "preserve" and dtype is not None:
        data = data.astype(dtype, copy=False)

    tifffile.imwrite(
        path,
        data,
        bigtiff=True,
        photometric="minisblack",
        compression=compression,
        metadata={"axes": "ZYX"},
        description=description,
    )
