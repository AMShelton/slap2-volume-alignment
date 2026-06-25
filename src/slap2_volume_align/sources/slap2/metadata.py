"""Metadata helpers for SLAP2 GUI reference-stack TIFFs.

The SLAP2 GUI ``*-REFERENCE.tif`` files used by this package store one JSON
``ImageDescription`` per page. In version 2 reference stacks that JSON contains
at least:

    - z: plane depth in microns
    - channel: 1-based channel index
    - acquisitionPathIdx: 1 for Path1/DMD1, 2 for Path2/DMD2
    - dmdPixel2SampleTransform: 4 x 4 homogeneous transform

For ImageJ/Fiji-derived substacks, the original JSON metadata may be missing.
For those cases use :func:`make_manual_reference_stack_spec` in notebooks/tests
by supplying the z start/spacing and transform explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile


@dataclass(frozen=True)
class Slap2ReferencePageInfo:
    """Per-page metadata for a SLAP2 reference stack."""

    page_index: int
    z_um: float
    channel: int
    acquisition_path_idx: int
    dmd_pixel_to_sample_transform: list[list[float]]
    source_tif_file: Optional[str] = None
    raw_description: Optional[str] = None

    def transform_array(self) -> np.ndarray:
        return np.asarray(self.dmd_pixel_to_sample_transform, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Slap2ReferenceStackSpec:
    """Metadata needed to read and place a SLAP2 ``*-REFERENCE.tif`` stack."""

    path: str
    n_pages: int
    image_shape: tuple[int, int]
    dtype: str
    axes: str
    pages: tuple[Slap2ReferencePageInfo, ...] = field(default_factory=tuple)

    @property
    def channels(self) -> list[int]:
        return sorted({p.channel for p in self.pages})

    @property
    def z_positions_um(self) -> np.ndarray:
        return np.asarray([p.z_um for p in self.pages], dtype=float)

    @property
    def z_positions_by_channel(self) -> dict[int, np.ndarray]:
        out: dict[int, list[float]] = {}
        for p in self.pages:
            out.setdefault(p.channel, []).append(p.z_um)
        return {ch: np.asarray(sorted(vals), dtype=float) for ch, vals in out.items()}

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_z_planes(self) -> int:
        if not self.pages:
            return self.n_pages
        return max(len(v) for v in self.z_positions_by_channel.values())

    @property
    def acquisition_path_idx(self) -> Optional[int]:
        vals = sorted({p.acquisition_path_idx for p in self.pages})
        return vals[0] if len(vals) == 1 else None

    @property
    def representative_transform(self) -> np.ndarray:
        if not self.pages:
            raise ValueError("No page metadata available; cannot get representative transform.")
        return self.pages[0].transform_array()

    @property
    def z_min_um(self) -> float:
        return float(np.nanmin(self.z_positions_um))

    @property
    def z_max_um(self) -> float:
        return float(np.nanmax(self.z_positions_um))

    @property
    def z_spacing_um(self) -> Optional[float]:
        z = np.unique(np.round(self.z_positions_um.astype(float), decimals=6))
        if z.size < 2:
            return None
        dz = np.diff(np.sort(z))
        return float(np.median(dz))

    def pages_for_channel(self, channel: int) -> list[Slap2ReferencePageInfo]:
        return sorted([p for p in self.pages if p.channel == channel], key=lambda p: p.z_um)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["image_shape"] = list(self.image_shape)
        out["channels"] = self.channels
        out["n_channels"] = self.n_channels
        out["n_z_planes"] = self.n_z_planes
        out["z_min_um"] = self.z_min_um if self.pages else None
        out["z_max_um"] = self.z_max_um if self.pages else None
        out["z_spacing_um"] = self.z_spacing_um if self.pages else None
        return out



def offset_reference_stack_z(
    spec: Slap2ReferenceStackSpec,
    offset_um: float,
    *,
    path: Optional[str | Path] = None,
) -> Slap2ReferenceStackSpec:
    """Return a copy of ``spec`` with every page z position shifted by ``offset_um``.

    This is used for DMD-to-DMD axial stitch calibration. The original TIFF
    metadata is not modified; only the in-memory z coordinates used for grid
    inference, z interpolation, overlap inference, and blending are shifted.
    The XY transform is intentionally preserved because the z-offset corrects
    axial placement only.
    """

    offset_um = float(offset_um)
    if abs(offset_um) < 1e-12 and path is None:
        return spec

    shifted_pages = tuple(replace(p, z_um=float(p.z_um + offset_um)) for p in spec.pages)
    return replace(spec, path=str(path) if path is not None else spec.path, pages=shifted_pages)

def _parse_json_description(description: str) -> dict[str, Any]:
    desc = (description or "").strip()
    if not desc.startswith("{"):
        raise ValueError("TIFF ImageDescription does not appear to be JSON.")
    return json.loads(desc)


def parse_reference_page_description(description: str, page_index: int) -> Slap2ReferencePageInfo:
    """Parse one SLAP2 reference-stack JSON ImageDescription."""

    info = _parse_json_description(description)

    missing = [
        key
        for key in ("z", "channel", "acquisitionPathIdx", "dmdPixel2SampleTransform")
        if key not in info
    ]
    if missing:
        raise KeyError(f"Missing required SLAP2 reference metadata keys: {missing}")

    return Slap2ReferencePageInfo(
        page_index=page_index,
        z_um=float(info["z"]),
        channel=int(info["channel"]),
        acquisition_path_idx=int(info["acquisitionPathIdx"]),
        dmd_pixel_to_sample_transform=[list(map(float, row)) for row in info["dmdPixel2SampleTransform"]],
        source_tif_file=info.get("sourceTifFile"),
        raw_description=description,
    )


def read_reference_stack_spec(path: str | Path, *, require_json: bool = True) -> Slap2ReferenceStackSpec:
    """Read stack shape and per-page SLAP2 JSON metadata from a REFERENCE TIFF."""

    path = Path(path)
    pages: list[Slap2ReferencePageInfo] = []

    with tifffile.TiffFile(path) as tif:
        n_pages = len(tif.pages)
        image_shape = tuple(int(v) for v in tif.pages[0].shape[-2:])
        dtype = str(tif.pages[0].dtype)
        axes = tif.series[0].axes if tif.series else "IYX"

        for i, page in enumerate(tif.pages):
            desc = page.description or ""
            try:
                pages.append(parse_reference_page_description(desc, i))
            except Exception:
                if require_json:
                    raise
                continue

    return Slap2ReferenceStackSpec(
        path=str(path),
        n_pages=n_pages,
        image_shape=image_shape,  # type: ignore[arg-type]
        dtype=dtype,
        axes=axes,
        pages=tuple(pages),
    )


def make_manual_reference_stack_spec(
    path: str | Path,
    *,
    z_start_um: float,
    z_spacing_um: float,
    transform: list[list[float]] | np.ndarray,
    channel: int = 1,
    acquisition_path_idx: int = 1,
    n_pages: Optional[int] = None,
) -> Slap2ReferenceStackSpec:
    """Construct a stack spec for metadata-stripped Fiji substacks.

    This is intended for QC notebooks only. Full processing should use the
    original ``*-REFERENCE.tif`` files because they preserve page-level JSON.
    """

    path = Path(path)
    transform_list = np.asarray(transform, dtype=float).tolist()

    with tifffile.TiffFile(path) as tif:
        detected_pages = len(tif.pages)
        image_shape = tuple(int(v) for v in tif.pages[0].shape[-2:])
        dtype = str(tif.pages[0].dtype)
        axes = tif.series[0].axes if tif.series else "IYX"

    if n_pages is None:
        n_pages = detected_pages
    if n_pages > detected_pages:
        raise ValueError(f"n_pages={n_pages} exceeds TIFF page count {detected_pages}.")

    pages = []
    for i in range(n_pages):
        pages.append(
            Slap2ReferencePageInfo(
                page_index=i,
                z_um=float(z_start_um + i * z_spacing_um),
                channel=int(channel),
                acquisition_path_idx=int(acquisition_path_idx),
                dmd_pixel_to_sample_transform=transform_list,
                source_tif_file=None,
                raw_description=None,
            )
        )

    return Slap2ReferenceStackSpec(
        path=str(path),
        n_pages=n_pages,
        image_shape=image_shape,  # type: ignore[arg-type]
        dtype=dtype,
        axes=axes,
        pages=tuple(pages),
    )
