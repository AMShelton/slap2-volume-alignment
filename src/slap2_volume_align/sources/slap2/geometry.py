"""Geometry utilities for placing SLAP2 DMD reference stacks in sample space."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from slap2_volume_align.sources.slap2.metadata import Slap2ReferenceStackSpec


@dataclass(frozen=True)
class SampleGridSpec:
    """Regular output grid in SLAP2/sample coordinates."""

    x_min_um: float
    x_max_um: float
    y_min_um: float
    y_max_um: float
    z_min_um: float
    z_max_um: float
    xy_resolution_um: float
    z_resolution_um: float
    x_size: int
    y_size: int
    z_size: int

    @property
    def x_coords_um(self) -> np.ndarray:
        return self.x_min_um + np.arange(self.x_size, dtype=float) * self.xy_resolution_um

    @property
    def y_coords_um(self) -> np.ndarray:
        return self.y_min_um + np.arange(self.y_size, dtype=float) * self.xy_resolution_um

    @property
    def z_coords_um(self) -> np.ndarray:
        return self.z_min_um + np.arange(self.z_size, dtype=float) * self.z_resolution_um

    def to_dict(self) -> dict:
        return asdict(self)


def affine_xy_from_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A, t`` for ``sample_xy = A @ pixel_xy + t``.

    The SLAP2 reference TIFF JSON stores a standard column-vector homogeneous
    transform with XY translation in the fourth column. Pixel coordinates use
    ``x = column`` and ``y = row``.
    """

    T = np.asarray(transform, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {T.shape}.")
    A = T[:2, :2]
    t = T[:2, 3]
    return A, t


def pixel_to_sample_xy(transform: np.ndarray, x_px: np.ndarray, y_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform DMD/image pixel coordinates to sample-space XY microns."""

    A, t = affine_xy_from_transform(transform)
    x = A[0, 0] * x_px + A[0, 1] * y_px + t[0]
    y = A[1, 0] * x_px + A[1, 1] * y_px + t[1]
    return x, y


def sample_to_pixel_xy(transform: np.ndarray, x_um: np.ndarray, y_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse transform from sample-space XY microns to DMD/image pixels."""

    A, t = affine_xy_from_transform(transform)
    invA = np.linalg.inv(A)
    dx = x_um - t[0]
    dy = y_um - t[1]
    x_px = invA[0, 0] * dx + invA[0, 1] * dy
    y_px = invA[1, 0] * dx + invA[1, 1] * dy
    return x_px, y_px


def stack_corners_sample_xy(spec: Slap2ReferenceStackSpec) -> np.ndarray:
    """Return ``N x 2`` sample-space XY corners for a reference stack."""

    h, w = spec.image_shape
    # Pixel-center corner coordinates. This is enough for footprint QC and grid bounds.
    corners_px = np.asarray(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=float,
    )
    T = spec.representative_transform
    x_um, y_um = pixel_to_sample_xy(T, corners_px[:, 0], corners_px[:, 1])
    return np.column_stack([x_um, y_um])


def estimate_native_xy_resolution_um(spec: Slap2ReferenceStackSpec) -> float:
    """Estimate native pixel spacing in sample-space microns."""

    A, _ = affine_xy_from_transform(spec.representative_transform)
    dx = float(np.linalg.norm(A[:, 0]))
    dy = float(np.linalg.norm(A[:, 1]))
    return float(np.mean([dx, dy]))


def compute_output_grid(
    specs: list[Slap2ReferenceStackSpec],
    *,
    xy_resolution_um: Optional[float] = None,
    z_resolution_um: Optional[float] = None,
    padding_um: float = 2.0,
    z_grid: str = "union",
) -> SampleGridSpec:
    """Compute a common output grid covering one or more DMD stacks."""

    if not specs:
        raise ValueError("At least one reference stack spec is required.")

    all_corners = np.concatenate([stack_corners_sample_xy(s) for s in specs], axis=0)
    x_min = float(np.floor((np.nanmin(all_corners[:, 0]) - padding_um) / 1.0) * 1.0)
    x_max = float(np.ceil((np.nanmax(all_corners[:, 0]) + padding_um) / 1.0) * 1.0)
    y_min = float(np.floor((np.nanmin(all_corners[:, 1]) - padding_um) / 1.0) * 1.0)
    y_max = float(np.ceil((np.nanmax(all_corners[:, 1]) + padding_um) / 1.0) * 1.0)

    if xy_resolution_um is None:
        xy_resolution_um = float(np.median([estimate_native_xy_resolution_um(s) for s in specs]))

    if z_resolution_um is None:
        dzs = [s.z_spacing_um for s in specs if s.z_spacing_um is not None]
        z_resolution_um = float(np.median(dzs)) if dzs else 1.5

    z_min = float(min(s.z_min_um for s in specs))
    z_max = float(max(s.z_max_um for s in specs))

    if z_grid == "first":
        # Useful when DMD1 is the superficial/master stack.
        master = specs[0]
        z_min = float(master.z_min_um)
        # Still extend to union max so deep DMDs are included.
        z_max = float(max(s.z_max_um for s in specs))
    elif z_grid != "union":
        raise ValueError("z_grid must be 'union' or 'first'.")

    x_size = int(np.floor((x_max - x_min) / xy_resolution_um)) + 1
    y_size = int(np.floor((y_max - y_min) / xy_resolution_um)) + 1
    z_size = int(np.floor((z_max - z_min) / z_resolution_um)) + 1

    return SampleGridSpec(
        x_min_um=x_min,
        x_max_um=x_min + (x_size - 1) * xy_resolution_um,
        y_min_um=y_min,
        y_max_um=y_min + (y_size - 1) * xy_resolution_um,
        z_min_um=z_min,
        z_max_um=z_min + (z_size - 1) * z_resolution_um,
        xy_resolution_um=float(xy_resolution_um),
        z_resolution_um=float(z_resolution_um),
        x_size=x_size,
        y_size=y_size,
        z_size=z_size,
    )


def infer_z_overlap_um(spec1: Slap2ReferenceStackSpec, spec2: Slap2ReferenceStackSpec) -> tuple[float, float]:
    """Return overlapping z interval in microns."""

    z0 = max(spec1.z_min_um, spec2.z_min_um)
    z1 = min(spec1.z_max_um, spec2.z_max_um)
    if z1 < z0:
        raise ValueError(f"No z-overlap: [{spec1.z_min_um}, {spec1.z_max_um}] vs [{spec2.z_min_um}, {spec2.z_max_um}]")
    return float(z0), float(z1)
