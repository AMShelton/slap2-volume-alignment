"""Metadata containers for future SLAP2 reference-stack support."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Slap2ReferenceStackSpec:
    """Minimal metadata expected from an exported SLAP2 reference-stack TIFF.

    This is intentionally small until the Python implementation is validated
    against ``slap2.util.computeReferenceImage`` on real SLAP2 GUI outputs.
    """

    n_pages: int
    image_shape: tuple[int, int]
    dtype: str
    n_channels: int | None = None
    n_z_planes: int | None = None
    frames_per_slice: int | None = None
    n_volume_repeats: int | None = None
    acquisition_path_index: int | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["image_shape"] = list(self.image_shape)
        return out
