"""Bruker/ScanImage TIFF stack support."""

from slap2_volume_align.sources.scanimage.metadata import (
    ScanImageFrameDescription,
    ScanImageStackSpec,
    infer_scanimage_plane_count,
    parse_scanimage_description,
)
from slap2_volume_align.sources.scanimage.pipeline import (
    ScanImageAverageConfig,
    average_scanimage_volume,
)
from slap2_volume_align.sources.scanimage.subset import save_scanimage_plane_subset

__all__ = [
    "ScanImageAverageConfig",
    "average_scanimage_volume",
    "ScanImageFrameDescription",
    "ScanImageStackSpec",
    "infer_scanimage_plane_count",
    "parse_scanimage_description",
    "save_scanimage_plane_subset",
]
