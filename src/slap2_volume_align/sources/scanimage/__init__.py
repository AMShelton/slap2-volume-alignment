"""Bruker/ScanImage TIFF stack support.

Keep this package initializer intentionally light.

Importing heavy pipeline objects here creates circular imports because the
low-level TIFF reader needs ScanImage metadata classes, while the ScanImage
pipeline also needs the TIFF reader. Import concrete objects directly from their
submodules instead, e.g.:

    from slap2_volume_align.sources.scanimage.pipeline import average_scanimage_volume
    from slap2_volume_align.sources.scanimage.metadata import ScanImageStackSpec
"""

__all__: list[str] = []
