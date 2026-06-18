"""Large-file readers and writers.

Keep this initializer lightweight to avoid circular imports during notebook and
CLI startup. Import TIFF helpers directly from ``slap2_volume_align.readers.tiff``.
"""

__all__: list[str] = []
