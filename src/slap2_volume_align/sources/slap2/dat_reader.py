"""Future raw SLAP2 .dat/.meta reader support.

Raw SLAP2 support is intentionally separated from exported reference-stack TIFF
support because it requires reconstruction from ParsePlan metadata, superpixel
IDs, and pixel replacement maps. Implement this only after the TIFF reference
stack path is validated.
"""
