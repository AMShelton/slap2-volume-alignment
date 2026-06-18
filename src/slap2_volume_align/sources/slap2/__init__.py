"""Prospective SLAP2 GUI reference-stack and raw-data support.

This namespace is reserved for the Python port of the standard MATLAB SLAP2
reference image / reference stack processing pipeline. The first implementation
should target exported SLAP2 reference-stack TIFFs; raw .dat + .meta support can
be added later once the binary layout and ParsePlan metadata are validated.
"""

from slap2_volume_align.sources.slap2.metadata import Slap2ReferenceStackSpec

__all__ = ["Slap2ReferenceStackSpec"]
