"""Future Python port of SLAP2 reference-stack averaging.

Target MATLAB reference behavior:
    ``slap2/+slap2/+util/computeReferenceImage.m``

Planned responsibilities:
    - read exported SLAP2 reference-stack TIFFs lazily;
    - parse JSON ImageDescription metadata per page;
    - reshape pages into X/Y/channel/frame/z/volume-repeat layout;
    - build z-local templates and allow limited axial reassignment;
    - apply shared x/y shifts to all channels;
    - write averaged reference volumes and QC summaries.
"""

from __future__ import annotations


class Slap2ReferenceStackNotImplementedError(NotImplementedError):
    """Raised for SLAP2 reference-stack calls before implementation."""


def average_slap2_reference_stack(*args, **kwargs):
    """Placeholder for future SLAP2 reference-stack averaging."""

    raise Slap2ReferenceStackNotImplementedError(
        "SLAP2 reference-stack averaging is planned but not implemented yet. "
        "Start from slap2.util.computeReferenceImage.m and validate on a small "
        "exported SLAP2 GUI reference stack."
    )
