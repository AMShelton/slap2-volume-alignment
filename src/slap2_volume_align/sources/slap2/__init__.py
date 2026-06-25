"""SLAP2 GUI reference-stack support."""

from slap2_volume_align.sources.slap2.metadata import (
    Slap2ReferencePageInfo,
    Slap2ReferenceStackSpec,
    make_manual_reference_stack_spec,
    read_reference_stack_spec,
)

__all__ = [
    "Slap2ReferencePageInfo",
    "Slap2ReferenceStackSpec",
    "read_reference_stack_spec",
    "make_manual_reference_stack_spec",
]
