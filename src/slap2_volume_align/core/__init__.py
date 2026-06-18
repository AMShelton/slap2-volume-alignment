"""Core image-processing primitives shared across acquisition systems."""

from slap2_volume_align.core.registration import (
    ShiftResult,
    apply_rigid_shift,
    as_float_image,
    estimate_rigid_shift,
    make_template,
    mean_shifted_frames,
    nanmean_stack,
    robust_rescale_for_registration,
)

__all__ = [
    "ShiftResult",
    "as_float_image",
    "robust_rescale_for_registration",
    "estimate_rigid_shift",
    "apply_rigid_shift",
    "make_template",
    "nanmean_stack",
    "mean_shifted_frames",
]
