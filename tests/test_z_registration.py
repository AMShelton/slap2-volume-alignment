
import numpy as np
from scipy import ndimage

from slap2_volume_align.core.z_registration import (
    apply_z_registration,
    estimate_crop_z_registration,
    estimate_local_ncc_shift,
    infer_registration_crops,
)


def _synthetic_plane(shape=(96, 96)):
    y, x = np.indices(shape)
    img = np.zeros(shape, dtype=np.float32)
    spots = [
        (30, 35, 2.8, 1.0),
        (45, 55, 4.0, 0.7),
        (62, 42, 3.2, 0.5),
    ]
    for cy, cx, sig, amp in spots:
        img += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sig**2))
    return img.astype(np.float32)


def test_local_ncc_shift_recovers_known_translation():
    fixed = _synthetic_plane()
    true_shift = (3.0, -5.0)
    # moving is displaced opposite the corrective shift; applying true_shift should align it.
    moving = ndimage.shift(fixed, shift=(-true_shift[0], -true_shift[1]), order=1, mode="nearest")
    result = estimate_local_ncc_shift(fixed, moving, max_shift_px=10, min_corr=0.1, binning=1)
    assert result.accepted
    assert abs(result.shift_yx[0] - true_shift[0]) < 0.75
    assert abs(result.shift_yx[1] - true_shift[1]) < 0.75


def test_crop_z_registration_applies_consensus_shift():
    base = _synthetic_plane((128, 128))
    shifts_to_apply = np.array([
        [0, 0],
        [2, -3],
        [4, -6],
        [1, -2],
        [-2, 3],
    ], dtype=float)
    volume = np.empty((len(shifts_to_apply), *base.shape), dtype=np.float32)
    # Input planes are displaced opposite the shifts needed to correct them.
    for z, (dy, dx) in enumerate(shifts_to_apply):
        volume[z] = ndimage.shift(base, shift=(-dy, -dx), order=1, mode="nearest")

    rows, crop_rows, crops = estimate_crop_z_registration(
        volume,
        crops_yx=[(16, 96, 16, 96)],
        anchor_z=0,
        max_shift_px=10,
        binning=1,
        highpass_sigma_px=3,
        min_corr=0.05,
        min_overlap_fraction=0.2,
        smooth_window=0,
    )
    assert len(rows) == volume.shape[0]
    assert len(crop_rows) == volume.shape[0]
    assert crops == [(16, 96, 16, 96)]
    for z in range(volume.shape[0]):
        assert rows[z]["accepted"]
        assert abs(float(rows[z]["shift_y_px"]) - shifts_to_apply[z, 0]) < 1.0
        assert abs(float(rows[z]["shift_x_px"]) - shifts_to_apply[z, 1]) < 1.0

    corrected = apply_z_registration(volume, rows)
    mse_before = np.mean((volume[2] - volume[0]) ** 2)
    mse_after = np.mean((corrected[2] - corrected[0]) ** 2)
    assert mse_after < mse_before * 0.5


def test_infer_registration_crops_returns_valid_crop():
    vol = np.stack([_synthetic_plane((128,128)) for _ in range(3)])
    crops = infer_registration_crops(vol, n_crops=1, crop_size_px=48)
    assert len(crops) == 1
    y0, y1, x0, x1 = crops[0]
    assert 0 <= y0 < y1 <= 128
    assert 0 <= x0 < x1 <= 128
