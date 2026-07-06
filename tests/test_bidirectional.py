import numpy as np

from slap2_volume_align.core.bidirectional import apply_bidirectional_phase_2d
from slap2_volume_align.sources.scanimage.pipeline import ScanImageAverageConfig


def test_bidirectional_symmetric_preserves_relative_integer_shift_float():
    image = np.tile(np.arange(12, dtype=np.float32), (6, 1))

    selected = apply_bidirectional_phase_2d(
        image,
        4.0,
        line_parity="odd",
        fill_mode="nearest",
        shift_mode="selected",
    )
    symmetric = apply_bidirectional_phase_2d(
        image,
        4.0,
        line_parity="odd",
        fill_mode="nearest",
        shift_mode="symmetric",
    )

    # Selected mode leaves even rows untouched and shifts odd rows by +4 px.
    assert np.allclose(selected[0], image[0])
    assert np.allclose(selected[1, 4:], image[1, :-4])

    # Symmetric mode shifts odd rows by +2 px and even rows by -2 px, preserving
    # the same 4 px relative odd/even correction while resampling both parities.
    assert np.allclose(symmetric[1, 2:], image[1, :-2])
    assert np.allclose(symmetric[0, :-2], image[0, 2:])


def test_scanimage_config_exposes_bidi_shift_mode():
    cfg = ScanImageAverageConfig(input_tif="dummy.tif", out_dir="out")
    assert cfg.bidi_shift_mode == "selected"
