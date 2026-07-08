import numpy as np

from slap2_volume_align.core.bidirectional import (
    apply_bidirectional_phase,
    apply_bidirectional_phase_2d,
    estimate_bidirectional_phase_stack,
)
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


def test_stack_api_accepts_shift_mode_used_by_scanimage_notebook():
    rng = np.random.default_rng(123)
    frames = [rng.normal(size=(32, 48)).astype(np.float32) for _ in range(3)]

    estimate = estimate_bidirectional_phase_stack(
        frames,
        phase_candidates=[-1.0, 0.0, 1.0],
        line_parity_candidates=("odd", "even"),
        crop_yx=(0, 0),
        highpass_sigma_px=0.0,
        shift_mode="symmetric",
    )

    assert "best_phase" in estimate
    assert "best_line_parity" in estimate
    assert estimate["best_line_parity"] in {"odd", "even"}


def test_stack_apply_accepts_shift_mode_for_multiframe_arrays():
    data = np.tile(np.arange(12, dtype=np.float32), (2, 6, 1))
    out = apply_bidirectional_phase(
        data,
        4.0,
        line_parity="odd",
        fill_mode="nearest",
        shift_mode="symmetric",
    )

    assert out.shape == data.shape
    assert np.allclose(out[:, 1, 2:], data[:, 1, :-2])
    assert np.allclose(out[:, 0, :-2], data[:, 0, 2:])


def test_scanimage_config_exposes_bidi_shift_mode():
    cfg = ScanImageAverageConfig(input_tif="dummy.tif", out_dir="out")
    assert cfg.bidi_shift_mode == "selected"


def test_scanimage_pipeline_resolves_auto_parity_before_apply():
    from slap2_volume_align.sources.scanimage.pipeline import _resolve_bidi_line_parity

    rng = np.random.default_rng(456)
    frames = [rng.normal(size=(48, 64)).astype(np.float32) for _ in range(4)]
    cfg = ScanImageAverageConfig(
        input_tif="dummy.tif",
        out_dir="out",
        bidiphase=2.0,
        bidi_line_parity="auto",
        bidi_shift_mode="symmetric",
        highpass_sigma_px=0.0,
    )

    parity, estimate = _resolve_bidi_line_parity(cfg, frames, z_index=3)

    assert parity in {"odd", "even"}
    assert estimate is not None
    assert estimate["z_index"] == 3
    assert estimate["resolved_line_parity"] == parity


def test_scanimage_pipeline_rejects_invalid_bidi_parity_early():
    import pytest
    from slap2_volume_align.sources.scanimage.pipeline import _resolve_bidi_line_parity

    cfg = ScanImageAverageConfig(
        input_tif="dummy.tif",
        out_dir="out",
        bidiphase=2.0,
        bidi_line_parity="both",
    )

    with pytest.raises(ValueError, match="bidi_line_parity"):
        _resolve_bidi_line_parity(cfg, [np.zeros((16, 16), dtype=np.float32)], z_index=0)
