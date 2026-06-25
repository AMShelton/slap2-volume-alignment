import numpy as np

from slap2_volume_align.sources.slap2.geometry import (
    compute_output_grid,
    infer_z_overlap_um,
    pixel_to_sample_xy,
    sample_to_pixel_xy,
    stack_corners_sample_xy,
)
from slap2_volume_align.sources.slap2.metadata import (
    Slap2ReferencePageInfo,
    Slap2ReferenceStackSpec,
    offset_reference_stack_z,
)


def _spec(path="dummy.tif", transform=None, z0=0.0, n=3):
    if transform is None:
        transform = [
            [0.25, 0.0, 0.0, 1.0],
            [0.0, 0.25, 0.0, 2.0],
            [0.0, 0.0, 1.0, z0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    pages = tuple(
        Slap2ReferencePageInfo(
            page_index=i,
            z_um=z0 + i * 1.5,
            channel=1,
            acquisition_path_idx=1,
            dmd_pixel_to_sample_transform=transform,
        )
        for i in range(n)
    )
    return Slap2ReferenceStackSpec(
        path=path,
        n_pages=n,
        image_shape=(10, 20),
        dtype="float32",
        axes="IYX",
        pages=pages,
    )


def test_pixel_sample_roundtrip():
    T = np.asarray(_spec().representative_transform)
    x = np.asarray([0.0, 10.0, 19.0])
    y = np.asarray([0.0, 5.0, 9.0])
    sx, sy = pixel_to_sample_xy(T, x, y)
    rx, ry = sample_to_pixel_xy(T, sx, sy)
    assert np.allclose(rx, x)
    assert np.allclose(ry, y)


def test_footprint_and_overlap():
    s1 = _spec(z0=-10.0, n=5)
    s2 = _spec(z0=-5.5, n=5)
    corners = stack_corners_sample_xy(s1)
    assert corners.shape == (4, 2)
    assert infer_z_overlap_um(s1, s2) == (-5.5, -4.0)
    grid = compute_output_grid([s1, s2], xy_resolution_um=1.0, z_resolution_um=1.5)
    assert grid.x_size > 1
    assert grid.y_size > 1
    assert grid.z_size > 1


def test_offset_reference_stack_z():
    s = _spec(z0=-60.0, n=3)
    shifted = offset_reference_stack_z(s, -7.5)
    assert s.z_min_um == -60.0
    assert shifted.z_min_um == -67.5
    assert shifted.z_max_um == -64.5
    assert shifted.pages[0].page_index == s.pages[0].page_index
    assert np.allclose(shifted.representative_transform, s.representative_transform)
