from slap2_volume_align.core.registration import estimate_rigid_shift
from slap2_volume_align.readers.tiff import scanimage_page_index
from slap2_volume_align.sources.scanimage.metadata import ScanImageStackSpec
from slap2_volume_align.sources.scanimage.pipeline import ScanImageAverageConfig


def test_scanimage_page_index_slice_blocks():
    spec = ScanImageStackSpec(
        n_pages=60,
        image_shape=(2048, 2048),
        dtype="int16",
        n_planes=3,
        repeats_per_plane=20,
        n_channels=1,
        order="slice_blocks",
    )
    assert scanimage_page_index(z_index=2, repeat_index=0, channel_index=0, spec=spec) == 40


def test_public_imports_available():
    assert estimate_rigid_shift is not None
    assert ScanImageAverageConfig is not None
