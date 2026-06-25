# Repository organization

This repo is structured around two principles:

1. **Source-specific code lives under `sources/`**. ScanImage and SLAP2 have very
different metadata conventions and file layouts, so their parsing and workflow
code should stay separate.
2. **Reusable image-processing code lives outside source-specific folders**.
Rigid registration, TIFF I/O, transforms, projections, and QC plotting should be
shared between ScanImage and SLAP2 pipelines where possible.

## Canonical package layout

```text
src/slap2_volume_align/
├── cli.py
├── core/
│   ├── registration.py             # implemented rigid x/y registration utilities
│   ├── preprocessing.py            # future filtering/background helpers
│   ├── transforms.py               # future geometry/coordinate transforms
│   ├── orientation.py              # future orientation conventions
│   ├── projections.py              # future MIP/mean/projection helpers
│   └── landmarks.py                # future fiducial/landmark helpers
├── readers/
│   └── tiff.py                     # implemented lazy TIFF indexing/writing
├── qc/
│   └── scanimage.py                # implemented ScanImage shift/QC outputs
├── sources/
│   ├── scanimage/
│   │   ├── metadata.py             # implemented ScanImage ImageDescription parsing
│   │   ├── pipeline.py             # implemented memory-safe volume averaging
│   │   └── subset.py               # implemented large-TIFF subset extraction
│   └── slap2/
│       ├── metadata.py             # REFERENCE TIFF JSON parsing and z-offset specs
│       ├── reference_stack.py      # lazy plane reads and z interpolation
│       ├── geometry.py             # DMD pixel/sample-coordinate transforms
│       ├── merge.py                # DMD1+DMD2 super-stack warping/blending
│       ├── overlap.py              # overlap helper namespace
│       └── qc.py                   # footprint and merge QC plots
└── visualization/
    └── viewer.py                   # optional napari/matplotlib viewers later
```

## Import conventions

Prefer canonical imports in new code:

```python
from slap2_volume_align.sources.scanimage.pipeline import (
    ScanImageAverageConfig,
    average_scanimage_volume,
)
from slap2_volume_align.readers.tiff import read_tiff_stack_spec
from slap2_volume_align.core.registration import estimate_rigid_shift
```

## Development direction

### Near-term: Bruker/ScanImage

- Harden the current single-channel `slice_blocks` workflow.
- Validate two-channel page-interleaved data.
- Add optional focus/z-quality metrics.
- Add output choices for `float32` analysis TIFF, display-scaled `uint16`, and OME-Zarr.

### Current: SLAP2 GUI reference super-stacks

The `sources/slap2/` code currently starts from GUI-generated `*-REFERENCE.tif`
files. It parses per-page JSON `ImageDescription` metadata, uses
`dmdPixel2SampleTransform` for metadata-based sample-space placement, then can
apply:

- a manual DMD2 axial stitch correction via `dmd2_z_offset_um`;
- residual DMD2-to-DMD1 XY registration from the corrected overlap slab;
- XY and z feathering before writing Fiji-compatible BigTIFF outputs.

For the current 836174 example, overlap diagnostics estimated
`dmd2_z_offset_um = -7.5`, meaning DMD2 is shifted 5 planes superficial before
z interpolation, overlap inference, residual XY registration, and blending.

### Next: raw SLAP2 reference-stack averaging

Implement a Pythonic port of:

```text
slap2/+slap2/+util/computeReferenceImage.m
```

as a separate workflow from DMD merging. Expected pieces include per-frame
metadata inference, z-local template matching / limited axial reassignment,
shared x/y shifts across channels, averaged reference volumes, and shift/QC
summaries.

### Later: raw SLAP2 `.dat` + `.meta`

Keep raw SLAP2 support separate in `sources/slap2/dat_reader.py`. Raw `.dat` support
requires reconstructing frames from ParsePlan metadata, superpixel IDs, and pixel
replacement maps; this should not be mixed into the reference-stack TIFF path.
