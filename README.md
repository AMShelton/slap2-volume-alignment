# SLAP2 / ScanImage volume alignment

This repository supports alignment and averaging of structural reference volumes.
The current implemented path is a memory-safe Bruker/ScanImage TIFF pipeline:
repeated frames at each z-plane are rigidly x/y motion-corrected, averaged, and
written as one `ZYX` TIFF per channel.

The package is organized so that common image-processing utilities can be shared
between ScanImage/Bruker volumes and a future Python port of SLAP2 GUI reference
stack processing.

## Current ScanImage/Bruker assumptions

The first supported dataset is a single large ScanImage TIFF with pages ordered as:

```text
z0 repeat0, z0 repeat1, ..., z0 repeat19,
z1 repeat0, z1 repeat1, ...
```

For Andrew's 2026-06-17 stack this corresponds to:

```text
3540 pages = 177 z-planes x 20 repeats x 1 channel
page shape = 2048 x 2048
page dtype = int16
```

The code also supports `volume_interleaved` ordering and page-interleaved channels,
but multi-channel ScanImage data should be verified with a small subset before a
full run.

## Install in editable mode

Run this from the repository root, not from inside `src/slap2_volume_align`:

```bash
pip install -e .
```

For notebook diagnostics:

```bash
pip install -e ".[notebooks]"
```

For optional napari-based viewing:

```bash
pip install -e ".[viewer]"
```

Avoid running Python from inside `src/slap2_volume_align`; local package names can
shadow standard-library modules when the working directory is inside a package.

## Repository organization

```text
slap2-volume-alignment/
├── docs/
│   └── REPO_ORGANIZATION.md
├── examples/
│   └── configs/
├── notebooks/
│   └── ScanImage_Alignment_QC.ipynb
├── src/slap2_volume_align/
│   ├── cli.py
│   ├── core/                       # shared registration/transforms/projections
│   ├── readers/                    # large TIFF readers/writers
│   ├── qc/                         # QC tables and figures
│   ├── sources/
│   │   ├── scanimage/              # implemented Bruker/ScanImage pipeline
│   │   └── slap2/                  # reserved for SLAP2 GUI stack support
│   └── visualization/              # optional interactive viewers
└── tests/
```

Use the canonical nested modules documented in `docs/REPO_ORGANIZATION.md` for new code and notebooks.

## Save a small representative subset

Preferred CLI form:

```bash
slap2-align scanimage-subset \
  "Z:/path/to/large_stack.tif" \
  "Z:/path/to/large_stack_subset.tif" \
  --planes 0-2 \
  --n-planes 177 \
  --repeats-per-plane 20 \
  --n-channels 1 \
  --order slice_blocks \
  --make-thumbnail-qc
```

For a low-bandwidth subset, add a central crop:

```bash
slap2-align scanimage-subset \
  "Z:/path/to/large_stack.tif" \
  "Z:/path/to/large_stack_subset_crop.tif" \
  --planes 0-2 \
  --n-planes 177 \
  --repeats-per-plane 20 \
  --n-channels 1 \
  --order slice_blocks \
  --crop 768:1280,768:1280 \
  --make-thumbnail-qc
```

## Average a small range first

Run a smoke test on a few planes before the full 30 GB stack:

```bash
slap2-align scanimage-average \
  "Z:/path/to/large_stack.tif" \
  "Z:/path/to/avg_test" \
  --n-planes 177 \
  --repeats-per-plane 20 \
  --n-channels 1 \
  --alignment-channel 0 \
  --order slice_blocks \
  --plane-start 0 \
  --plane-stop 3
```

## Full stack run

```bash
slap2-align scanimage-average \
  "Z:/path/to/large_stack.tif" \
  "Z:/path/to/avg_full" \
  --n-planes 177 \
  --repeats-per-plane 20 \
  --n-channels 1 \
  --alignment-channel 0 \
  --order slice_blocks
```

Outputs include:

```text
*_avg_ch1.tif                 # float32 averaged ZYX volume
*_alignment_shifts.csv        # per-repeat x/y shifts
*_alignment_summary.json      # config and stack metadata
*_alignment_qc.png            # quick visual QC mosaic
```

## Notes on dtype

Raw subsets preserve the original TIFF dtype. Averaged/motion-corrected output is
`float32` by default because subpixel interpolation and averaging create non-integer
values. Use `--output-dtype int16` only for display/export copies after confirming
that clipping/quantization are acceptable.

## SLAP2 GUI DMD super-stack merging

The repo now includes a first-pass pipeline for merging GUI-generated SLAP2
`*-REFERENCE.tif` stacks from DMD1 and DMD2 into one sample-coordinate super
stack. The code expects original SLAP2 reference TIFFs with JSON page
metadata containing `z`, `channel`, `acquisitionPathIdx`, and
`dmdPixel2SampleTransform`.

For Fiji/ImageJ-created substacks, the original SLAP2 JSON descriptions may be
lost. Use the notebook workflow to reconstruct minimal metadata manually for
these overlap-subset tests.

### Footprint QC

Run this before processing image data heavily:

```bash
slap2-align slap2-footprints \
  "Z:/path/to/structure_volume_DMD1-REFERENCE.tif" \
  "Z:/path/to/structure_volume_DMD2-REFERENCE.tif" \
  "Z:/path/to/super_stack_qc"
```

This writes:

```text
slap2_dmd_footprints_qc.png
slap2_dmd_footprints_summary.json
```

The plot should show DMD1/DMD2 footprints in common SLAP2/sample coordinates,
including the rotated DMD2 footprint and the inferred axial overlap.

### Merge DMD reference stacks

```bash
slap2-align slap2-merge-dmds \
  "Z:/path/to/structure_volume_DMD1-REFERENCE.tif" \
  "Z:/path/to/structure_volume_DMD2-REFERENCE.tif" \
  "Z:/path/to/super_stack" \
  --xy-resolution-um 0.25 \
  --z-resolution-um 1.5 \
  --fine-register-overlap
```

Outputs include:

```text
*_super_stack_ch1.tif              # Fiji/ImageJ-compatible merged ZYX volume
*_warped_ch1.tif                   # optional DMD-specific warped volumes
merge_weights.tif                  # accumulated merge weights/valid support
slap2_super_stack_merge_summary.json
slap2_dmd_footprints_qc.png
slap2_super_stack_merge_qc.png
```

Start with `--xy-resolution-um 1.0` or `0.5` for fast smoke tests, then use
native-ish `0.25` µm once geometry and overlap QC look correct.

### Notebook

Use:

```text
notebooks/slap2/SLAP2_SuperStack_Merge_QC.ipynb
```

for interactive testing, especially with metadata-stripped Fiji overlap
substacks. The notebook includes manual metadata reconstruction for the current
DMD1 last-plane / DMD2 first-plane overlap test.
