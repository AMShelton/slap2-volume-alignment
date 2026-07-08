"""Command line interface for slap2-volume-align."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from .sources.scanimage.pipeline import ScanImageAverageConfig, average_scanimage_volume
from .sources.scanimage.subset import save_scanimage_plane_subset
from .sources.slap2.merge import (
    Slap2MergeConfig,
    make_slap2_footprint_summary,
    merge_dmd_reference_stacks,
)

app = typer.Typer(help="SLAP2/ScanImage structural volume alignment utilities.")


@app.command("version")
def version() -> None:
    """Print package version."""

    from . import __version__

    print(__version__)


@app.command("scanimage-average")
def scanimage_average(
    input_tif: Path = typer.Argument(..., exists=True, help="Large ScanImage TIFF."),
    out_dir: Path = typer.Argument(..., help="Output directory."),
    n_planes: Optional[int] = typer.Option(None, help="Number of z-planes."),
    repeats_per_plane: Optional[int] = typer.Option(None, help="Repeats per z-plane."),
    n_channels: int = typer.Option(1, help="Number of page-interleaved channels."),
    alignment_channel: int = typer.Option(
        0, help="Zero-based channel used to estimate x/y shifts."
    ),
    order: str = typer.Option(
        "slice_blocks",
        help="slice_blocks or volume_interleaved. Your current stack is slice_blocks.",
    ),
    plane_start: int = typer.Option(0, help="First zero-based z-plane to process."),
    plane_stop: Optional[int] = typer.Option(
        None, help="Exclusive z-plane stop. Omit for all planes."
    ),
    no_align: bool = typer.Option(False, help="Average without motion correction."),
    registration_binning: int = typer.Option(
        2, help="Integer binning for shift estimation. Shifts are applied full-res."
    ),
    upsample_factor: int = typer.Option(10, help="Subpixel registration factor."),
    max_shift_px: float = typer.Option(100.0, help="Reject shifts larger than this."),
    highpass_sigma_px: float = typer.Option(8.0, help="High-pass sigma in full-res pixels."),
    output_dtype: str = typer.Option(
        "float32", help="Averaged output dtype. Prefer float32 for registered averages."
    ),
    output_compression: Optional[str] = typer.Option(
        "zlib", help="Lossless TIFF compression, e.g. zlib, lzw, or omit with ''."
    ),
    bidiphase: float = typer.Option(0.0, help="Optional bidirectional odd/even line correction in pixels. 0 disables."),
    bidi_line_parity: str = typer.Option("odd", help="Rows to shift for bidi correction: odd, even, or auto. Auto is resolved from the first processed alignment plane using bidiphase."),
    bidi_fill_mode: str = typer.Option("nearest", help="Bidi fill mode: nearest, preserve, or zero."),
    bidi_shift_mode: str = typer.Option(
        "selected",
        help="Bidi shift mode: selected shifts one row parity; symmetric shifts both parities by half the relative phase.",
    ),
    register_z: bool = typer.Option(False, help="Run crop-based z-plane registration after averaging."),
    z_registration_crops_yx: Optional[str] = typer.Option(None, help="Optional semicolon-delimited crops as 'y0:y1,x0:x1;y0:y1,x0:x1'. Omit for auto-crops."),
    z_anchor: Optional[int] = typer.Option(None, help="Anchor z-plane for crop-based z registration. Omit to auto-pick."),
    z_auto_n_crops: int = typer.Option(1, help="Number of auto-detected bright crops if no manual crop is supplied."),
    z_auto_crop_size_px: int = typer.Option(384, help="Auto crop size in full-resolution pixels."),
    z_template_radius: int = typer.Option(2, help="Local aligned-template radius for crop-based z registration."),
    z_registration_binning: int = typer.Option(4, help="Binning for crop-based z registration."),
    z_max_shift_px: int = typer.Option(20, help="Max crop-registration shift in full-resolution pixels."),
    z_highpass_sigma_px: float = typer.Option(8.0, help="High-pass sigma for crop-based z registration."),
    z_intensity_transform: str = typer.Option("sqrt", help="Intensity transform for crop registration: sqrt, log1p, or none."),
    z_min_corr: float = typer.Option(0.08, help="Minimum local NCC correlation to accept one crop shift."),
    z_min_overlap_fraction: float = typer.Option(0.25, help="Minimum crop overlap fraction during local NCC search."),
    z_min_accepted_crops: int = typer.Option(1, help="Minimum accepted crops needed to shift a z-plane."),
    z_smooth_window: int = typer.Option(0, help="Optional median smoothing window over accepted z shifts. 0 disables."),
    z_max_interpolation_gap: int = typer.Option(1, help="Maximum unsupported z gap to interpolate before optional smoothing."),
) -> None:
    """Motion-correct repeats at each z-plane and average a ScanImage volume."""

    compression = output_compression if output_compression else None
    config = ScanImageAverageConfig(
        input_tif=input_tif,
        out_dir=out_dir,
        n_planes=n_planes,
        repeats_per_plane=repeats_per_plane,
        n_channels=n_channels,
        alignment_channel=alignment_channel,
        order=order,
        plane_start=plane_start,
        plane_stop=plane_stop,
        align=not no_align,
        registration_binning=registration_binning,
        upsample_factor=upsample_factor,
        max_shift_px=max_shift_px,
        highpass_sigma_px=highpass_sigma_px,
        output_dtype=output_dtype,
        output_compression=compression,
        bidiphase=bidiphase,
        bidi_line_parity=bidi_line_parity,
        bidi_fill_mode=bidi_fill_mode,
        bidi_shift_mode=bidi_shift_mode,
        register_z=register_z,
        z_registration_crops_yx=z_registration_crops_yx,
        z_anchor=z_anchor,
        z_auto_n_crops=z_auto_n_crops,
        z_auto_crop_size_px=z_auto_crop_size_px,
        z_template_radius=z_template_radius,
        z_registration_binning=z_registration_binning,
        z_max_shift_px=z_max_shift_px,
        z_highpass_sigma_px=z_highpass_sigma_px,
        z_intensity_transform=z_intensity_transform,
        z_min_corr=z_min_corr,
        z_min_overlap_fraction=z_min_overlap_fraction,
        z_min_accepted_crops=z_min_accepted_crops,
        z_smooth_window=z_smooth_window,
        z_max_interpolation_gap=z_max_interpolation_gap,
    )
    summary = average_scanimage_volume(config)
    print("[green]Finished ScanImage averaging[/green]")
    print(summary)


@app.command("scanimage-subset")
def scanimage_subset(
    input_tif: Path = typer.Argument(..., exists=True, help="Large ScanImage TIFF."),
    output_tif: Path = typer.Argument(..., help="Output subset TIFF."),
    planes: str = typer.Option(..., help='Zero-based planes, e.g. "0-2" or "0,10,20".'),
    n_planes: Optional[int] = typer.Option(None, help="Number of z-planes."),
    repeats_per_plane: Optional[int] = typer.Option(None, help="Repeats per z-plane."),
    n_channels: int = typer.Option(1, help="Number of page-interleaved channels."),
    order: str = typer.Option("slice_blocks", help="slice_blocks or volume_interleaved."),
    compression: Optional[str] = typer.Option(
        None, help="Optional lossless TIFF compression, e.g. zlib or lzw."
    ),
    crop: Optional[str] = typer.Option(None, help="Optional crop as y0:y1,x0:x1."),
    make_thumbnail_qc: bool = typer.Option(False, help="Write a thumbnail contact sheet."),
    dry_run: bool = typer.Option(False, help="Print page selection without writing TIFF."),
) -> None:
    """Save a small representative plane subset from a large ScanImage TIFF."""

    compression_value = compression if compression else None
    summary = save_scanimage_plane_subset(
        input_tif,
        output_tif,
        planes=planes,
        n_planes=n_planes,
        repeats_per_plane=repeats_per_plane,
        n_channels=n_channels,
        order=order,
        compression=compression_value,
        crop=crop,
        make_thumbnail_qc=make_thumbnail_qc,
        dry_run=dry_run,
    )
    print("[green]Finished ScanImage subset extraction[/green]")
    print(summary)


@app.command("slap2-footprints")
def slap2_footprints(
    dmd1_tif: Path = typer.Argument(..., exists=True, help="DMD1 SLAP2 *-REFERENCE.tif."),
    dmd2_tif: Path = typer.Argument(..., exists=True, help="DMD2 SLAP2 *-REFERENCE.tif."),
    out_dir: Path = typer.Argument(..., help="Output directory for footprint QC."),
    xy_resolution_um: Optional[float] = typer.Option(None, help="Output XY resolution for grid preview. Defaults to native-ish."),
    z_resolution_um: Optional[float] = typer.Option(None, help="Output Z spacing. Defaults to inferred median dz."),
    dmd2_z_offset_um: float = typer.Option(0.0, help="Axial offset, in microns, applied to DMD2 before overlap/grid inference. Negative shifts DMD2 superficial."),
) -> None:
    """Plot SLAP2 DMD reference-stack footprints in sample coordinates."""

    summary = make_slap2_footprint_summary(
        dmd1_tif,
        dmd2_tif,
        out_dir,
        xy_resolution_um=xy_resolution_um,
        z_resolution_um=z_resolution_um,
        dmd2_z_offset_um=dmd2_z_offset_um,
    )
    print("[green]Finished SLAP2 footprint QC[/green]")
    print(summary)


@app.command("slap2-merge-dmds")
def slap2_merge_dmds(
    dmd1_tif: Path = typer.Argument(..., exists=True, help="DMD1 SLAP2 *-REFERENCE.tif."),
    dmd2_tif: Path = typer.Argument(..., exists=True, help="DMD2 SLAP2 *-REFERENCE.tif."),
    out_dir: Path = typer.Argument(..., help="Output directory for merged super stack."),
    channel: int = typer.Option(1, help="1-based channel to merge."),
    xy_resolution_um: Optional[float] = typer.Option(None, help="Output XY resolution in microns. Defaults to native-ish."),
    z_resolution_um: Optional[float] = typer.Option(None, help="Output Z spacing in microns. Defaults to inferred dz."),
    z_grid: str = typer.Option("first", help="'first' uses DMD1 z origin; 'union' uses union min/max."),
    dmd2_z_offset_um: float = typer.Option(0.0, help="Axial offset, in microns, applied to DMD2 before z interpolation/blending. Negative shifts DMD2 superficial."),
    padding_um: float = typer.Option(2.0, help="Sample-space XY padding around combined footprint."),
    z_interp_method: str = typer.Option("linear", help="Z sampling: linear or nearest."),
    output_compression: Optional[str] = typer.Option(None, help="TIFF compression, e.g. zlib/lzw, or omit."),
    fine_register_overlap: bool = typer.Option(False, help="Estimate residual XY shift in overlap and apply to DMD2."),
    residual_upsample_factor: int = typer.Option(10, help="Overlap residual registration upsample factor."),
    residual_registration_binning: int = typer.Option(2, help="Binning for residual overlap registration."),
    residual_highpass_sigma_px: float = typer.Option(8.0, help="High-pass sigma for residual overlap registration."),
    residual_max_shift_px: float = typer.Option(50.0, help="Reject residual shifts larger than this."),
    xy_feather_px: float = typer.Option(32.0, help="XY feather width in output pixels."),
    no_z_feather: bool = typer.Option(False, help="Disable axial feathering in the DMD overlap."),
    no_intermediates: bool = typer.Option(False, help="Do not write DMD1/DMD2 warped intermediate TIFFs."),
    no_qc_png: bool = typer.Option(False, help="Do not write QC PNGs."),
) -> None:
    """Merge two SLAP2 GUI DMD reference stacks into one Fiji-compatible super stack."""

    config = Slap2MergeConfig(
        dmd1_tif=dmd1_tif,
        dmd2_tif=dmd2_tif,
        out_dir=out_dir,
        channel=channel,
        xy_resolution_um=xy_resolution_um,
        z_resolution_um=z_resolution_um,
        z_grid=z_grid,
        dmd2_z_offset_um=dmd2_z_offset_um,
        padding_um=padding_um,
        z_interp_method=z_interp_method,
        output_compression=output_compression if output_compression else None,
        fine_register_overlap=fine_register_overlap,
        residual_upsample_factor=residual_upsample_factor,
        residual_registration_binning=residual_registration_binning,
        residual_highpass_sigma_px=residual_highpass_sigma_px,
        residual_max_shift_px=residual_max_shift_px,
        xy_feather_px=xy_feather_px,
        z_feather=not no_z_feather,
        write_intermediates=not no_intermediates,
        write_qc_png=not no_qc_png,
    )
    summary = merge_dmd_reference_stacks(config)
    print("[green]Finished SLAP2 DMD super-stack merge[/green]")
    print(summary)


if __name__ == "__main__":
    app()
