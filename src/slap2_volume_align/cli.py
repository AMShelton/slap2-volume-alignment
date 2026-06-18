"""Command line interface for slap2-volume-align."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from .sources.scanimage.pipeline import ScanImageAverageConfig, average_scanimage_volume
from .sources.scanimage.subset import save_scanimage_plane_subset

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


if __name__ == "__main__":
    app()
