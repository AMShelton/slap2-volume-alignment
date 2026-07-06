"""Memory-safe Bruker/ScanImage structural volume averaging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

from slap2_volume_align.readers.tiff import (
    read_plane_channel_frames,
    read_tiff_stack_spec,
    write_volume_tiff,
)
from slap2_volume_align.sources.scanimage.metadata import ScanImageStackSpec
from slap2_volume_align.qc.scanimage import make_volume_qc_png, write_shift_csv
from slap2_volume_align.core.registration import (
    estimate_rigid_shift,
    make_template,
    mean_shifted_frames,
)
from slap2_volume_align.core.bidirectional import apply_bidirectional_phase
from slap2_volume_align.core.straightening import (
    apply_z_straightening,
    estimate_z_straightening_shifts,
    make_straightening_qc_png,
    write_straightening_csv,
)


@dataclass
class ScanImageAverageConfig:
    input_tif: str | Path
    out_dir: str | Path
    n_planes: int | None = None
    repeats_per_plane: int | None = None
    n_channels: int = 1
    alignment_channel: int = 0
    order: str = "slice_blocks"
    plane_start: int = 0
    plane_stop: int | None = None
    template_method: str = "median"
    align: bool = True
    registration_binning: int = 2
    upsample_factor: int = 10
    max_shift_px: float = 100.0
    highpass_sigma_px: float = 8.0
    interpolation_order: int = 1
    output_dtype: str = "float32"
    output_compression: str | None = "zlib"
    infer_from_descriptions: bool = True
    write_qc_png: bool = True
    # Optional bidirectional odd/even line phase correction before registration.
    # bidiphase=0 disables this. Fractional phases are supported and often useful.
    bidiphase: float = 0.0
    bidi_line_parity: str = "odd"
    bidi_fill_mode: str = "nearest"
<<<<<<< HEAD
    # ``selected`` shifts one row parity only; ``symmetric`` shifts both parities
    # by half the requested relative phase and usually reduces sawtooth texture.
    bidi_shift_mode: str = "selected"
=======
>>>>>>> 452465a780b8d014f66bf276ff5efb2d9fa32e92
    # Optional repeated rigid-registration/template-refinement passes.
    registration_n_passes: int = 2
    # Optional second-pass inter-plane straightening of the averaged z volume.
    straighten_z: bool = False
    z_anchor: int | None = None
    z_template_radius: int = 2
    z_registration_binning: int = 4
    z_upsample_factor: int = 10
    z_max_step_shift_px: float = 12.0
    z_highpass_sigma_px: float = 16.0
    z_smooth_window: int = 9


def average_scanimage_volume(config: ScanImageAverageConfig) -> dict:
    """Align repeated frames per z-plane and average into one volume/channel.

    Returns a dictionary containing output paths and summary metadata.
    """

    input_tif = Path(config.input_tif)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = read_tiff_stack_spec(
        input_tif,
        n_planes=config.n_planes,
        repeats_per_plane=config.repeats_per_plane,
        n_channels=config.n_channels,
        order=config.order,
        infer_from_descriptions=config.infer_from_descriptions,
    )

    if config.alignment_channel < 0 or config.alignment_channel >= spec.n_channels:
        raise ValueError(
            f"alignment_channel must be in [0, {spec.n_channels - 1}], "
            f"got {config.alignment_channel}"
        )

    plane_stop = spec.n_planes if config.plane_stop is None else config.plane_stop
    z_indices = list(range(config.plane_start, plane_stop))
    if not z_indices:
        raise ValueError("No z-planes selected")

    stem = input_tif.stem
    volumes: dict[int, np.ndarray] = {
        c: np.empty((len(z_indices), *spec.image_shape), dtype=np.float32)
        for c in range(spec.n_channels)
    }
    shift_rows: list[dict] = []

    with tifffile.TiffFile(input_tif) as tif:
        for out_z, z_index in enumerate(z_indices):
            print(f"Processing z {z_index} ({out_z + 1}/{len(z_indices)})")

            alignment_frames = read_plane_channel_frames(
                tif,
                z_index=z_index,
                channel_index=config.alignment_channel,
                spec=spec,
            )
            if float(config.bidiphase) != 0.0:
                alignment_frames = [
                    apply_bidirectional_phase(
                        frame,
                        config.bidiphase,
                        line_parity=config.bidi_line_parity,
                        fill_mode=config.bidi_fill_mode,
                        shift_mode=config.bidi_shift_mode,
                    )
                    for frame in alignment_frames
                ]
            template = make_template(alignment_frames, method=config.template_method)

            shifts: list[tuple[float, float]] = [(0.0, 0.0)] * len(alignment_frames)
            results = [None] * len(alignment_frames)
            if config.align:
                n_passes = max(1, int(config.registration_n_passes))
                for pass_index in range(n_passes):
                    pass_shifts: list[tuple[float, float]] = []
                    pass_results = []
                    for frame in alignment_frames:
                        result = estimate_rigid_shift(
                            template,
                            frame,
                            upsample_factor=config.upsample_factor,
                            max_shift_px=config.max_shift_px,
                            binning=config.registration_binning,
                            highpass_sigma_px=config.highpass_sigma_px,
                        )
                        pass_results.append(result)
                        pass_shifts.append(result.shift_yx if result.accepted else (0.0, 0.0))

                    shifts = pass_shifts
                    results = pass_results
                    if pass_index < n_passes - 1:
                        template = mean_shifted_frames(
                            alignment_frames,
                            shifts,
                            order=config.interpolation_order,
                        )

            for repeat_index, shift_yx in enumerate(shifts):
                result = results[repeat_index]
                shift_rows.append(
                    {
                        "z_index": z_index,
                        "repeat_index": repeat_index,
                        "alignment_channel": config.alignment_channel,
                        "shift_y_px": shift_yx[0],
                        "shift_x_px": shift_yx[1],
                        "accepted": True if result is None else result.accepted,
                        "error": "" if result is None else result.error,
                        "phase_difference": "" if result is None else result.phase_difference,
                        "reason": "" if result is None else result.reason,
                    }
                )

            # Apply the same shifts to every channel, preserving cross-channel alignment.
            for channel_index in range(spec.n_channels):
                frames = (
                    alignment_frames
                    if channel_index == config.alignment_channel
                    else read_plane_channel_frames(
                        tif,
                        z_index=z_index,
                        channel_index=channel_index,
                        spec=spec,
                    )
                )
                if float(config.bidiphase) != 0.0 and channel_index != config.alignment_channel:
                    frames = [
                        apply_bidirectional_phase(
                            frame,
                            config.bidiphase,
                            line_parity=config.bidi_line_parity,
                            fill_mode=config.bidi_fill_mode,
                        )
                        for frame in frames
                    ]
                volumes[channel_index][out_z] = mean_shifted_frames(
                    frames,
                    shifts,
                    order=config.interpolation_order,
                )

    output_paths: dict[str, str] = {}
    for channel_index, volume in sorted(volumes.items()):
        out_path = out_dir / f"{stem}_avg_ch{channel_index + 1}.tif"
        write_volume_tiff(
            out_path,
            volume,
            dtype=config.output_dtype,
            compression=config.output_compression,
            description=(
                f"Averaged ScanImage volume from {input_tif.name}; "
                f"channel={channel_index + 1}; axes=ZYX"
            ),
        )
        output_paths[f"channel_{channel_index + 1}"] = str(out_path)


    straightened_output_paths: dict[str, str] = {}
    if config.straighten_z:
        straightening_rows = estimate_z_straightening_shifts(
            volumes[config.alignment_channel],
            z_indices=z_indices,
            anchor_z=config.z_anchor,
            template_radius=config.z_template_radius,
            binning=config.z_registration_binning,
            upsample_factor=config.z_upsample_factor,
            max_step_shift_px=config.z_max_step_shift_px,
            highpass_sigma_px=config.z_highpass_sigma_px,
            smooth_window=config.z_smooth_window,
            interpolation_order=config.interpolation_order,
        )
        straightening_csv = out_dir / f"{stem}_z_straightening_shifts.csv"
        write_straightening_csv(straightening_csv, straightening_rows)

        for channel_index, volume in sorted(volumes.items()):
            straightened = apply_z_straightening(
                volume,
                straightening_rows,
                use_smoothed=True,
                interpolation_order=config.interpolation_order,
            )
            out_path = out_dir / f"{stem}_avg_ch{channel_index + 1}_straightened.tif"
            write_volume_tiff(
                out_path,
                straightened,
                dtype=config.output_dtype,
                compression=config.output_compression,
                description=(
                    f"Averaged and z-straightened ScanImage volume from {input_tif.name}; "
                    f"channel={channel_index + 1}; axes=ZYX"
                ),
            )
            straightened_output_paths[f"channel_{channel_index + 1}"] = str(out_path)
            volumes[channel_index] = straightened

        if config.write_qc_png:
            straight_qc_path = out_dir / f"{stem}_z_straightening_qc.png"
            make_straightening_qc_png(straight_qc_path, straightening_rows)
    else:
        straightening_rows = []
        straightening_csv = None

    shifts_path = out_dir / f"{stem}_alignment_shifts.csv"
    write_shift_csv(shifts_path, shift_rows)

    summary = {
        "config": _jsonify_dataclass(config),
        "stack_spec": spec.to_dict(),
        "z_indices": z_indices,
        "outputs": output_paths,
        "straightened_outputs": straightened_output_paths,
        "shifts_csv": str(shifts_path),
        "z_straightening_csv": str(straightening_csv) if straightening_csv is not None else None,
    }

    if config.write_qc_png:
        qc_path = out_dir / f"{stem}_alignment_qc.png"
        make_volume_qc_png(qc_path, volumes)
        summary["qc_png"] = str(qc_path)
        if config.straighten_z:
            summary["z_straightening_qc_png"] = str(out_dir / f"{stem}_z_straightening_qc.png")

    summary_path = out_dir / f"{stem}_alignment_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def _jsonify_dataclass(obj) -> dict:
    raw = asdict(obj)
    out = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out
