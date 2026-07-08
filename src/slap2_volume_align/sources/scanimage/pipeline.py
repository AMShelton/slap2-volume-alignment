"""Memory-safe ScanImage structural volume averaging and z-plane registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile

from slap2_volume_align.readers.tiff import (
    read_plane_channel_frames,
    read_tiff_stack_spec,
    write_volume_tiff,
)
from slap2_volume_align.qc.scanimage import make_volume_qc_png, write_shift_csv
from slap2_volume_align.core.registration import (
    estimate_rigid_shift,
    make_template,
    mean_shifted_frames,
)
from slap2_volume_align.core.bidirectional import (
    apply_bidirectional_phase,
    estimate_bidirectional_phase_stack,
)
from slap2_volume_align.core.z_registration import (
    apply_z_registration,
    estimate_crop_z_registration,
    make_z_registration_qc_png,
    parse_crop_list_yx,
    write_z_registration_csv,
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
    # "odd" or "even" for application. "auto" is accepted by the pipeline and
    # resolved once from the first processed alignment plane using the configured
    # bidiphase; the notebook estimator should still be preferred when possible.
    bidi_line_parity: str = "odd"
    bidi_fill_mode: str = "nearest"
    # ``selected`` shifts one row parity only; ``symmetric`` shifts both parities
    # by half the requested relative phase and usually reduces sawtooth texture.
    bidi_shift_mode: str = "selected"
    # Optional repeated rigid-registration/template-refinement passes for repeats
    # within the same z plane.
    registration_n_passes: int = 2

    # Optional crop-based inter-plane registration of the averaged z volume.
    # This replaces the older global full-frame z-alignment path. Manual crops are
    # recommended for production; if z_registration_crops_yx is None, bright
    # auto-crops are inferred from the max projection.
    register_z: bool = False
    z_registration_crops_yx: Sequence[Sequence[int]] | str | None = None
    z_anchor: int | None = None
    z_auto_n_crops: int = 1
    z_auto_crop_size_px: int = 384
    z_template_radius: int = 2
    z_registration_binning: int = 4
    z_max_shift_px: int = 20
    z_highpass_sigma_px: float = 8.0
    z_intensity_transform: str = "sqrt"
    z_min_corr: float = 0.08
    z_min_overlap_fraction: float = 0.25
    z_min_accepted_crops: int = 1
    z_smooth_window: int = 0
    z_max_interpolation_gap: int = 1


def average_scanimage_volume(config: ScanImageAverageConfig) -> dict:
    """Align repeated frames per z-plane, average, and optionally register z planes.

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
    resolved_bidi_line_parity: str | None = None
    bidi_auto_estimate: dict | None = None

    with tifffile.TiffFile(input_tif) as tif:
        for out_z, z_index in enumerate(z_indices):
            print(f"Processing z {z_index} ({out_z + 1}/{len(z_indices)})")

            alignment_frames = read_plane_channel_frames(
                tif,
                z_index=z_index,
                channel_index=config.alignment_channel,
                spec=spec,
            )
            bidi_apply_line_parity: str | None = None
            if float(config.bidiphase) != 0.0:
                if resolved_bidi_line_parity is None:
                    resolved_bidi_line_parity, bidi_auto_estimate = _resolve_bidi_line_parity(
                        config, alignment_frames, z_index=z_index
                    )
                    if str(config.bidi_line_parity).lower() == "auto":
                        print(
                            "Resolved bidi_line_parity='auto' to "
                            f"{resolved_bidi_line_parity!r} from z {z_index}."
                        )
                bidi_apply_line_parity = resolved_bidi_line_parity
                alignment_frames = [
                    apply_bidirectional_phase(
                        frame,
                        config.bidiphase,
                        line_parity=bidi_apply_line_parity,
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
                            line_parity=bidi_apply_line_parity,
                            fill_mode=config.bidi_fill_mode,
                            shift_mode=config.bidi_shift_mode,
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

    z_registered_output_paths: dict[str, str] = {}
    z_registration_csv = None
    z_registration_crop_csv = None
    z_registration_rows: list[dict] = []
    z_registration_crop_rows: list[dict] = []
    z_registration_crops: list[tuple[int, int, int, int]] = []

    if config.register_z:
        crop_list = parse_crop_list_yx(config.z_registration_crops_yx, spec.image_shape)
        z_registration_rows, z_registration_crop_rows, z_registration_crops = estimate_crop_z_registration(
            volumes[config.alignment_channel],
            z_indices=z_indices,
            crops_yx=crop_list,
            anchor_z=config.z_anchor,
            auto_n_crops=config.z_auto_n_crops,
            auto_crop_size_px=config.z_auto_crop_size_px,
            template_radius=config.z_template_radius,
            max_shift_px=config.z_max_shift_px,
            binning=config.z_registration_binning,
            highpass_sigma_px=config.z_highpass_sigma_px,
            intensity_transform=config.z_intensity_transform,
            min_corr=config.z_min_corr,
            min_overlap_fraction=config.z_min_overlap_fraction,
            min_accepted_crops=config.z_min_accepted_crops,
            smooth_window=config.z_smooth_window,
            max_interpolation_gap=config.z_max_interpolation_gap,
            interpolation_order=config.interpolation_order,
        )
        z_registration_csv = out_dir / f"{stem}_z_registration_shifts.csv"
        z_registration_crop_csv = out_dir / f"{stem}_z_registration_crop_shifts.csv"
        write_z_registration_csv(z_registration_csv, z_registration_rows)
        write_z_registration_csv(z_registration_crop_csv, z_registration_crop_rows)

        for channel_index, volume in sorted(volumes.items()):
            z_registered = apply_z_registration(
                volume,
                z_registration_rows,
                interpolation_order=config.interpolation_order,
            )
            out_path = out_dir / f"{stem}_avg_ch{channel_index + 1}_zregistered.tif"
            write_volume_tiff(
                out_path,
                z_registered,
                dtype=config.output_dtype,
                compression=config.output_compression,
                description=(
                    f"Averaged and crop-based z-registered ScanImage volume from {input_tif.name}; "
                    f"channel={channel_index + 1}; axes=ZYX"
                ),
            )
            z_registered_output_paths[f"channel_{channel_index + 1}"] = str(out_path)
            volumes[channel_index] = z_registered

        if config.write_qc_png:
            z_qc_path = out_dir / f"{stem}_z_registration_qc.png"
            projection = np.nanmax(volumes[config.alignment_channel], axis=0)
            make_z_registration_qc_png(
                z_qc_path,
                z_registration_rows,
                z_registration_crop_rows,
                z_registration_crops,
                projection_image=projection,
            )

    shifts_path = out_dir / f"{stem}_alignment_shifts.csv"
    write_shift_csv(shifts_path, shift_rows)

    summary = {
        "config": _jsonify_dataclass(config),
        "stack_spec": spec.to_dict(),
        "z_indices": z_indices,
        "resolved_bidi_line_parity": resolved_bidi_line_parity,
        "bidi_auto_estimate": bidi_auto_estimate,
        "outputs": output_paths,
        "z_registered_outputs": z_registered_output_paths,
        "shifts_csv": str(shifts_path),
        "z_registration_csv": str(z_registration_csv) if z_registration_csv is not None else None,
        "z_registration_crop_csv": str(z_registration_crop_csv) if z_registration_crop_csv is not None else None,
        "z_registration_crops_yx": [list(c) for c in z_registration_crops],
    }

    if config.write_qc_png:
        qc_path = out_dir / f"{stem}_alignment_qc.png"
        make_volume_qc_png(qc_path, volumes)
        summary["qc_png"] = str(qc_path)
        if config.register_z:
            summary["z_registration_qc_png"] = str(out_dir / f"{stem}_z_registration_qc.png")

    summary_path = out_dir / f"{stem}_alignment_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def _resolve_bidi_line_parity(
    config: ScanImageAverageConfig,
    alignment_frames: Iterable[np.ndarray],
    *,
    z_index: int,
) -> tuple[str, dict | None]:
    """Return an application-safe BiDi line parity.

    ``apply_bidirectional_phase`` only accepts ``"odd"`` or ``"even"``. The
    notebook uses ``"auto"`` as a pre-estimation sentinel, so the pipeline also
    accepts ``"auto"`` and resolves it once from the first processed alignment
    plane. This prevents a stale notebook state from reaching the low-level apply
    function as an invalid parity.
    """

    parity = str(config.bidi_line_parity).lower()
    if parity in ("odd", "even"):
        return parity, None
    if parity != "auto":
        raise ValueError(
            "bidi_line_parity must be 'odd', 'even', or 'auto', "
            f"got {config.bidi_line_parity!r}"
        )

    frames = [np.asarray(frame) for frame in alignment_frames]
    if not frames:
        raise ValueError("Cannot resolve bidi_line_parity='auto' from an empty frame list")

    # Preserve the user's configured phase and choose only which row parity should
    # receive it. This is a safety fallback; the notebook's explicit estimator is
    # still preferred because it can jointly estimate phase sign/magnitude/parity
    # from hand-picked informative z planes.
    estimate = estimate_bidirectional_phase_stack(
        frames,
        phase_candidates=[float(config.bidiphase)],
        line_parity_candidates=("odd", "even"),
        crop_yx=(128, 128),
        x_stride=1,
        highpass_sigma_px=config.highpass_sigma_px,
        shift_mode=config.bidi_shift_mode,
    )
    resolved = str(estimate["best_line_parity"])
    return resolved, {
        "z_index": int(z_index),
        "input_bidiphase": float(config.bidiphase),
        "resolved_line_parity": resolved,
        "best_median_score": float(estimate["best_median_score"]),
        "aggregate_scores": estimate["aggregate_scores"],
    }


def _jsonify_dataclass(obj) -> dict:
    raw = asdict(obj)
    out = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out
