"""Save small representative subsets from large ScanImage TIFF stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from slap2_volume_align.readers.tiff import read_tiff_stack_spec, selected_scanimage_pages


def parse_int_list(text: str) -> list[int]:
    """Parse zero-based plane syntax such as ``"0-2"`` or ``"0,5,10"``."""

    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_crop(text: str | None) -> tuple[int, int, int, int] | None:
    """Parse ``y0:y1,x0:x1`` crop syntax."""

    if text is None:
        return None
    ypart, xpart = text.split(",")
    y0, y1 = [int(v) for v in ypart.split(":")]
    x0, x1 = [int(v) for v in xpart.split(":")]
    return y0, y1, x0, x1


def crop_image(img: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return img
    y0, y1, x0, x1 = crop
    return img[y0:y1, x0:x1]


def make_thumbnail_png(
    *,
    tif: tifffile.TiffFile,
    page_indices: list[int],
    out_png: Path,
    max_pages: int = 24,
    downsample: int = 8,
) -> None:
    """Write a contact sheet of selected source pages for page-order QC."""

    import matplotlib.pyplot as plt

    show_pages = page_indices[:max_pages]
    n = len(show_pages)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, page_idx in zip(axes, show_pages):
        img = tif.pages[page_idx].asarray()
        thumb = img[::downsample, ::downsample]
        lo, hi = np.percentile(thumb, [1, 99.5])
        ax.imshow(thumb, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"page {page_idx}", fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_scanimage_plane_subset(
    input_tif: str | Path,
    output_tif: str | Path,
    *,
    planes: str | list[int],
    n_planes: int | None = None,
    repeats_per_plane: int | None = None,
    n_channels: int = 1,
    order: str = "slice_blocks",
    compression: str | None = None,
    crop: str | tuple[int, int, int, int] | None = None,
    make_thumbnail_qc: bool = False,
    dry_run: bool = False,
) -> dict:
    """Save selected z-plane pages from a large ScanImage TIFF.

    Parameters are intentionally similar to the CLI. Raw bit depth is preserved;
    optional TIFF compression is lossless.
    """

    input_tif = Path(input_tif)
    output_tif = Path(output_tif)
    plane_indices = parse_int_list(planes) if isinstance(planes, str) else sorted(set(planes))
    crop_tuple = parse_crop(crop) if isinstance(crop, str) else crop

    spec = read_tiff_stack_spec(
        input_tif,
        n_planes=n_planes,
        repeats_per_plane=repeats_per_plane,
        n_channels=n_channels,
        order=order,
        infer_from_descriptions=True,
    )
    page_indices = selected_scanimage_pages(plane_indices=plane_indices, spec=spec)

    with tifffile.TiffFile(input_tif) as tif:
        bad = [p for p in page_indices if p < 0 or p >= len(tif.pages)]
        if bad:
            raise ValueError(f"Selected page indices exceed TIFF length: {bad[:10]}")

        bytes_per_page = np.prod(tif.pages[0].shape) * np.dtype(tif.pages[0].dtype).itemsize
        estimated_bytes = int(bytes_per_page * len(page_indices))
        if crop_tuple is not None:
            y0, y1, x0, x1 = crop_tuple
            estimated_bytes = int(
                (y1 - y0)
                * (x1 - x0)
                * np.dtype(tif.pages[0].dtype).itemsize
                * len(page_indices)
            )

        manifest = {
            "input_tif": str(input_tif),
            "output_tif": str(output_tif),
            "planes_zero_based": plane_indices,
            "stack_spec": spec.to_dict(),
            "crop": crop_tuple,
            "compression": compression,
            "source_page_indices": page_indices,
            "estimated_raw_output_bytes": estimated_bytes,
        }

        print(f"Input pages: {len(tif.pages)}")
        print(f"First page shape: {tif.pages[0].shape}")
        print(f"First page dtype: {tif.pages[0].dtype}")
        print(f"Selected planes: {plane_indices}")
        print(f"Selected source pages: {page_indices[:20]} ... {page_indices[-20:]}")
        print(f"Number of pages to write: {len(page_indices)}")
        print(f"Estimated raw output size: {estimated_bytes / 1024**3:.2f} GiB")

        if dry_run:
            return manifest

        output_tif.parent.mkdir(parents=True, exist_ok=True)
        with tifffile.TiffWriter(output_tif, bigtiff=True) as writer:
            for out_i, src_i in enumerate(page_indices):
                page = tif.pages[src_i]
                img = crop_image(page.asarray(), crop_tuple)
                writer.write(
                    img,
                    photometric="minisblack",
                    compression=compression,
                    description=page.description,
                    metadata=None,
                )
                if (out_i + 1) % 10 == 0 or out_i == len(page_indices) - 1:
                    print(f"Wrote {out_i + 1}/{len(page_indices)} pages")

        manifest_path = output_tif.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifest["manifest_json"] = str(manifest_path)
        print(f"Wrote: {output_tif}")
        print(f"Wrote: {manifest_path}")

        if make_thumbnail_qc:
            thumb_path = output_tif.with_suffix(".thumbnail_qc.png")
            make_thumbnail_png(tif=tif, page_indices=page_indices, out_png=thumb_path)
            manifest["thumbnail_qc_png"] = str(thumb_path)
            print(f"Wrote: {thumb_path}")

    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Save selected z-plane blocks from a large ScanImage TIFF."
    )
    parser.add_argument("input_tif", type=Path)
    parser.add_argument("output_tif", type=Path)
    parser.add_argument("--planes", required=True, help='Zero-based planes, e.g. "0-2".')
    parser.add_argument("--n-planes", type=int, default=None)
    parser.add_argument("--repeats-per-plane", type=int, default=None)
    parser.add_argument("--n-channels", type=int, default=1)
    parser.add_argument(
        "--order",
        choices=["slice_blocks", "volume_interleaved"],
        default="slice_blocks",
    )
    parser.add_argument(
        "--compression",
        default=None,
        choices=[None, "zlib", "lzw"],
        help="Optional lossless TIFF compression.",
    )
    parser.add_argument("--crop", default=None, help="Optional y0:y1,x0:x1 crop.")
    parser.add_argument("--make-thumbnail-qc", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    save_scanimage_plane_subset(
        args.input_tif,
        args.output_tif,
        planes=args.planes,
        n_planes=args.n_planes,
        repeats_per_plane=args.repeats_per_plane,
        n_channels=args.n_channels,
        order=args.order,
        compression=args.compression,
        crop=args.crop,
        make_thumbnail_qc=args.make_thumbnail_qc,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
