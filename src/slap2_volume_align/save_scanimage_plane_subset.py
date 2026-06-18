from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


def parse_int_list(text: str) -> list[int]:
    """
    Parse comma/range syntax like:
        "0,1,2"
        "10,11,12"
        "0-2"
    Plane indices are zero-based.
    """
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


def selected_page_indices(
    *,
    plane_indices: Iterable[int],
    n_planes: int,
    repeats_per_plane: int,
    n_channels: int,
    order: str,
) -> list[int]:
    """
    Return original TIFF page indices for the requested z planes.

    Assumptions:
    - All channels are retained.
    - Channel pages are adjacent within each frame/repeat.
    - Plane indices are zero-based.
    """
    pages: list[int] = []

    if order == "slice_blocks":
        # z0 all repeats/channels, z1 all repeats/channels, ...
        pages_per_plane = repeats_per_plane * n_channels
        for z in plane_indices:
            start = z * pages_per_plane
            stop = start + pages_per_plane
            pages.extend(range(start, stop))

    elif order == "volume_interleaved":
        # repeat0 all z/channels, repeat1 all z/channels, ...
        pages_per_volume = n_planes * n_channels
        for rep in range(repeats_per_plane):
            volume_start = rep * pages_per_volume
            for z in plane_indices:
                start = volume_start + z * n_channels
                stop = start + n_channels
                pages.extend(range(start, stop))

    else:
        raise ValueError(f"Unknown order: {order}")

    return pages


def crop_image(img: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    """
    crop = (y0, y1, x0, x1)
    """
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
    """
    Make a quick contact sheet to check whether selected pages look like repeats
    of the same plane or different z planes.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save selected z-plane blocks from a large ScanImage TIFF."
    )
    parser.add_argument("input_tif", type=Path)
    parser.add_argument("output_tif", type=Path)

    parser.add_argument(
        "--planes",
        required=True,
        help='Zero-based plane indices, e.g. "0-2" or "20,21,22".',
    )
    parser.add_argument("--n-planes", type=int, default=177)
    parser.add_argument("--repeats-per-plane", type=int, default=20)
    parser.add_argument("--n-channels", type=int, default=1)

    parser.add_argument(
        "--order",
        choices=["slice_blocks", "volume_interleaved"],
        default="slice_blocks",
        help=(
            "slice_blocks: all repeats for z0, then all repeats for z1. "
            "volume_interleaved: all z planes for repeat0, then repeat1."
        ),
    )

    parser.add_argument(
        "--compression",
        default=None,
        choices=[None, "zlib", "lzw"],
        help=(
            "Optional lossless TIFF compression. "
            "None is fastest and safest. zlib/lzw preserve bit depth exactly."
        ),
    )

    parser.add_argument(
        "--crop",
        default=None,
        help=(
            "Optional spatial crop as y0:y1,x0:x1. "
            "Example: 512:1536,512:1536. "
            "Omit to preserve full 2048 x 2048 frames."
        ),
    )

    parser.add_argument(
        "--make-thumbnail-qc",
        action="store_true",
        help="Also write a PNG contact sheet of selected pages.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected page indices but do not write output TIFF.",
    )

    args = parser.parse_args()

    plane_indices = parse_int_list(args.planes)

    crop = None
    if args.crop is not None:
        ypart, xpart = args.crop.split(",")
        y0, y1 = [int(v) for v in ypart.split(":")]
        x0, x1 = [int(v) for v in xpart.split(":")]
        crop = (y0, y1, x0, x1)

    page_indices = selected_page_indices(
        plane_indices=plane_indices,
        n_planes=args.n_planes,
        repeats_per_plane=args.repeats_per_plane,
        n_channels=args.n_channels,
        order=args.order,
    )

    with tifffile.TiffFile(args.input_tif) as tif:
        n_pages = len(tif.pages)
        first_shape = tif.pages[0].shape
        first_dtype = tif.pages[0].dtype

        bad = [p for p in page_indices if p < 0 or p >= n_pages]
        if bad:
            raise ValueError(
                f"Selected page indices exceed TIFF length. "
                f"Bad examples: {bad[:10]}; TIFF has {n_pages} pages."
            )

        print(f"Input pages: {n_pages}")
        print(f"First page shape: {first_shape}")
        print(f"First page dtype: {first_dtype}")
        print(f"Selected planes: {plane_indices}")
        print(f"Selected source pages: {page_indices[:20]} ... {page_indices[-20:]}")
        print(f"Number of pages to write: {len(page_indices)}")

        bytes_per_page = np.prod(first_shape) * np.dtype(first_dtype).itemsize
        estimated_bytes = bytes_per_page * len(page_indices)
        print(f"Estimated raw output size: {estimated_bytes / 1024**3:.2f} GiB")

        if args.dry_run:
            return

        args.output_tif.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "input_tif": str(args.input_tif),
            "output_tif": str(args.output_tif),
            "planes_zero_based": plane_indices,
            "n_planes": args.n_planes,
            "repeats_per_plane": args.repeats_per_plane,
            "n_channels": args.n_channels,
            "order": args.order,
            "crop": crop,
            "compression": args.compression,
            "source_page_indices": page_indices,
            "source_shape": list(first_shape),
            "source_dtype": str(first_dtype),
        }

        with tifffile.TiffWriter(args.output_tif, bigtiff=True) as writer:
            for out_i, src_i in enumerate(page_indices):
                page = tif.pages[src_i]
                img = page.asarray()
                img = crop_image(img, crop)

                # Preserve original per-frame ScanImage description where possible.
                desc = page.description

                writer.write(
                    img,
                    photometric="minisblack",
                    compression=args.compression,
                    description=desc,
                    metadata=None,
                )

                if (out_i + 1) % 10 == 0 or out_i == len(page_indices) - 1:
                    print(f"Wrote {out_i + 1}/{len(page_indices)} pages")

        manifest_path = args.output_tif.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote: {args.output_tif}")
        print(f"Wrote: {manifest_path}")

        if args.make_thumbnail_qc:
            thumb_path = args.output_tif.with_suffix(".thumbnail_qc.png")
            make_thumbnail_png(
                tif=tif,
                page_indices=page_indices,
                out_png=thumb_path,
            )
            print(f"Wrote: {thumb_path}")


if __name__ == "__main__":
    main()