"""Metadata helpers for ScanImage/Bruker TIFF stacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable


_KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z0-9_\.]+)\s*=\s*(.*?)\s*$")


@dataclass(frozen=True)
class ScanImageFrameDescription:
    """Small parsed subset of a ScanImage per-frame ImageDescription."""

    frame_number: int | None = None
    acquisition_number: int | None = None
    frame_number_acquisition: int | None = None
    frame_timestamp_sec: float | None = None
    acq_trigger_timestamp_sec: float | None = None


@dataclass(frozen=True)
class ScanImageStackSpec:
    """Layout information needed to index a ScanImage TIFF hyperstack.

    Plane and channel indices are zero-based in Python. Channel numbering in
    filenames/metadata can still be displayed as one-based.
    """

    n_pages: int
    image_shape: tuple[int, int]
    dtype: str
    n_planes: int
    repeats_per_plane: int
    n_channels: int = 1
    order: str = "slice_blocks"

    @property
    def pages_per_plane(self) -> int:
        return self.repeats_per_plane * self.n_channels

    @property
    def pages_per_volume_repeat(self) -> int:
        return self.n_planes * self.n_channels

    def to_dict(self) -> dict:
        out = asdict(self)
        out["image_shape"] = list(self.image_shape)
        return out


def parse_scanimage_description(description: str | None) -> ScanImageFrameDescription:
    """Parse common scalar fields from a ScanImage ImageDescription string."""

    if not description:
        return ScanImageFrameDescription()

    values: dict[str, str] = {}
    for line in description.splitlines():
        match = _KEY_VALUE_RE.match(line)
        if match:
            values[match.group(1)] = match.group(2)

    def as_int(key: str) -> int | None:
        value = values.get(key)
        if value is None:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def as_float(key: str) -> float | None:
        value = values.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    return ScanImageFrameDescription(
        frame_number=as_int("frameNumbers"),
        acquisition_number=as_int("acquisitionNumbers"),
        frame_number_acquisition=as_int("frameNumberAcquisition"),
        frame_timestamp_sec=as_float("frameTimestamps_sec"),
        acq_trigger_timestamp_sec=as_float("acqTriggerTimestamps_sec"),
    )


def infer_scanimage_plane_count(
    descriptions: Iterable[str | None], *, n_channels: int = 1
) -> tuple[int | None, int | None]:
    """Infer (n_planes, repeats_per_plane) from ScanImage frame descriptions.

    This works well for your current one-channel stack, where ScanImage writes
    ``acquisitionNumbers`` as the z-plane/acquisition index and
    ``frameNumberAcquisition`` as the repeat/frame within that acquisition.
    For multi-channel stacks, still pass n_channels explicitly and verify the
    result using the thumbnail/QC outputs.
    """

    acqs: list[int] = []
    frame_nums_within_acq: list[int] = []

    for desc in descriptions:
        parsed = parse_scanimage_description(desc)
        if parsed.acquisition_number is not None:
            acqs.append(parsed.acquisition_number)
        if parsed.frame_number_acquisition is not None:
            frame_nums_within_acq.append(parsed.frame_number_acquisition)

    if not acqs or not frame_nums_within_acq:
        return None, None

    # Acquisition numbers are one-based in ScanImage metadata.
    n_planes = len(set(acqs))
    repeats_per_plane = max(frame_nums_within_acq)

    # If there are multiple channels as pages, each frameNumberAcquisition may
    # appear once per channel. The max still gives repeats per plane.
    return n_planes, repeats_per_plane
