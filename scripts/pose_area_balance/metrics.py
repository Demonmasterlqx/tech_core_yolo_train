from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pose_offline_aug.object_scale import input_area_ratio
from pose_offline_aug.structures import ObjectAnnotation


@dataclass(frozen=True)
class AreaBin:
    index: int
    lower: float
    upper: float

    @property
    def label(self) -> str:
        return f"[{self.lower:.3f},{self.upper:.3f})"

    def contains(self, value: float) -> bool:
        return self.lower <= value < self.upper


def build_area_bins(min_area_ratio: float, max_area_ratio: float, bin_count: int) -> list[AreaBin]:
    if min_area_ratio <= 0.0 or max_area_ratio <= min_area_ratio:
        raise ValueError("Area ratio range must satisfy 0 < min < max.")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")

    width = (max_area_ratio - min_area_ratio) / float(bin_count)
    bins: list[AreaBin] = []
    for index in range(bin_count):
        lower = min_area_ratio + width * index
        upper = min_area_ratio + width * (index + 1)
        bins.append(AreaBin(index=index, lower=lower, upper=upper))
    return bins


def sample_target_ratio(area_bin: AreaBin, margin_ratio: float) -> tuple[float, float]:
    width = area_bin.upper - area_bin.lower
    if width <= 0.0:
        return area_bin.lower, area_bin.upper
    margin = max(0.0, min(width * margin_ratio, width * 0.45))
    low = area_bin.lower + margin
    high = area_bin.upper - margin
    if high <= low:
        center = (area_bin.lower + area_bin.upper) / 2.0
        return center, center
    return low, high


def bin_for_ratio(value: float, bins: list[AreaBin]) -> AreaBin | None:
    for area_bin in bins:
        if area_bin.contains(value):
            return area_bin
    return None


def build_bin_histogram(values: Iterable[float], bins: list[AreaBin]) -> dict[str, int]:
    histogram = {area_bin.label: 0 for area_bin in bins}
    for value in values:
        matched = bin_for_ratio(value, bins)
        if matched is not None:
            histogram[matched.label] += 1
    return histogram


def annotation_area_ratio_after_letterbox(
    annotation: ObjectAnnotation,
    image_width: int,
    image_height: int,
    imgsz: float,
) -> float:
    return input_area_ratio(
        bbox_width_px=annotation.bbox.width,
        bbox_height_px=annotation.bbox.height,
        image_width=image_width,
        image_height=image_height,
        input_size_px=imgsz,
    )
