from __future__ import annotations

import random

import numpy as np

from .structures import BBox, OcclusionRegion


def sample_cutout_regions(
    bbox: BBox,
    image_width: int,
    image_height: int,
    count_range: tuple[int, int],
    size_ratio_range: tuple[float, float],
    rng: random.Random,
) -> list[OcclusionRegion]:
    min_count, max_count = count_range
    count = rng.randint(int(min_count), int(max_count))
    regions: list[OcclusionRegion] = []
    box_width = max(1.0, bbox.width)
    box_height = max(1.0, bbox.height)
    margin_x = box_width * 0.15
    margin_y = box_height * 0.15
    for _ in range(count):
        width_ratio = rng.uniform(*size_ratio_range)
        height_ratio = rng.uniform(*size_ratio_range)
        region_width = max(4.0, box_width * width_ratio)
        region_height = max(4.0, box_height * height_ratio)
        center_x = rng.uniform(max(0.0, bbox.x1 - margin_x), min(image_width - 1.0, bbox.x2 + margin_x))
        center_y = rng.uniform(max(0.0, bbox.y1 - margin_y), min(image_height - 1.0, bbox.y2 + margin_y))
        x1 = max(0.0, center_x - region_width / 2.0)
        y1 = max(0.0, center_y - region_height / 2.0)
        x2 = min(image_width - 1.0, x1 + region_width)
        y2 = min(image_height - 1.0, y1 + region_height)
        regions.append(OcclusionRegion(x1=x1, y1=y1, x2=x2, y2=y2))
    return regions


def apply_cutout(
    image: np.ndarray,
    regions: list[OcclusionRegion],
    *,
    fill_mode: str = "mean",
    rng: random.Random,
) -> np.ndarray:
    result = image.copy()
    mean_color = tuple(int(round(value)) for value in image.mean(axis=(0, 1)))
    for region in regions:
        x1 = int(round(region.x1))
        y1 = int(round(region.y1))
        x2 = int(round(region.x2))
        y2 = int(round(region.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if fill_mode == "random":
            color = tuple(rng.randint(0, 255) for _ in range(3))
        else:
            color = mean_color
        result[y1:y2, x1:x2] = np.array(color, dtype=np.uint8)
    return result

