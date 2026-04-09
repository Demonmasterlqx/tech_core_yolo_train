from __future__ import annotations

import random

import numpy as np

from .structures import BBox, OcclusionRegion


def clip_region(region: OcclusionRegion, image_width: int, image_height: int) -> OcclusionRegion | None:
    x1 = max(0.0, min(region.x1, image_width - 1.0))
    y1 = max(0.0, min(region.y1, image_height - 1.0))
    x2 = max(0.0, min(region.x2, image_width - 1.0))
    y2 = max(0.0, min(region.y2, image_height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return OcclusionRegion(x1=x1, y1=y1, x2=x2, y2=y2, kind=region.kind)


def bounded_center(low: float, high: float, fallback: float, rng: random.Random) -> float:
    if high <= low:
        return fallback
    return rng.uniform(low, high)


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
        clipped = clip_region(OcclusionRegion(x1=x1, y1=y1, x2=x2, y2=y2, kind="cutout"), image_width, image_height)
        if clipped is not None:
            regions.append(clipped)
    return regions


def sample_edge_cutout_regions(
    bbox: BBox,
    image_width: int,
    image_height: int,
    count_range: tuple[int, int],
    thickness_ratio_range: tuple[float, float],
    length_ratio_range: tuple[float, float],
    rng: random.Random,
) -> list[OcclusionRegion]:
    min_count, max_count = count_range
    count = rng.randint(int(min_count), int(max_count))
    regions: list[OcclusionRegion] = []
    box_width = max(1.0, bbox.width)
    box_height = max(1.0, bbox.height)
    for _ in range(count):
        edge = rng.choice(["left", "right", "top", "bottom"])
        thickness_ratio = rng.uniform(*thickness_ratio_range)
        length_ratio = rng.uniform(*length_ratio_range)
        if edge in {"left", "right"}:
            region_width = max(4.0, box_width * thickness_ratio)
            region_height = max(4.0, box_height * length_ratio)
            center_y = bounded_center(
                bbox.y1 + region_height / 2.0,
                bbox.y2 - region_height / 2.0,
                (bbox.y1 + bbox.y2) / 2.0,
                rng,
            )
            if edge == "left":
                x1 = bbox.x1
                x2 = bbox.x1 + region_width
            else:
                x2 = bbox.x2
                x1 = bbox.x2 - region_width
            y1 = center_y - region_height / 2.0
            y2 = center_y + region_height / 2.0
        else:
            region_width = max(4.0, box_width * length_ratio)
            region_height = max(4.0, box_height * thickness_ratio)
            center_x = bounded_center(
                bbox.x1 + region_width / 2.0,
                bbox.x2 - region_width / 2.0,
                (bbox.x1 + bbox.x2) / 2.0,
                rng,
            )
            if edge == "top":
                y1 = bbox.y1
                y2 = bbox.y1 + region_height
            else:
                y2 = bbox.y2
                y1 = bbox.y2 - region_height
            x1 = center_x - region_width / 2.0
            x2 = center_x + region_width / 2.0

        clipped = clip_region(
            OcclusionRegion(x1=x1, y1=y1, x2=x2, y2=y2, kind="edge_cutout"),
            image_width,
            image_height,
        )
        if clipped is not None:
            regions.append(clipped)
    return regions


def sample_corner_cutout_regions(
    bbox: BBox,
    image_width: int,
    image_height: int,
    size_ratio_range: tuple[float, float],
    rng: random.Random,
) -> list[OcclusionRegion]:
    box_width = max(1.0, bbox.width)
    box_height = max(1.0, bbox.height)
    width_ratio = rng.uniform(*size_ratio_range)
    height_ratio = rng.uniform(*size_ratio_range)
    region_width = max(4.0, box_width * width_ratio)
    region_height = max(4.0, box_height * height_ratio)
    corner = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    if corner == "top_left":
        x1, y1 = bbox.x1, bbox.y1
    elif corner == "top_right":
        x1, y1 = bbox.x2 - region_width, bbox.y1
    elif corner == "bottom_left":
        x1, y1 = bbox.x1, bbox.y2 - region_height
    else:
        x1, y1 = bbox.x2 - region_width, bbox.y2 - region_height

    clipped = clip_region(
        OcclusionRegion(
            x1=x1,
            y1=y1,
            x2=x1 + region_width,
            y2=y1 + region_height,
            kind="corner_cutout",
        ),
        image_width,
        image_height,
    )
    return [clipped] if clipped is not None else []


def sample_local_patch(image: np.ndarray, patch_height: int, patch_width: int, rng: random.Random) -> np.ndarray | None:
    image_height, image_width = image.shape[:2]
    if patch_height <= 0 or patch_width <= 0 or patch_height > image_height or patch_width > image_width:
        return None
    max_x = image_width - patch_width
    max_y = image_height - patch_height
    src_x1 = rng.randint(0, max_x) if max_x > 0 else 0
    src_y1 = rng.randint(0, max_y) if max_y > 0 else 0
    src_x2 = src_x1 + patch_width
    src_y2 = src_y1 + patch_height
    return image[src_y1:src_y2, src_x1:src_x2].copy()


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
            result[y1:y2, x1:x2] = np.array(color, dtype=np.uint8)
        elif fill_mode == "local_patch":
            patch = sample_local_patch(image, y2 - y1, x2 - x1, rng)
            if patch is None or patch.shape[:2] != (y2 - y1, x2 - x1):
                result[y1:y2, x1:x2] = np.array(mean_color, dtype=np.uint8)
            else:
                result[y1:y2, x1:x2] = patch
        else:
            color = mean_color
            result[y1:y2, x1:x2] = np.array(color, dtype=np.uint8)
    return result
