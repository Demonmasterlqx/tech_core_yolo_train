from __future__ import annotations

from typing import Iterable

import numpy as np

from .geometry import transform_xy
from .structures import BBox, Keypoint, ObjectAnnotation, OcclusionRegion


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bbox_from_points(points: Iterable[tuple[float, float]]) -> BBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def bbox_corners(bbox: BBox) -> list[tuple[float, float]]:
    return [
        (bbox.x1, bbox.y1),
        (bbox.x2, bbox.y1),
        (bbox.x2, bbox.y2),
        (bbox.x1, bbox.y2),
    ]


def transform_bbox(bbox: BBox, matrix: np.ndarray) -> BBox:
    transformed = [transform_xy(matrix, x_coord, y_coord) for x_coord, y_coord in bbox_corners(bbox)]
    return bbox_from_points(transformed)


def clip_bbox_to_image(bbox: BBox, image_width: int, image_height: int) -> BBox:
    return BBox(
        x1=clamp(bbox.x1, 0.0, image_width - 1.0),
        y1=clamp(bbox.y1, 0.0, image_height - 1.0),
        x2=clamp(bbox.x2, 1.0, image_width - 1.0),
        y2=clamp(bbox.y2, 1.0, image_height - 1.0),
    )


def bbox_out_of_frame_ratio(raw_bbox: BBox, clipped_bbox: BBox) -> float:
    raw_area = raw_bbox.area
    if raw_area <= 0.0:
        return 1.0
    kept_area = max(0.0, clipped_bbox.area)
    return max(0.0, min(1.0, 1.0 - kept_area / raw_area))


def point_in_bounds(x_coord: float, y_coord: float, image_width: int, image_height: int) -> bool:
    return 0.0 <= x_coord < image_width and 0.0 <= y_coord < image_height


def point_in_valid_mask(valid_mask: np.ndarray, x_coord: float, y_coord: float) -> bool:
    height, width = valid_mask.shape[:2]
    if not point_in_bounds(x_coord, y_coord, width, height):
        return False
    x_index = int(round(clamp(x_coord, 0.0, width - 1.0)))
    y_index = int(round(clamp(y_coord, 0.0, height - 1.0)))
    return int(valid_mask[y_index, x_index]) > 0


def point_in_occlusion(regions: list[OcclusionRegion], x_coord: float, y_coord: float) -> bool:
    return any(region.contains(x_coord, y_coord) for region in regions)


def transform_keypoints(
    keypoints: list[Keypoint],
    matrix: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    valid_mask: np.ndarray,
    occlusion_regions: list[OcclusionRegion],
) -> list[Keypoint]:
    transformed: list[Keypoint] = []
    for keypoint in keypoints:
        new_x, new_y = transform_xy(matrix, keypoint.x, keypoint.y)
        if keypoint.v <= 0:
            transformed.append(Keypoint(x=0.0, y=0.0, v=0))
            continue

        visible = point_in_bounds(new_x, new_y, image_width, image_height)
        valid = visible and point_in_valid_mask(valid_mask, new_x, new_y)
        occluded = visible and point_in_occlusion(occlusion_regions, new_x, new_y)
        if not visible or not valid:
            # Ultralytics 8.4.30 supervises all keypoints with v != 0, so
            # points that leave the image must be dropped rather than clipped.
            transformed.append(Keypoint(x=0.0, y=0.0, v=0))
        elif occluded:
            transformed.append(Keypoint(x=new_x, y=new_y, v=1))
        elif keypoint.v >= 2:
            transformed.append(Keypoint(x=new_x, y=new_y, v=2))
        else:
            transformed.append(Keypoint(x=new_x, y=new_y, v=1))
    return transformed


def apply_geometry_to_annotation(
    annotation: ObjectAnnotation,
    matrix: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    valid_mask: np.ndarray,
    occlusion_regions: list[OcclusionRegion],
) -> tuple[ObjectAnnotation, BBox]:
    raw_bbox = transform_bbox(annotation.bbox, matrix)
    clipped_bbox = clip_bbox_to_image(raw_bbox, image_width=image_width, image_height=image_height)
    keypoints = transform_keypoints(
        annotation.keypoints,
        matrix,
        image_width=image_width,
        image_height=image_height,
        valid_mask=valid_mask,
        occlusion_regions=occlusion_regions,
    )
    return ObjectAnnotation(class_id=annotation.class_id, bbox=clipped_bbox, keypoints=keypoints), raw_bbox


def count_visible_keypoints(keypoints: list[Keypoint]) -> int:
    return sum(1 for keypoint in keypoints if keypoint.v >= 2)
