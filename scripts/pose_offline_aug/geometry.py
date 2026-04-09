from __future__ import annotations

import math
import random
from typing import Iterable

import cv2
import numpy as np

from .structures import BBox


BORDER_MODES = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT,
    "reflect101": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
}


def identity_matrix() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


def compose_matrices(matrices: Iterable[np.ndarray]) -> np.ndarray:
    result = identity_matrix()
    for matrix in matrices:
        result = matrix @ result
    return result.astype(np.float32)


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def scale_about(center_x: float, center_y: float, scale: float) -> np.ndarray:
    return np.array(
        [
            [scale, 0.0, center_x - scale * center_x],
            [0.0, scale, center_y - scale * center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_about(center_x: float, center_y: float, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    return np.array(
        [
            [cos_theta, -sin_theta, center_x - cos_theta * center_x + sin_theta * center_y],
            [sin_theta, cos_theta, center_y - sin_theta * center_x - cos_theta * center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def shear_about(center_x: float, center_y: float, shear_x_deg: float, shear_y_deg: float) -> np.ndarray:
    shear_x = math.tan(math.radians(shear_x_deg))
    shear_y = math.tan(math.radians(shear_y_deg))
    to_origin = translation_matrix(-center_x, -center_y)
    shear = np.array(
        [
            [1.0, shear_x, 0.0],
            [shear_y, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    back = translation_matrix(center_x, center_y)
    return (back @ shear @ to_origin).astype(np.float32)


def perspective_matrix(image_width: int, image_height: int, strength: float, rng: random.Random) -> np.ndarray:
    jitter_x = strength * image_width
    jitter_y = strength * image_height
    source = np.array(
        [
            [0.0, 0.0],
            [image_width - 1.0, 0.0],
            [image_width - 1.0, image_height - 1.0],
            [0.0, image_height - 1.0],
        ],
        dtype=np.float32,
    )
    destination = source.copy()
    for index in range(4):
        destination[index, 0] += rng.uniform(-jitter_x, jitter_x)
        destination[index, 1] += rng.uniform(-jitter_y, jitter_y)
    return cv2.getPerspectiveTransform(source, destination).astype(np.float32)


def crop_and_resize_matrix(
    image_width: int,
    image_height: int,
    crop_scale: float,
    jitter_x_ratio: float,
    jitter_y_ratio: float,
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, float]]:
    crop_width = max(2.0, image_width * crop_scale)
    crop_height = max(2.0, image_height * crop_scale)
    max_shift_x = max(0.0, (image_width - crop_width) * jitter_x_ratio)
    max_shift_y = max(0.0, (image_height - crop_height) * jitter_y_ratio)
    center_x = image_width / 2.0 + rng.uniform(-max_shift_x, max_shift_x)
    center_y = image_height / 2.0 + rng.uniform(-max_shift_y, max_shift_y)
    crop_x1 = min(max(0.0, center_x - crop_width / 2.0), image_width - crop_width)
    crop_y1 = min(max(0.0, center_y - crop_height / 2.0), image_height - crop_height)

    scale_x = image_width / crop_width
    scale_y = image_height / crop_height
    matrix = np.array(
        [
            [scale_x, 0.0, -crop_x1 * scale_x],
            [0.0, scale_y, -crop_y1 * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    meta = {
        "crop_scale": crop_scale,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_width": crop_width,
        "crop_height": crop_height,
    }
    return matrix, meta


def bbox_crop_and_resize_matrix(
    image_width: int,
    image_height: int,
    bbox: BBox,
    crop_scale: float,
    jitter_x_bbox_ratio: float,
    jitter_y_bbox_ratio: float,
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, float]]:
    crop_width = max(2.0, image_width * crop_scale)
    crop_height = max(2.0, image_height * crop_scale)
    bbox_center_x, bbox_center_y = bbox.center
    max_shift_x = max(0.0, bbox.width * jitter_x_bbox_ratio)
    max_shift_y = max(0.0, bbox.height * jitter_y_bbox_ratio)
    center_x = bbox_center_x + rng.uniform(-max_shift_x, max_shift_x)
    center_y = bbox_center_y + rng.uniform(-max_shift_y, max_shift_y)
    center_x = min(max(crop_width / 2.0, center_x), image_width - crop_width / 2.0)
    center_y = min(max(crop_height / 2.0, center_y), image_height - crop_height / 2.0)
    crop_x1 = min(max(0.0, center_x - crop_width / 2.0), image_width - crop_width)
    crop_y1 = min(max(0.0, center_y - crop_height / 2.0), image_height - crop_height)

    scale_x = image_width / crop_width
    scale_y = image_height / crop_height
    matrix = np.array(
        [
            [scale_x, 0.0, -crop_x1 * scale_x],
            [0.0, scale_y, -crop_y1 * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    meta = {
        "bbox_crop_scale": crop_scale,
        "bbox_crop_x1": crop_x1,
        "bbox_crop_y1": crop_y1,
        "bbox_crop_width": crop_width,
        "bbox_crop_height": crop_height,
        "bbox_crop_center_x": center_x,
        "bbox_crop_center_y": center_y,
    }
    return matrix, meta


def warp_image_and_mask(
    image: np.ndarray,
    matrix: np.ndarray,
    *,
    border_mode: str = "reflect101",
) -> tuple[np.ndarray, np.ndarray]:
    image_height, image_width = image.shape[:2]
    border = BORDER_MODES.get(border_mode.lower())
    if border is None:
        raise ValueError(f"Unsupported border mode: {border_mode}")

    warped = cv2.warpPerspective(
        image,
        matrix,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=border,
        borderValue=(0, 0, 0),
    )
    valid_mask = cv2.warpPerspective(
        np.full((image_height, image_width), 255, dtype=np.uint8),
        matrix,
        (image_width, image_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, valid_mask


def transform_xy(matrix: np.ndarray, x_coord: float, y_coord: float) -> tuple[float, float]:
    point = np.array([x_coord, y_coord, 1.0], dtype=np.float32)
    transformed = matrix @ point
    if transformed[2] == 0.0:
        raise ValueError("Encountered zero homogeneous coordinate during point transform.")
    return float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2])
