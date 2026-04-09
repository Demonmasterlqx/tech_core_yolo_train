from __future__ import annotations

import math
import random
from typing import Any

import cv2
import numpy as np

from .structures import BBox, Keypoint, ObjectAnnotation


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clip_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    clipped_x1 = int(math.floor(clamp(x1, 0.0, image_width - 1.0)))
    clipped_y1 = int(math.floor(clamp(y1, 0.0, image_height - 1.0)))
    clipped_x2 = int(math.ceil(clamp(x2, 1.0, image_width)))
    clipped_y2 = int(math.ceil(clamp(y2, 1.0, image_height)))
    if clipped_x2 <= clipped_x1:
        clipped_x2 = min(image_width, clipped_x1 + 1)
    if clipped_y2 <= clipped_y1:
        clipped_y2 = min(image_height, clipped_y1 + 1)
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def inscribed_rect(region_width: int, region_height: int, aspect_ratio: float) -> tuple[int, int]:
    if region_width <= 0 or region_height <= 0:
        return (0, 0)
    width = region_width
    height = int(round(width / aspect_ratio))
    if height > region_height:
        height = region_height
        width = int(round(height * aspect_ratio))
    return max(0, width), max(0, height)


def build_feather_mask(height: int, width: int, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return np.ones((height, width, 1), dtype=np.float32)
    effective = max(1, min(feather_px, width // 2 if width > 1 else 1, height // 2 if height > 1 else 1))
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    x_dist = np.minimum(x, (width - 1) - x)
    y_dist = np.minimum(y, (height - 1) - y)

    def alpha_curve(distance: np.ndarray) -> np.ndarray:
        inside = np.clip(distance / float(effective), 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * inside)

    alpha_x = alpha_curve(x_dist)
    alpha_y = alpha_curve(y_dist)
    mask = np.outer(alpha_y, alpha_x).astype(np.float32)
    return mask[..., None]


def padded_bbox_rect(
    bbox: BBox,
    exclusion_margin_ratio: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    margin = max(bbox.width, bbox.height) * exclusion_margin_ratio
    return clip_rect(
        bbox.x1 - margin,
        bbox.y1 - margin,
        bbox.x2 + margin,
        bbox.y2 + margin,
        image_width,
        image_height,
    )


def rects_overlap(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def select_same_image_background_patch(
    image: np.ndarray,
    bbox: BBox,
    *,
    exclusion_margin_ratio: float,
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    image_height, image_width = image.shape[:2]
    aspect_ratio = image_width / float(image_height)
    fx1, fy1, fx2, fy2 = padded_bbox_rect(
        bbox,
        exclusion_margin_ratio=exclusion_margin_ratio,
        image_width=image_width,
        image_height=image_height,
    )

    regions = [
        (0, 0, image_width, fy1),
        (0, fy2, image_width, image_height),
        (0, 0, fx1, image_height),
        (fx2, 0, image_width, image_height),
    ]
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for rx1, ry1, rx2, ry2 in regions:
        region_width = rx2 - rx1
        region_height = ry2 - ry1
        crop_width, crop_height = inscribed_rect(region_width, region_height, aspect_ratio)
        if crop_width < 32 or crop_height < 32:
            continue
        max_x1 = rx2 - crop_width
        max_y1 = ry2 - crop_height
        crop_x1 = rx1 if max_x1 <= rx1 else rng.randint(rx1, max_x1)
        crop_y1 = ry1 if max_y1 <= ry1 else rng.randint(ry1, max_y1)
        rect = (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height)
        if rects_overlap(rect, (fx1, fy1, fx2, fy2)):
            continue
        candidates.append((crop_width * crop_height, rect))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, rect = rng.choice(candidates[: min(4, len(candidates))])
    crop_x1, crop_y1, crop_x2, crop_y2 = rect
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    resized = cv2.resize(crop, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    return resized, rect


def visible_keypoints_in_bounds(annotation: ObjectAnnotation, image_width: int, image_height: int) -> bool:
    for keypoint in annotation.keypoints:
        if keypoint.v <= 0:
            continue
        if not (0.0 <= keypoint.x < image_width and 0.0 <= keypoint.y < image_height):
            return False
    return True


def apply_same_image_object_scale(
    image: np.ndarray,
    annotation: ObjectAnnotation,
    op_cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[np.ndarray, ObjectAnnotation, dict[str, Any]]:
    image_height, image_width = image.shape[:2]
    bbox = annotation.bbox
    bbox_center_x, bbox_center_y = bbox.center
    context_lo, context_hi = op_cfg["context_scale_range"]
    resize_lo, resize_hi = op_cfg["resize_scale_range"]
    resize_scale = rng.uniform(float(resize_lo), float(resize_hi))
    context_scale = rng.uniform(float(context_lo), float(context_hi))
    min_source_context_px = float(op_cfg.get("min_source_context_px", 32.0))

    crop_width = max(min_source_context_px, bbox.width * context_scale)
    crop_height = max(min_source_context_px, bbox.height * context_scale)
    crop_x1, crop_y1, crop_x2, crop_y2 = clip_rect(
        bbox_center_x - crop_width / 2.0,
        bbox_center_y - crop_height / 2.0,
        bbox_center_x + crop_width / 2.0,
        bbox_center_y + crop_height / 2.0,
        image_width,
        image_height,
    )
    roi = image[crop_y1:crop_y2, crop_x1:crop_x2]
    roi_height, roi_width = roi.shape[:2]
    if roi_width < 2 or roi_height < 2:
        raise ValueError("object_scale_source_roi_too_small")

    max_fit_scale = min((image_width - 2.0) / roi_width, (image_height - 2.0) / roi_height)
    if max_fit_scale <= 0.0:
        raise ValueError("object_scale_no_fit_scale")
    if resize_scale > max_fit_scale:
        resize_scale = max_fit_scale * 0.98
    if resize_scale <= 0.0:
        raise ValueError("object_scale_non_positive_resize_scale")

    resized_width = max(2, int(round(roi_width * resize_scale)))
    resized_height = max(2, int(round(roi_height * resize_scale)))
    if resized_width >= image_width or resized_height >= image_height:
        raise ValueError("object_scale_resized_roi_does_not_fit_canvas")

    background = select_same_image_background_patch(
        image,
        bbox,
        exclusion_margin_ratio=float(op_cfg["exclusion_margin_ratio"]),
        rng=rng,
    )
    if background is None:
        raise ValueError("object_scale_no_valid_background_patch")
    background_canvas, background_rect = background

    interpolation = cv2.INTER_AREA if resize_scale <= 1.0 else cv2.INTER_LINEAR
    resized_roi = cv2.resize(roi, (resized_width, resized_height), interpolation=interpolation)

    local_bbox_center_x = bbox_center_x - crop_x1
    local_bbox_center_y = bbox_center_y - crop_y1
    max_jitter_x = bbox.width * float(op_cfg.get("center_jitter_ratio", 0.0))
    max_jitter_y = bbox.height * float(op_cfg.get("center_jitter_ratio", 0.0))
    desired_center_x = bbox_center_x + rng.uniform(-max_jitter_x, max_jitter_x)
    desired_center_y = bbox_center_y + rng.uniform(-max_jitter_y, max_jitter_y)

    paste_x = int(round(desired_center_x - local_bbox_center_x * resize_scale))
    paste_y = int(round(desired_center_y - local_bbox_center_y * resize_scale))
    paste_x = int(clamp(float(paste_x), 0.0, float(image_width - resized_width)))
    paste_y = int(clamp(float(paste_y), 0.0, float(image_height - resized_height)))

    feather_px = int(op_cfg.get("feather_px", 0))
    alpha = build_feather_mask(resized_height, resized_width, feather_px)
    canvas = background_canvas.astype(np.float32)
    patch = resized_roi.astype(np.float32)
    view = canvas[paste_y : paste_y + resized_height, paste_x : paste_x + resized_width]
    view[:] = patch * alpha + view * (1.0 - alpha)
    transformed_image = np.clip(canvas, 0, 255).astype(np.uint8)

    transformed_center_x = local_bbox_center_x * resize_scale + paste_x
    transformed_center_y = local_bbox_center_y * resize_scale + paste_y
    transformed_width = bbox.width * resize_scale
    transformed_height = bbox.height * resize_scale
    transformed_bbox = BBox(
        x1=transformed_center_x - transformed_width / 2.0,
        y1=transformed_center_y - transformed_height / 2.0,
        x2=transformed_center_x + transformed_width / 2.0,
        y2=transformed_center_y + transformed_height / 2.0,
    )
    transformed_keypoints = [
        Keypoint(
            x=(keypoint.x - crop_x1) * resize_scale + paste_x if keypoint.v > 0 else 0.0,
            y=(keypoint.y - crop_y1) * resize_scale + paste_y if keypoint.v > 0 else 0.0,
            v=keypoint.v,
        )
        for keypoint in annotation.keypoints
    ]
    transformed_annotation = ObjectAnnotation(
        class_id=annotation.class_id,
        bbox=transformed_bbox,
        keypoints=transformed_keypoints,
    )

    if transformed_bbox.x1 < 0.0 or transformed_bbox.y1 < 0.0 or transformed_bbox.x2 > image_width or transformed_bbox.y2 > image_height:
        raise ValueError("object_scale_bbox_out_of_frame")
    if not visible_keypoints_in_bounds(transformed_annotation, image_width=image_width, image_height=image_height):
        raise ValueError("object_scale_visible_keypoint_out_of_frame")

    meta = {
        "object_scale_applied": True,
        "context_scale": context_scale,
        "resize_scale": resize_scale,
        "paste_x": paste_x,
        "paste_y": paste_y,
        "source_crop_rect": {
            "x1": crop_x1,
            "y1": crop_y1,
            "x2": crop_x2,
            "y2": crop_y2,
        },
        "background_rect": {
            "x1": background_rect[0],
            "y1": background_rect[1],
            "x2": background_rect[2],
            "y2": background_rect[3],
        },
    }
    return transformed_image, transformed_annotation, meta
