from __future__ import annotations

from .labels import bbox_out_of_frame_ratio, count_visible_keypoints
from .structures import BBox, ObjectAnnotation, SampleMetrics


def validate_augmented_sample(
    annotation: ObjectAnnotation,
    raw_bbox: BBox,
    *,
    image_width: int,
    image_height: int,
    source_visible_keypoints: int,
    filter_cfg: dict,
) -> tuple[SampleMetrics, str | None]:
    bbox = annotation.bbox
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        return SampleMetrics(0, bbox.width, bbox.height, 0.0, 1.0), "bbox_non_positive"

    bbox_area_ratio = bbox.area / float(image_width * image_height)
    visible_keypoints = count_visible_keypoints(annotation.keypoints)
    out_ratio = bbox_out_of_frame_ratio(raw_bbox, bbox)

    metrics = SampleMetrics(
        visible_keypoints=visible_keypoints,
        bbox_width_px=bbox.width,
        bbox_height_px=bbox.height,
        bbox_area_ratio=bbox_area_ratio,
        out_of_frame_ratio=out_ratio,
    )

    if bbox.width < float(filter_cfg["min_bbox_width_px"]):
        return metrics, "bbox_too_narrow"
    if bbox.height < float(filter_cfg["min_bbox_height_px"]):
        return metrics, "bbox_too_short"
    if bbox_area_ratio < float(filter_cfg["min_bbox_area_ratio"]):
        return metrics, "bbox_area_ratio_below_threshold"
    required_visible_keypoints = min(int(filter_cfg["min_visible_keypoints"]), int(source_visible_keypoints))
    if visible_keypoints < required_visible_keypoints:
        return metrics, "visible_keypoints_below_threshold"
    if out_ratio > float(filter_cfg["max_out_of_frame_ratio"]):
        return metrics, "bbox_out_of_frame_ratio_too_high"
    return metrics, None
