from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import appearance, geometry, occlusion
from .io import read_image_bgr
from .labels import apply_geometry_to_annotation, count_visible_keypoints
from .object_scale import apply_same_image_object_scale, apply_same_image_video_reframe
from .structures import AugmentRecord, ImageAnnotation, ObjectAnnotation, OcclusionRegion
from .validator import validate_augmented_sample


GEOMETRY_ORDER = ("scale", "rotate", "translate", "shear", "perspective", "bbox_crop", "crop")
APPEARANCE_ORDER = (
    "brightness_contrast",
    "gamma",
    "hsv",
    "rgb_shift",
    "dominant_channel",
    "channel_shuffle",
    "blur",
    "motion_blur",
    "noise",
    "jpeg",
    "clahe",
    "channel_gain",
    "shadow",
)
OCCLUSION_ORDER = ("cutout", "edge_cutout", "corner_cutout")


@dataclass(frozen=True)
class GeneratedSample:
    image: np.ndarray
    annotation: ObjectAnnotation
    record: AugmentRecord


def config_enabled(name: str, config: dict[str, Any]) -> bool:
    return bool(config.get(name, {}).get("enabled", False))


def choose_template(templates_cfg: dict[str, Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    names = list(templates_cfg)
    weights = [float(templates_cfg[name].get("weight", 1.0)) for name in names]
    chosen = rng.choices(names, weights=weights, k=1)[0]
    return chosen, templates_cfg[chosen]


def maybe_apply_special_geometry(
    image: np.ndarray,
    annotation: ObjectAnnotation,
    *,
    image_width: int,
    image_height: int,
    geometry_cfg: dict[str, Any],
    template_cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[np.ndarray, ObjectAnnotation, int, int, dict[str, Any]]:
    requested = set(template_cfg.get("geometry", []))
    meta: dict[str, Any] = {
        "object_scale_applied": False,
        "video_reframe_applied": False,
    }
    current_image = image
    current_annotation = annotation
    current_width = int(image_width)
    current_height = int(image_height)

    if "video_reframe" in requested and config_enabled("video_reframe", geometry_cfg):
        op_cfg = dict(geometry_cfg["video_reframe"])
        template_override = template_cfg.get("video_reframe")
        if isinstance(template_override, dict):
            op_cfg.update(template_override)
        if rng.random() <= float(op_cfg.get("prob", 1.0)):
            current_image, current_annotation, stage_meta = apply_same_image_video_reframe(
                current_image,
                current_annotation,
                op_cfg,
                rng,
            )
            current_height, current_width = current_image.shape[:2]
            meta.update(stage_meta)

    if "object_scale" in requested and config_enabled("object_scale", geometry_cfg):
        op_cfg = dict(geometry_cfg["object_scale"])
        template_override = template_cfg.get("object_scale")
        if isinstance(template_override, dict):
            op_cfg.update(template_override)
        if rng.random() <= float(op_cfg.get("prob", 1.0)):
            current_image, current_annotation, stage_meta = apply_same_image_object_scale(
                current_image,
                current_annotation,
                op_cfg,
                rng,
            )
            current_height, current_width = current_image.shape[:2]
            meta.update(stage_meta)

    return current_image, current_annotation, current_width, current_height, meta


def sample_geometry_matrix(
    annotation: ImageAnnotation,
    geometry_cfg: dict[str, Any],
    template_cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, Any]]:
    image_width = annotation.image_width
    image_height = annotation.image_height
    bbox_center = annotation.object_annotation.bbox.center
    requested = set(template_cfg.get("geometry", []))
    transforms: list[np.ndarray] = []
    meta: dict[str, Any] = {}

    for name in GEOMETRY_ORDER:
        if name not in requested or not config_enabled(name, geometry_cfg):
            continue
        op_cfg = geometry_cfg[name]
        if rng.random() > float(op_cfg.get("prob", 1.0)):
            continue

        if name == "scale":
            scale = rng.uniform(float(op_cfg["range"][0]), float(op_cfg["range"][1]))
            transforms.append(geometry.scale_about(bbox_center[0], bbox_center[1], scale))
            meta["scale"] = scale
        elif name == "rotate":
            degrees = rng.uniform(float(op_cfg["range_deg"][0]), float(op_cfg["range_deg"][1]))
            transforms.append(geometry.rotation_about(bbox_center[0], bbox_center[1], degrees))
            meta["rotate_deg"] = degrees
        elif name == "translate":
            x_range = op_cfg.get("x_range_ratio", op_cfg.get("range_ratio"))
            y_range = op_cfg.get("y_range_ratio", op_cfg.get("range_ratio"))
            if x_range is None or y_range is None:
                raise ValueError("Translate config requires range_ratio or x_range_ratio/y_range_ratio.")
            dx = rng.uniform(float(x_range[0]), float(x_range[1])) * image_width
            dy = rng.uniform(float(y_range[0]), float(y_range[1])) * image_height
            transforms.append(geometry.translation_matrix(dx, dy))
            meta["translate_dx_px"] = dx
            meta["translate_dy_px"] = dy
        elif name == "shear":
            range_values = op_cfg.get("range_deg", op_cfg.get("range"))
            if range_values is None:
                raise ValueError("Shear config requires range_deg or range.")
            shear_x_deg = rng.uniform(float(range_values[0]), float(range_values[1]))
            shear_y_deg = rng.uniform(float(range_values[0]), float(range_values[1]))
            transforms.append(geometry.shear_about(bbox_center[0], bbox_center[1], shear_x_deg, shear_y_deg))
            meta["shear_x_deg"] = shear_x_deg
            meta["shear_y_deg"] = shear_y_deg
        elif name == "perspective":
            strength = rng.uniform(float(op_cfg["strength_range"][0]), float(op_cfg["strength_range"][1]))
            transforms.append(geometry.perspective_matrix(image_width, image_height, strength, rng))
            meta["perspective_strength"] = strength
        elif name == "bbox_crop":
            crop_scale = rng.uniform(float(op_cfg["scale_range"][0]), float(op_cfg["scale_range"][1]))
            crop_matrix, crop_meta = geometry.bbox_crop_and_resize_matrix(
                image_width,
                image_height,
                annotation.object_annotation.bbox,
                crop_scale=crop_scale,
                jitter_x_bbox_ratio=float(op_cfg.get("jitter_x_bbox_ratio", 0.0)),
                jitter_y_bbox_ratio=float(op_cfg.get("jitter_y_bbox_ratio", 0.0)),
                rng=rng,
            )
            transforms.append(crop_matrix)
            meta.update(crop_meta)
        elif name == "crop":
            crop_scale = rng.uniform(float(op_cfg["scale_range"][0]), float(op_cfg["scale_range"][1]))
            jitter_x, jitter_y = op_cfg.get("center_jitter_ratio", [1.0, 1.0])
            crop_matrix, crop_meta = geometry.crop_and_resize_matrix(
                image_width,
                image_height,
                crop_scale=crop_scale,
                jitter_x_ratio=float(jitter_x),
                jitter_y_ratio=float(jitter_y),
                rng=rng,
            )
            transforms.append(crop_matrix)
            meta.update(crop_meta)

    return geometry.compose_matrices(transforms), meta


def apply_appearance_ops(
    image: np.ndarray,
    appearance_cfg: dict[str, Any],
    template_cfg: dict[str, Any],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    requested = set(template_cfg.get("appearance", []))
    current = image.copy()
    meta: dict[str, Any] = {}
    image_height, image_width = image.shape[:2]

    for name in APPEARANCE_ORDER:
        if name not in requested or not config_enabled(name, appearance_cfg):
            continue
        op_cfg = appearance_cfg[name]
        if rng.random() > float(op_cfg.get("prob", 1.0)):
            continue

        if name == "brightness_contrast":
            brightness = rng.uniform(float(op_cfg["brightness_range"][0]), float(op_cfg["brightness_range"][1]))
            contrast = rng.uniform(float(op_cfg["contrast_range"][0]), float(op_cfg["contrast_range"][1]))
            current = appearance.adjust_brightness_contrast(current, brightness=brightness, contrast=contrast)
            meta["brightness"] = brightness
            meta["contrast"] = contrast
        elif name == "gamma":
            gamma_value = rng.uniform(float(op_cfg["range"][0]), float(op_cfg["range"][1]))
            current = appearance.adjust_gamma(current, gamma=gamma_value)
            meta["gamma"] = gamma_value
        elif name == "hsv":
            hue_delta = rng.uniform(float(op_cfg["hue_range"][0]), float(op_cfg["hue_range"][1]))
            sat_scale = rng.uniform(float(op_cfg["sat_range"][0]), float(op_cfg["sat_range"][1]))
            val_scale = rng.uniform(float(op_cfg["val_range"][0]), float(op_cfg["val_range"][1]))
            current = appearance.adjust_hsv(current, hue_delta=hue_delta, sat_scale=sat_scale, val_scale=val_scale)
            meta["hue_delta"] = hue_delta
            meta["sat_scale"] = sat_scale
            meta["val_scale"] = val_scale
        elif name == "rgb_shift":
            shifts = tuple(
                rng.uniform(float(op_cfg["shift_range"][0]), float(op_cfg["shift_range"][1]))
                for _ in range(3)
            )
            current = appearance.apply_rgb_shift(current, shifts=shifts)
            meta["rgb_shift_b"] = shifts[0]
            meta["rgb_shift_g"] = shifts[1]
            meta["rgb_shift_r"] = shifts[2]
        elif name == "dominant_channel":
            dominant_index = int(rng.choice([0, 1, 2]))
            dominant_gain = rng.uniform(float(op_cfg["dominant_gain_range"][0]), float(op_cfg["dominant_gain_range"][1]))
            other_gain = rng.uniform(float(op_cfg["other_gain_range"][0]), float(op_cfg["other_gain_range"][1]))
            bias = rng.uniform(float(op_cfg["bias_range"][0]), float(op_cfg["bias_range"][1]))
            current = appearance.apply_dominant_channel(
                current,
                dominant_index=dominant_index,
                dominant_gain=dominant_gain,
                other_gain=other_gain,
                bias=bias,
            )
            meta["dominant_channel_id"] = dominant_index
            meta["dominant_channel_gain"] = dominant_gain
            meta["dominant_other_gain"] = other_gain
            meta["dominant_bias"] = bias
        elif name == "channel_shuffle":
            order = tuple(rng.sample([0, 1, 2], 3))
            current = appearance.apply_channel_shuffle(current, order=order)
            meta["channel_shuffle_order"] = list(order)
        elif name == "blur":
            kernel = int(rng.choice(list(op_cfg["kernel_choices"])))
            current = appearance.gaussian_blur(current, kernel_size=kernel)
            meta["blur_kernel"] = kernel
        elif name == "motion_blur":
            kernel = int(rng.choice(list(op_cfg["kernel_choices"])))
            angle = rng.uniform(0.0, 180.0)
            current = appearance.motion_blur(current, kernel_size=kernel, angle_deg=angle)
            meta["motion_blur_kernel"] = kernel
            meta["motion_blur_angle_deg"] = angle
        elif name == "noise":
            std = rng.uniform(float(op_cfg["std_range"][0]), float(op_cfg["std_range"][1]))
            current = appearance.add_gaussian_noise(current, std=std, rng=np_rng)
            meta["noise_std"] = std
        elif name == "jpeg":
            quality = int(round(rng.uniform(float(op_cfg["quality_range"][0]), float(op_cfg["quality_range"][1]))))
            current = appearance.jpeg_compress(current, quality=quality)
            meta["jpeg_quality"] = quality
        elif name == "clahe":
            clip_limit = rng.uniform(float(op_cfg["clip_limit_range"][0]), float(op_cfg["clip_limit_range"][1]))
            tile_grid = int(rng.choice(list(op_cfg["tile_grid_choices"])))
            current = appearance.apply_clahe(current, clip_limit=clip_limit, tile_grid_size=tile_grid)
            meta["clahe_clip_limit"] = clip_limit
            meta["clahe_tile_grid"] = tile_grid
        elif name == "channel_gain":
            gains = tuple(
                rng.uniform(float(op_cfg["range"][0]), float(op_cfg["range"][1]))
                for _ in range(3)
            )
            current = appearance.apply_channel_gain(current, gains=gains)
            meta["channel_gain_b"] = gains[0]
            meta["channel_gain_g"] = gains[1]
            meta["channel_gain_r"] = gains[2]
        elif name == "shadow":
            darkness = rng.uniform(float(op_cfg["darkness_range"][0]), float(op_cfg["darkness_range"][1]))
            major = rng.uniform(float(op_cfg["major_axis_ratio"][0]), float(op_cfg["major_axis_ratio"][1])) * image_width
            minor = rng.uniform(float(op_cfg["minor_axis_ratio"][0]), float(op_cfg["minor_axis_ratio"][1])) * image_height
            center = (rng.uniform(0.0, image_width - 1.0), rng.uniform(0.0, image_height - 1.0))
            angle = rng.uniform(0.0, 180.0)
            blur_kernel = int(rng.choice(list(op_cfg["blur_kernel_choices"])))
            current = appearance.apply_shadow(
                current,
                center=center,
                axes=(major, minor),
                angle_deg=angle,
                darkness=darkness,
                blur_kernel=blur_kernel,
            )
            meta["shadow_darkness"] = darkness
            meta["shadow_center_x"] = center[0]
            meta["shadow_center_y"] = center[1]
            meta["shadow_major_axis"] = major
            meta["shadow_minor_axis"] = minor
            meta["shadow_angle_deg"] = angle

    return current, meta


def apply_occlusion_ops(
    image: np.ndarray,
    annotation: ObjectAnnotation,
    occlusion_cfg: dict[str, Any],
    template_cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[np.ndarray, list[OcclusionRegion], dict[str, Any]]:
    requested = set(template_cfg.get("occlusion", []))
    current = image.copy()
    meta: dict[str, Any] = {}
    image_height, image_width = image.shape[:2]
    all_regions: list[OcclusionRegion] = []

    def serialize_regions(regions: list[OcclusionRegion]) -> list[dict[str, float | str]]:
        return [
            {"x1": region.x1, "y1": region.y1, "x2": region.x2, "y2": region.y2, "kind": region.kind}
            for region in regions
        ]

    for name in OCCLUSION_ORDER:
        if name not in requested or not config_enabled(name, occlusion_cfg):
            continue
        op_cfg = occlusion_cfg[name]
        if rng.random() > float(op_cfg.get("prob", 1.0)):
            continue
        if name == "cutout":
            regions = occlusion.sample_cutout_regions(
                annotation.bbox,
                image_width=image_width,
                image_height=image_height,
                count_range=(int(op_cfg["count_range"][0]), int(op_cfg["count_range"][1])),
                size_ratio_range=(float(op_cfg["size_ratio_range"][0]), float(op_cfg["size_ratio_range"][1])),
                rng=rng,
            )
        elif name == "edge_cutout":
            regions = occlusion.sample_edge_cutout_regions(
                annotation.bbox,
                image_width=image_width,
                image_height=image_height,
                count_range=(int(op_cfg["count_range"][0]), int(op_cfg["count_range"][1])),
                thickness_ratio_range=(
                    float(op_cfg["thickness_ratio_range"][0]),
                    float(op_cfg["thickness_ratio_range"][1]),
                ),
                length_ratio_range=(float(op_cfg["length_ratio_range"][0]), float(op_cfg["length_ratio_range"][1])),
                rng=rng,
            )
        elif name == "corner_cutout":
            regions = occlusion.sample_corner_cutout_regions(
                annotation.bbox,
                image_width=image_width,
                image_height=image_height,
                size_ratio_range=(float(op_cfg["size_ratio_range"][0]), float(op_cfg["size_ratio_range"][1])),
                rng=rng,
            )
        else:
            continue

        current = occlusion.apply_cutout(
            current,
            regions,
            fill_mode=str(op_cfg.get("fill_mode", "mean")),
            rng=rng,
        )
        all_regions.extend(regions)
        meta[f"{name}_count"] = len(regions)
        meta[f"{name}_regions"] = serialize_regions(regions)

    return current, all_regions, meta


def generate_augmented_sample(
    annotation: ImageAnnotation,
    config: dict[str, Any],
    *,
    output_image: str,
    output_label: str,
    slot_index: int,
    attempt_index: int,
    rng: random.Random,
) -> GeneratedSample:
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    template_name, template_cfg = choose_template(config["templates"], rng)
    image = read_image_bgr(annotation.image_path)
    try:
        special_image, special_annotation, special_width, special_height, special_geometry_meta = maybe_apply_special_geometry(
            image,
            annotation.object_annotation,
            image_width=annotation.image_width,
            image_height=annotation.image_height,
            geometry_cfg=config["geometry"],
            template_cfg=template_cfg,
            rng=rng,
        )
        working_annotation = ImageAnnotation(
            sample_id=annotation.sample_id,
            split=annotation.split,
            image_path=annotation.image_path,
            label_path=annotation.label_path,
            image_width=special_width,
            image_height=special_height,
            object_annotation=special_annotation,
        )
        matrix, geometry_meta = sample_geometry_matrix(working_annotation, config["geometry"], template_cfg, rng)
        warped_image, valid_mask = geometry.warp_image_and_mask(
            special_image,
            matrix,
            border_mode=str(config["geometry"].get("border_mode", "reflect101")),
        )
        transformed_annotation, raw_bbox = apply_geometry_to_annotation(
            special_annotation,
            matrix,
            image_width=special_width,
            image_height=special_height,
            valid_mask=valid_mask,
            occlusion_regions=[],
        )
        image_after_appearance, appearance_meta = apply_appearance_ops(
            warped_image,
            config["appearance"],
            template_cfg,
            rng,
            np_rng,
        )
        image_after_occlusion, occlusion_regions, occlusion_meta = apply_occlusion_ops(
            image_after_appearance,
            transformed_annotation,
            config["occlusion"],
            template_cfg,
            rng,
        )
        final_annotation, raw_bbox_after_occlusion = apply_geometry_to_annotation(
            special_annotation,
            matrix,
            image_width=special_width,
            image_height=special_height,
            valid_mask=valid_mask,
            occlusion_regions=occlusion_regions,
        )
        metrics, reject_reason = validate_augmented_sample(
            final_annotation,
            raw_bbox_after_occlusion,
            image_width=special_width,
            image_height=special_height,
            source_visible_keypoints=count_visible_keypoints(special_annotation.keypoints),
            filter_cfg=config["filter"],
            aggressive_geometry_used=(
                "perspective_strength" in geometry_meta
                or "bbox_crop_scale" in geometry_meta
                or "crop_scale" in geometry_meta
            ),
        )
        geometry_meta = dict(special_geometry_meta) | geometry_meta
        transforms = {
            "template": template_name,
            "geometry": geometry_meta,
            "appearance": appearance_meta,
            "occlusion": occlusion_meta,
        }
        record = AugmentRecord(
            source_split=annotation.split,
            source_sample_id=annotation.sample_id,
            source_image=str(annotation.image_path),
            source_label=str(annotation.label_path),
            output_image=output_image,
            output_label=output_label,
            template=template_name,
            slot_index=slot_index,
            attempt_index=attempt_index,
            valid=reject_reason is None,
            reject_reason=reject_reason or "",
            transforms=transforms,
            metrics=metrics,
        )
        return GeneratedSample(image=image_after_occlusion, annotation=final_annotation, record=record)
    except ValueError as exc:
        record = AugmentRecord(
            source_split=annotation.split,
            source_sample_id=annotation.sample_id,
            source_image=str(annotation.image_path),
            source_label=str(annotation.label_path),
            output_image=output_image,
            output_label=output_label,
            template=template_name,
            slot_index=slot_index,
            attempt_index=attempt_index,
            valid=False,
            reject_reason=str(exc),
            transforms={"template": template_name},
            metrics=None,
        )
        return GeneratedSample(image=image, annotation=annotation.object_annotation, record=record)
