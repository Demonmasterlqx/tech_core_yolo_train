from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from pose_offline_aug import appearance, occlusion
from pose_offline_aug.object_scale import (
    apply_same_image_object_scale_to_area_ratio,
    apply_same_image_video_reframe_to_area_ratio,
)
from pose_offline_aug.structures import Keypoint, ObjectAnnotation, OcclusionRegion
from pose_offline_aug.validator import validate_augmented_sample
from .metrics import AreaBin, annotation_area_ratio_after_letterbox, bin_for_ratio, sample_target_ratio


@dataclass(frozen=True)
class SynthesizedSample:
    image: np.ndarray
    annotation: ObjectAnnotation
    strategy: str
    target_area_ratio: float
    actual_area_ratio: float
    transforms: dict[str, Any]


def _config_enabled(cfg: dict[str, Any], name: str) -> bool:
    return bool(cfg.get(name, {}).get("enabled", False))


def _update_annotation_for_occlusions(annotation: ObjectAnnotation, regions: list[OcclusionRegion]) -> ObjectAnnotation:
    if not regions:
        return annotation
    keypoints: list[Keypoint] = []
    for keypoint in annotation.keypoints:
        if keypoint.v <= 0:
            keypoints.append(keypoint)
            continue
        if any(region.contains(keypoint.x, keypoint.y) for region in regions):
            keypoints.append(Keypoint(x=keypoint.x, y=keypoint.y, v=1))
        else:
            keypoints.append(keypoint)
    return ObjectAnnotation(class_id=annotation.class_id, bbox=annotation.bbox, keypoints=keypoints)


def apply_random_appearance(
    image: np.ndarray,
    appearance_cfg: dict[str, Any],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = image.copy()
    meta: dict[str, Any] = {}

    if _config_enabled(appearance_cfg, "brightness_contrast") and rng.random() <= float(appearance_cfg["brightness_contrast"].get("prob", 1.0)):
        op_cfg = appearance_cfg["brightness_contrast"]
        brightness = rng.uniform(float(op_cfg["brightness_range"][0]), float(op_cfg["brightness_range"][1]))
        contrast = rng.uniform(float(op_cfg["contrast_range"][0]), float(op_cfg["contrast_range"][1]))
        current = appearance.adjust_brightness_contrast(current, brightness=brightness, contrast=contrast)
        meta["brightness"] = brightness
        meta["contrast"] = contrast

    if _config_enabled(appearance_cfg, "gamma") and rng.random() <= float(appearance_cfg["gamma"].get("prob", 1.0)):
        op_cfg = appearance_cfg["gamma"]
        gamma_value = rng.uniform(float(op_cfg["range"][0]), float(op_cfg["range"][1]))
        current = appearance.adjust_gamma(current, gamma=gamma_value)
        meta["gamma"] = gamma_value

    if _config_enabled(appearance_cfg, "hsv") and rng.random() <= float(appearance_cfg["hsv"].get("prob", 1.0)):
        op_cfg = appearance_cfg["hsv"]
        hue_delta = rng.uniform(float(op_cfg["hue_range"][0]), float(op_cfg["hue_range"][1]))
        sat_scale = rng.uniform(float(op_cfg["sat_range"][0]), float(op_cfg["sat_range"][1]))
        val_scale = rng.uniform(float(op_cfg["val_range"][0]), float(op_cfg["val_range"][1]))
        current = appearance.adjust_hsv(current, hue_delta=hue_delta, sat_scale=sat_scale, val_scale=val_scale)
        meta["hue_delta"] = hue_delta
        meta["sat_scale"] = sat_scale
        meta["val_scale"] = val_scale

    if _config_enabled(appearance_cfg, "temperature_tint") and rng.random() <= float(appearance_cfg["temperature_tint"].get("prob", 1.0)):
        op_cfg = appearance_cfg["temperature_tint"]
        temperature_shift = rng.uniform(float(op_cfg["temperature_range"][0]), float(op_cfg["temperature_range"][1]))
        tint_shift = rng.uniform(float(op_cfg["tint_range"][0]), float(op_cfg["tint_range"][1]))
        current = appearance.apply_temperature_tint(current, temperature_shift=temperature_shift, tint_shift=tint_shift)
        meta["temperature_shift"] = temperature_shift
        meta["tint_shift"] = tint_shift

    if _config_enabled(appearance_cfg, "rgb_shift") and rng.random() <= float(appearance_cfg["rgb_shift"].get("prob", 1.0)):
        op_cfg = appearance_cfg["rgb_shift"]
        shifts = tuple(rng.uniform(float(op_cfg["shift_range"][0]), float(op_cfg["shift_range"][1])) for _ in range(3))
        current = appearance.apply_rgb_shift(current, shifts=shifts)
        meta["rgb_shift_b"] = shifts[0]
        meta["rgb_shift_g"] = shifts[1]
        meta["rgb_shift_r"] = shifts[2]

    if _config_enabled(appearance_cfg, "dominant_channel") and rng.random() <= float(appearance_cfg["dominant_channel"].get("prob", 1.0)):
        op_cfg = appearance_cfg["dominant_channel"]
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

    if _config_enabled(appearance_cfg, "channel_shuffle") and rng.random() <= float(appearance_cfg["channel_shuffle"].get("prob", 1.0)):
        order = tuple(rng.sample([0, 1, 2], 3))
        current = appearance.apply_channel_shuffle(current, order=order)
        meta["channel_shuffle_order"] = list(order)

    if _config_enabled(appearance_cfg, "channel_gain") and rng.random() <= float(appearance_cfg["channel_gain"].get("prob", 1.0)):
        op_cfg = appearance_cfg["channel_gain"]
        gains = tuple(rng.uniform(float(op_cfg["range"][0]), float(op_cfg["range"][1])) for _ in range(3))
        current = appearance.apply_channel_gain(current, gains=gains)
        meta["channel_gain_b"] = gains[0]
        meta["channel_gain_g"] = gains[1]
        meta["channel_gain_r"] = gains[2]

    if _config_enabled(appearance_cfg, "highlight_exposure") and rng.random() <= float(appearance_cfg["highlight_exposure"].get("prob", 1.0)):
        op_cfg = appearance_cfg["highlight_exposure"]
        gain = rng.uniform(float(op_cfg["gain_range"][0]), float(op_cfg["gain_range"][1]))
        threshold = int(rng.uniform(float(op_cfg["threshold_range"][0]), float(op_cfg["threshold_range"][1])))
        rolloff = rng.uniform(float(op_cfg["rolloff_range"][0]), float(op_cfg["rolloff_range"][1]))
        current = appearance.apply_highlight_exposure(current, gain=gain, threshold=threshold, rolloff=rolloff)
        meta["highlight_gain"] = gain
        meta["highlight_threshold"] = threshold

    if _config_enabled(appearance_cfg, "blur") and rng.random() <= float(appearance_cfg["blur"].get("prob", 1.0)):
        kernel = int(rng.choice(list(appearance_cfg["blur"]["kernel_choices"])))
        current = appearance.gaussian_blur(current, kernel_size=kernel)
        meta["blur_kernel"] = kernel

    if _config_enabled(appearance_cfg, "motion_blur") and rng.random() <= float(appearance_cfg["motion_blur"].get("prob", 1.0)):
        op_cfg = appearance_cfg["motion_blur"]
        kernel = int(rng.choice(list(op_cfg["kernel_choices"])))
        angle = rng.uniform(0.0, 180.0)
        current = appearance.motion_blur(current, kernel_size=kernel, angle_deg=angle)
        meta["motion_blur_kernel"] = kernel
        meta["motion_blur_angle_deg"] = angle

    if _config_enabled(appearance_cfg, "noise") and rng.random() <= float(appearance_cfg["noise"].get("prob", 1.0)):
        op_cfg = appearance_cfg["noise"]
        std = rng.uniform(float(op_cfg["std_range"][0]), float(op_cfg["std_range"][1]))
        current = appearance.add_gaussian_noise(current, std=std, rng=np_rng)
        meta["noise_std"] = std

    if _config_enabled(appearance_cfg, "jpeg") and rng.random() <= float(appearance_cfg["jpeg"].get("prob", 1.0)):
        op_cfg = appearance_cfg["jpeg"]
        quality = int(round(rng.uniform(float(op_cfg["quality_range"][0]), float(op_cfg["quality_range"][1]))))
        current = appearance.jpeg_compress(current, quality=quality)
        meta["jpeg_quality"] = quality

    if _config_enabled(appearance_cfg, "clahe") and rng.random() <= float(appearance_cfg["clahe"].get("prob", 1.0)):
        op_cfg = appearance_cfg["clahe"]
        clip_limit = rng.uniform(float(op_cfg["clip_limit_range"][0]), float(op_cfg["clip_limit_range"][1]))
        tile_grid = int(rng.choice(list(op_cfg["tile_grid_choices"])))
        current = appearance.apply_clahe(current, clip_limit=clip_limit, tile_grid_size=tile_grid)
        meta["clahe_clip_limit"] = clip_limit
        meta["clahe_tile_grid"] = tile_grid

    if _config_enabled(appearance_cfg, "unsharp_mask") and rng.random() <= float(appearance_cfg["unsharp_mask"].get("prob", 1.0)):
        op_cfg = appearance_cfg["unsharp_mask"]
        kernel = int(rng.choice(list(op_cfg["kernel_choices"])))
        amount = rng.uniform(float(op_cfg["amount_range"][0]), float(op_cfg["amount_range"][1]))
        current = appearance.apply_unsharp_mask(current, kernel_size=kernel, amount=amount)
        meta["unsharp_kernel"] = kernel
        meta["unsharp_amount"] = amount

    if _config_enabled(appearance_cfg, "shadow") and rng.random() <= float(appearance_cfg["shadow"].get("prob", 1.0)):
        op_cfg = appearance_cfg["shadow"]
        image_height, image_width = current.shape[:2]
        darkness = rng.uniform(float(op_cfg["darkness_range"][0]), float(op_cfg["darkness_range"][1]))
        center = (rng.uniform(0.0, image_width - 1.0), rng.uniform(0.0, image_height - 1.0))
        major = rng.uniform(float(op_cfg["major_axis_ratio"][0]), float(op_cfg["major_axis_ratio"][1])) * image_width
        minor = rng.uniform(float(op_cfg["minor_axis_ratio"][0]), float(op_cfg["minor_axis_ratio"][1])) * image_height
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

    return current, meta


def apply_random_occlusion(
    image: np.ndarray,
    annotation: ObjectAnnotation,
    occlusion_cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[np.ndarray, ObjectAnnotation, dict[str, Any]]:
    current = image.copy()
    regions: list[OcclusionRegion] = []
    image_height, image_width = current.shape[:2]
    meta: dict[str, Any] = {}

    if _config_enabled(occlusion_cfg, "cutout") and rng.random() <= float(occlusion_cfg["cutout"].get("prob", 1.0)):
        op_cfg = occlusion_cfg["cutout"]
        regions.extend(
            occlusion.sample_cutout_regions(
                annotation.bbox,
                image_width=image_width,
                image_height=image_height,
                count_range=(int(op_cfg["count_range"][0]), int(op_cfg["count_range"][1])),
                size_ratio_range=(float(op_cfg["size_ratio_range"][0]), float(op_cfg["size_ratio_range"][1])),
                rng=rng,
            )
        )
        current = occlusion.apply_cutout(current, regions[-len(regions) :], fill_mode=str(op_cfg.get("fill_mode", "mean")), rng=rng)
        meta["cutout_count"] = len(regions)

    edge_regions: list[OcclusionRegion] = []
    if _config_enabled(occlusion_cfg, "edge_cutout") and rng.random() <= float(occlusion_cfg["edge_cutout"].get("prob", 1.0)):
        op_cfg = occlusion_cfg["edge_cutout"]
        edge_regions = occlusion.sample_edge_cutout_regions(
            annotation.bbox,
            image_width=image_width,
            image_height=image_height,
            count_range=(int(op_cfg["count_range"][0]), int(op_cfg["count_range"][1])),
            thickness_ratio_range=(float(op_cfg["thickness_ratio_range"][0]), float(op_cfg["thickness_ratio_range"][1])),
            length_ratio_range=(float(op_cfg["length_ratio_range"][0]), float(op_cfg["length_ratio_range"][1])),
            rng=rng,
        )
        current = occlusion.apply_cutout(current, edge_regions, fill_mode=str(op_cfg.get("fill_mode", "mean")), rng=rng)
        regions.extend(edge_regions)
        meta["edge_cutout_count"] = len(edge_regions)

    corner_regions: list[OcclusionRegion] = []
    if _config_enabled(occlusion_cfg, "corner_cutout") and rng.random() <= float(occlusion_cfg["corner_cutout"].get("prob", 1.0)):
        op_cfg = occlusion_cfg["corner_cutout"]
        corner_regions = occlusion.sample_corner_cutout_regions(
            annotation.bbox,
            image_width=image_width,
            image_height=image_height,
            size_ratio_range=(float(op_cfg["size_ratio_range"][0]), float(op_cfg["size_ratio_range"][1])),
            rng=rng,
        )
        current = occlusion.apply_cutout(current, corner_regions, fill_mode=str(op_cfg.get("fill_mode", "mean")), rng=rng)
        regions.extend(corner_regions)
        meta["corner_cutout_count"] = len(corner_regions)

    updated_annotation = _update_annotation_for_occlusions(annotation, regions)
    return current, updated_annotation, meta


def _choose_strategies(config: dict[str, Any], image_width: int, image_height: int) -> list[str]:
    geometry_cfg = config["geometry"]
    preference = str(config["runtime"].get("strategy_preference", "auto")).lower()
    enabled: list[str] = []
    if _config_enabled(geometry_cfg, "object_scale"):
        enabled.append("object_scale")
    if _config_enabled(geometry_cfg, "video_reframe"):
        enabled.append("video_reframe")
    if not enabled:
        raise ValueError("At least one of geometry.object_scale or geometry.video_reframe must be enabled.")

    if preference == "object_scale":
        return sorted(enabled, key=lambda name: 0 if name == "object_scale" else 1)
    if preference == "video_reframe":
        return sorted(enabled, key=lambda name: 0 if name == "video_reframe" else 1)
    if image_width > image_height and "video_reframe" in enabled:
        return sorted(enabled, key=lambda name: 0 if name == "video_reframe" else 1)
    return sorted(enabled, key=lambda name: 0 if name == "object_scale" else 1)


def synthesize_area_balanced_sample(
    image: np.ndarray,
    annotation: ObjectAnnotation,
    config: dict[str, Any],
    target_bin: AreaBin,
    rng: random.Random,
) -> SynthesizedSample:
    imgsz = float(config["metric"]["imgsz"])
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    image_height, image_width = image.shape[:2]
    ratio_low, ratio_high = sample_target_ratio(target_bin, float(config["metric"].get("target_margin_ratio", 0.15)))
    target_area_ratio = ratio_low if ratio_high <= ratio_low else rng.uniform(ratio_low, ratio_high)
    strategies = _choose_strategies(config, image_width=image_width, image_height=image_height)
    last_error: Exception | None = None

    for strategy in strategies:
        try:
            if strategy == "video_reframe":
                transformed_image, transformed_annotation, strategy_meta = apply_same_image_video_reframe_to_area_ratio(
                    image,
                    annotation,
                    config["geometry"]["video_reframe"],
                    target_area_ratio=target_area_ratio,
                    rng=rng,
                )
            else:
                transformed_image, transformed_annotation, strategy_meta = apply_same_image_object_scale_to_area_ratio(
                    image,
                    annotation,
                    config["geometry"]["object_scale"],
                    target_area_ratio=target_area_ratio,
                    rng=rng,
                )
            actual_area_ratio = annotation_area_ratio_after_letterbox(
                transformed_annotation,
                image_width=transformed_image.shape[1],
                image_height=transformed_image.shape[0],
                imgsz=imgsz,
            )
            if bin_for_ratio(actual_area_ratio, [target_bin]) is None:
                raise ValueError("target_area_ratio_bin_miss")
            appearance_image, appearance_meta = apply_random_appearance(
                transformed_image,
                config["appearance"],
                rng,
                np_rng,
            )
            final_image, final_annotation, occlusion_meta = apply_random_occlusion(
                appearance_image,
                transformed_annotation,
                config["occlusion"],
                rng,
            )
            metrics, reject_reason = validate_augmented_sample(
                final_annotation,
                transformed_annotation.bbox,
                image_width=final_image.shape[1],
                image_height=final_image.shape[0],
                source_visible_keypoints=sum(1 for item in annotation.keypoints if item.v >= 2),
                filter_cfg=config["filter"],
                aggressive_geometry_used=False,
            )
            if reject_reason is not None:
                raise ValueError(reject_reason)
            return SynthesizedSample(
                image=final_image,
                annotation=final_annotation,
                strategy=strategy,
                target_area_ratio=target_area_ratio,
                actual_area_ratio=actual_area_ratio,
                transforms={
                    "strategy": strategy,
                    "strategy_meta": strategy_meta,
                    "appearance": appearance_meta,
                    "occlusion": occlusion_meta,
                    "visible_keypoints": metrics.visible_keypoints,
                },
            )
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(str(last_error) if last_error is not None else "synthesis_failed")
