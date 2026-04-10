from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from pose_offline_aug.io import object_annotation_to_yolo_pose_line
from pose_offline_aug import appearance, geometry, occlusion
from pose_offline_aug.labels import transform_keypoints
from pose_offline_aug.object_scale import (
    apply_same_image_object_scale,
    apply_same_image_object_scale_to_area_ratio,
    apply_same_image_video_reframe,
    apply_same_image_video_reframe_to_area_ratio,
    input_max_side_px,
    input_area_ratio,
    padded_bbox_rect,
    rects_overlap,
    select_same_image_background_patch,
)
from pose_offline_aug.structures import BBox, Keypoint, ObjectAnnotation, OcclusionRegion
from pose_offline_aug.validator import validate_augmented_sample
from build_pose_area_balanced_dataset import DatasetBuilder as AreaBalancedDatasetBuilder
from build_pose_augmented_dataset import DatasetBuilder
from pose_dataset_build_utils import load_yaml_file
from repartition_pose_dataset import ensure_unique_target_names, largest_remainder_counts


class OfflineAugTests(unittest.TestCase):
    def make_annotation(self, visible_keypoints: int, total_keypoints: int = 33) -> ObjectAnnotation:
        keypoints = [
            Keypoint(x=8.0 + index, y=8.0 + index, v=2 if index < visible_keypoints else 0)
            for index in range(total_keypoints)
        ]
        return ObjectAnnotation(
            class_id=0,
            bbox=BBox(x1=4.0, y1=4.0, x2=28.0, y2=28.0),
            keypoints=keypoints,
        )

    def test_out_of_frame_keypoints_are_dropped_not_clipped(self) -> None:
        keypoints = [Keypoint(x=10.0, y=10.0, v=2)]
        matrix = np.array(
            [
                [1.0, 0.0, 100.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        valid_mask = np.full((32, 32), 255, dtype=np.uint8)
        transformed = transform_keypoints(
            keypoints,
            matrix,
            image_width=32,
            image_height=32,
            valid_mask=valid_mask,
            occlusion_regions=[],
        )
        self.assertEqual(transformed[0].v, 0)
        self.assertEqual(transformed[0].x, 0.0)
        self.assertEqual(transformed[0].y, 0.0)

    def test_occluded_in_frame_keypoints_stay_labeled(self) -> None:
        keypoints = [Keypoint(x=10.0, y=10.0, v=2)]
        matrix = np.eye(3, dtype=np.float32)
        valid_mask = np.full((32, 32), 255, dtype=np.uint8)
        transformed = transform_keypoints(
            keypoints,
            matrix,
            image_width=32,
            image_height=32,
            valid_mask=valid_mask,
            occlusion_regions=[OcclusionRegion(x1=8.0, y1=8.0, x2=12.0, y2=12.0)],
        )
        self.assertEqual(transformed[0].v, 1)
        self.assertAlmostEqual(transformed[0].x, 10.0)
        self.assertAlmostEqual(transformed[0].y, 10.0)

    def test_serialization_zeros_unlabeled_keypoints(self) -> None:
        annotation = ObjectAnnotation(
            class_id=0,
            bbox=BBox(x1=2.0, y1=2.0, x2=12.0, y2=12.0),
            keypoints=[Keypoint(x=99.0, y=-4.0, v=0)],
        )
        line = object_annotation_to_yolo_pose_line(annotation, image_width=32, image_height=32)
        parts = line.split()
        self.assertEqual(parts[5], "0")
        self.assertEqual(parts[6], "0")
        self.assertEqual(parts[7], "0")

    def test_bbox_crop_is_centered_on_bbox(self) -> None:
        bbox = BBox(x1=55.0, y1=45.0, x2=75.0, y2=65.0)
        matrix, _ = geometry.bbox_crop_and_resize_matrix(
            image_width=100,
            image_height=100,
            bbox=bbox,
            crop_scale=0.5,
            jitter_x_bbox_ratio=0.0,
            jitter_y_bbox_ratio=0.0,
            rng=random.Random(0),
        )
        center_x, center_y = geometry.transform_xy(matrix, *bbox.center)
        self.assertAlmostEqual(center_x, 50.0, places=4)
        self.assertAlmostEqual(center_y, 50.0, places=4)

    def test_new_appearance_ops_work(self) -> None:
        sparse = np.zeros((9, 9, 3), dtype=np.uint8)
        sparse[4, 4] = np.array([255, 255, 255], dtype=np.uint8)
        motion = appearance.motion_blur(sparse, kernel_size=5, angle_deg=0.0)
        self.assertEqual(motion.shape, sparse.shape)
        self.assertLess(int(motion[4, 4, 0]), 255)
        self.assertGreater(int(motion[4, 2, 0]), 0)

        flat = np.full((8, 8, 3), 120, dtype=np.uint8)
        clahe = appearance.apply_clahe(flat, clip_limit=2.0, tile_grid_size=8)
        self.assertEqual(clahe.shape, flat.shape)
        self.assertEqual(clahe.dtype, np.uint8)

        gained = appearance.apply_channel_gain(
            np.array([[[10, 20, 30]]], dtype=np.uint8),
            gains=(2.0, 0.5, 1.0),
        )
        self.assertEqual(gained.tolist(), [[[20, 10, 30]]])

        shifted = appearance.apply_rgb_shift(
            np.array([[[10, 20, 30]]], dtype=np.uint8),
            shifts=(5.0, -10.0, 20.0),
        )
        self.assertEqual(shifted.tolist(), [[[15, 10, 50]]])

        dominant = appearance.apply_dominant_channel(
            np.array([[[40, 40, 40]]], dtype=np.uint8),
            dominant_index=2,
            dominant_gain=2.0,
            other_gain=0.5,
            bias=0.0,
        )
        self.assertEqual(dominant.tolist(), [[[20, 20, 80]]])

        shuffled = appearance.apply_channel_shuffle(
            np.array([[[1, 2, 3]]], dtype=np.uint8),
            order=(2, 0, 1),
        )
        self.assertEqual(shuffled.tolist(), [[[3, 1, 2]]])

        warmed = appearance.apply_temperature_tint(
            np.array([[[80, 90, 100]]], dtype=np.uint8),
            temperature_shift=12.0,
            tint_shift=-6.0,
        )
        self.assertEqual(warmed.shape, (1, 1, 3))

        highlights = appearance.apply_highlight_exposure(
            np.full((4, 4, 3), 220, dtype=np.uint8),
            gain=1.2,
            threshold=180,
            rolloff=20.0,
        )
        self.assertGreaterEqual(int(highlights[0, 0, 0]), 220)

        sharpened = appearance.apply_unsharp_mask(
            np.full((5, 5, 3), 128, dtype=np.uint8),
            kernel_size=3,
            amount=0.5,
        )
        self.assertEqual(sharpened.shape, (5, 5, 3))

    def test_new_occlusion_ops_work(self) -> None:
        bbox = BBox(x1=20.0, y1=20.0, x2=60.0, y2=60.0)
        edge_regions = occlusion.sample_edge_cutout_regions(
            bbox,
            image_width=80,
            image_height=80,
            count_range=(1, 1),
            thickness_ratio_range=(0.1, 0.1),
            length_ratio_range=(0.5, 0.5),
            rng=random.Random(1),
        )
        self.assertEqual(len(edge_regions), 1)
        self.assertEqual(edge_regions[0].kind, "edge_cutout")
        self.assertTrue(
            edge_regions[0].x1 == bbox.x1
            or edge_regions[0].x2 == bbox.x2
            or edge_regions[0].y1 == bbox.y1
            or edge_regions[0].y2 == bbox.y2
        )

        corner_regions = occlusion.sample_corner_cutout_regions(
            bbox,
            image_width=80,
            image_height=80,
            size_ratio_range=(0.2, 0.2),
            rng=random.Random(2),
        )
        self.assertEqual(len(corner_regions), 1)
        self.assertEqual(corner_regions[0].kind, "corner_cutout")

        gradient = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        filled = occlusion.apply_cutout(
            gradient,
            [OcclusionRegion(x1=2.0, y1=2.0, x2=6.0, y2=6.0)],
            fill_mode="local_patch",
            rng=random.Random(3),
        )
        self.assertEqual(filled.shape, gradient.shape)
        self.assertEqual(filled.dtype, gradient.dtype)

    def test_same_image_background_patch_excludes_padded_bbox(self) -> None:
        image = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)
        bbox = BBox(x1=44.0, y1=44.0, x2=84.0, y2=84.0)
        result = select_same_image_background_patch(
            image,
            bbox,
            exclusion_margin_ratio=0.15,
            rng=random.Random(4),
        )
        self.assertIsNotNone(result)
        patch, rect = result or (None, None)
        self.assertEqual(patch.shape, image.shape)
        padded_rect = padded_bbox_rect(bbox, exclusion_margin_ratio=0.15, image_width=128, image_height=128)
        self.assertFalse(rects_overlap(rect, padded_rect))

    def test_object_scale_can_zoom_out_and_zoom_in(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[..., 0] = np.arange(128, dtype=np.uint8)[None, :]
        image[..., 1] = np.arange(128, dtype=np.uint8)[:, None]
        annotation = ObjectAnnotation(
            class_id=0,
            bbox=BBox(x1=44.0, y1=44.0, x2=84.0, y2=84.0),
            keypoints=[Keypoint(x=54.0, y=54.0, v=2), Keypoint(x=74.0, y=74.0, v=2)],
        )
        base_cfg = {
            "context_scale_range": [1.2, 1.2],
            "center_jitter_ratio": 0.0,
            "exclusion_margin_ratio": 0.15,
            "feather_px": 8,
            "min_source_context_px": 32,
        }
        zoom_out_image, zoom_out_annotation, zoom_out_meta = apply_same_image_object_scale(
            image,
            annotation,
            base_cfg | {"resize_scale_range": [0.75, 0.75]},
            random.Random(5),
        )
        zoom_in_image, zoom_in_annotation, zoom_in_meta = apply_same_image_object_scale(
            image,
            annotation,
            base_cfg | {"resize_scale_range": [1.25, 1.25]},
            random.Random(6),
        )

        self.assertEqual(zoom_out_image.shape, image.shape)
        self.assertEqual(zoom_in_image.shape, image.shape)
        self.assertLess(zoom_out_annotation.bbox.width, annotation.bbox.width)
        self.assertLess(zoom_out_annotation.bbox.height, annotation.bbox.height)
        self.assertGreater(zoom_in_annotation.bbox.width, annotation.bbox.width)
        self.assertGreater(zoom_in_annotation.bbox.height, annotation.bbox.height)
        self.assertTrue(all(keypoint.v == 2 for keypoint in zoom_out_annotation.keypoints))
        self.assertTrue(all(keypoint.v == 2 for keypoint in zoom_in_annotation.keypoints))
        self.assertTrue(all(0.0 <= keypoint.x < 128.0 and 0.0 <= keypoint.y < 128.0 for keypoint in zoom_out_annotation.keypoints))
        self.assertTrue(all(0.0 <= keypoint.x < 128.0 and 0.0 <= keypoint.y < 128.0 for keypoint in zoom_in_annotation.keypoints))
        self.assertTrue(zoom_out_meta["object_scale_applied"])
        self.assertTrue(zoom_in_meta["object_scale_applied"])

    def test_video_reframe_outputs_landscape_and_target_input_scale(self) -> None:
        image = np.zeros((256, 144, 3), dtype=np.uint8)
        image[..., 0] = 50
        image[60:180, 36:100] = np.array([180, 180, 180], dtype=np.uint8)
        annotation = ObjectAnnotation(
            class_id=0,
            bbox=BBox(x1=36.0, y1=60.0, x2=100.0, y2=180.0),
            keypoints=[Keypoint(x=48.0, y=82.0, v=2), Keypoint(x=82.0, y=158.0, v=2)],
        )
        cfg = {
            "output_width": 160,
            "output_height": 90,
            "context_scale_range": [1.15, 1.15],
            "target_input_max_side_range": [55.0, 65.0],
            "input_size_px": 120.0,
            "center_x_range": [0.6, 0.6],
            "center_y_range": [0.7, 0.7],
            "exclusion_margin_ratio": 0.1,
            "feather_px": 6,
            "min_source_context_px": 32.0,
            "target_tolerance_px": 8.0,
        }
        reframed_image, reframed_annotation, meta = apply_same_image_video_reframe(
            image,
            annotation,
            cfg,
            random.Random(7),
        )
        self.assertEqual(reframed_image.shape[:2], (90, 160))
        self.assertGreaterEqual(meta["actual_input_max_side_px"], 55.0)
        self.assertLessEqual(meta["actual_input_max_side_px"], 65.0)
        self.assertTrue(all(keypoint.v == 2 for keypoint in reframed_annotation.keypoints))
        self.assertTrue(all(0.0 <= keypoint.x < 160.0 and 0.0 <= keypoint.y < 90.0 for keypoint in reframed_annotation.keypoints))

    def test_object_scale_can_target_input_area_ratio(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[..., 0] = np.arange(128, dtype=np.uint8)[None, :]
        annotation = ObjectAnnotation(
            class_id=0,
            bbox=BBox(x1=36.0, y1=36.0, x2=92.0, y2=92.0),
            keypoints=[Keypoint(x=48.0, y=48.0, v=2), Keypoint(x=80.0, y=80.0, v=2)],
        )
        transformed_image, transformed_annotation, meta = apply_same_image_object_scale_to_area_ratio(
            image,
            annotation,
            {
                "input_size_px": 128.0,
                "context_scale_range": [1.2, 1.2],
                "center_jitter_ratio": 0.0,
                "exclusion_margin_ratio": 0.15,
                "feather_px": 8,
                "min_source_context_px": 32.0,
            },
            target_area_ratio=0.08,
            rng=random.Random(8),
        )
        actual_ratio = input_area_ratio(
            bbox_width_px=transformed_annotation.bbox.width,
            bbox_height_px=transformed_annotation.bbox.height,
            image_width=transformed_image.shape[1],
            image_height=transformed_image.shape[0],
            input_size_px=128.0,
        )
        self.assertAlmostEqual(actual_ratio, meta["actual_input_area_ratio"], places=6)
        self.assertLess(abs(actual_ratio - 0.08), 0.01)

    def test_video_reframe_can_target_input_area_ratio(self) -> None:
        image = np.zeros((256, 144, 3), dtype=np.uint8)
        image[60:180, 36:100] = np.array([180, 180, 180], dtype=np.uint8)
        annotation = ObjectAnnotation(
            class_id=0,
            bbox=BBox(x1=36.0, y1=60.0, x2=100.0, y2=180.0),
            keypoints=[Keypoint(x=48.0, y=82.0, v=2), Keypoint(x=82.0, y=158.0, v=2)],
        )
        transformed_image, transformed_annotation, meta = apply_same_image_video_reframe_to_area_ratio(
            image,
            annotation,
            {
                "output_width": 160,
                "output_height": 90,
                "context_scale_range": [1.15, 1.15],
                "input_size_px": 120.0,
                "center_x_range": [0.6, 0.6],
                "center_y_range": [0.7, 0.7],
                "exclusion_margin_ratio": 0.1,
                "feather_px": 6,
                "min_source_context_px": 32.0,
                "target_area_tolerance_ratio": 0.02,
            },
            target_area_ratio=0.03,
            rng=random.Random(9),
        )
        actual_ratio = input_area_ratio(
            bbox_width_px=transformed_annotation.bbox.width,
            bbox_height_px=transformed_annotation.bbox.height,
            image_width=transformed_image.shape[1],
            image_height=transformed_image.shape[0],
            input_size_px=120.0,
        )
        self.assertLess(abs(actual_ratio - 0.03), 0.02)
        self.assertAlmostEqual(actual_ratio, meta["actual_input_area_ratio"], places=6)

    def test_input_max_side_px_uses_letterbox_scale(self) -> None:
        value = input_max_side_px(
            bbox_width_px=120.0,
            bbox_height_px=80.0,
            image_width=1280,
            image_height=720,
            input_size_px=960.0,
        )
        self.assertAlmostEqual(value, 90.0)

    def test_visible_keypoint_threshold_uses_ratio_and_floor(self) -> None:
        filter_cfg = {
            "min_bbox_area_ratio": 0.0,
            "min_bbox_width_px": 1,
            "min_bbox_height_px": 1,
            "max_out_of_frame_ratio": 0.45,
            "analysis_imgsz": 32.0,
        }
        rejected_metrics, rejected_reason = validate_augmented_sample(
            self.make_annotation(visible_keypoints=17),
            BBox(x1=4.0, y1=4.0, x2=28.0, y2=28.0),
            image_width=32,
            image_height=32,
            source_visible_keypoints=20,
            filter_cfg=filter_cfg,
        )
        self.assertEqual(rejected_metrics.visible_keypoints, 17)
        self.assertEqual(rejected_reason, "visible_keypoints_below_threshold")
        self.assertAlmostEqual(rejected_metrics.bbox_max_side_input_px, 24.0)

        accepted_metrics, accepted_reason = validate_augmented_sample(
            self.make_annotation(visible_keypoints=18),
            BBox(x1=4.0, y1=4.0, x2=28.0, y2=28.0),
            image_width=32,
            image_height=32,
            source_visible_keypoints=20,
            filter_cfg=filter_cfg,
        )
        self.assertEqual(accepted_metrics.visible_keypoints, 18)
        self.assertIsNone(accepted_reason)
        self.assertAlmostEqual(accepted_metrics.bbox_max_side_input_px, 24.0)

    def test_aggressive_geometry_uses_stricter_out_of_frame_cap(self) -> None:
        annotation = self.make_annotation(visible_keypoints=20)
        raw_bbox = BBox(x1=-4.0, y1=4.0, x2=6.0, y2=14.0)
        clipped_annotation = ObjectAnnotation(
            class_id=annotation.class_id,
            bbox=BBox(x1=0.0, y1=4.0, x2=6.0, y2=14.0),
            keypoints=annotation.keypoints,
        )
        filter_cfg = {
            "min_bbox_area_ratio": 0.0,
            "min_bbox_width_px": 1,
            "min_bbox_height_px": 1,
            "max_out_of_frame_ratio": 0.45,
            "max_out_of_frame_ratio_aggressive": 0.35,
        }
        _, relaxed_reason = validate_augmented_sample(
            clipped_annotation,
            raw_bbox,
            image_width=32,
            image_height=32,
            source_visible_keypoints=20,
            filter_cfg=filter_cfg,
            aggressive_geometry_used=False,
        )
        _, aggressive_reason = validate_augmented_sample(
            clipped_annotation,
            raw_bbox,
            image_width=32,
            image_height=32,
            source_visible_keypoints=20,
            filter_cfg=filter_cfg,
            aggressive_geometry_used=True,
        )
        self.assertIsNone(relaxed_reason)
        self.assertEqual(aggressive_reason, "bbox_out_of_frame_ratio_too_high")

    def test_train_only_sources_write_none_eval_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            (source_root / "train" / "images").mkdir(parents=True)
            (source_root / "train" / "labels").mkdir(parents=True)
            (source_root / "dataset.yaml").write_text(
                "\n".join(
                    [
                        "train: train",
                        "val: none",
                        "test: none",
                        "nc: 1",
                        "names:",
                        "- tech_core_mark",
                        "kpt_shape:",
                        "- 33",
                        "- 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "source": {
                    "root": str(source_root),
                    "data_yaml": str(source_root / "dataset.yaml"),
                },
                "output": {
                    "root": str(root / "out"),
                    "clean_output": True,
                },
                "runtime": {
                    "seed": 52,
                    "num_aug_per_image": 1,
                    "keep_original_train": True,
                    "image_suffix": ".jpg",
                    "label_suffix": ".txt",
                    "max_attempts_per_augment": 1,
                },
                "geometry": {"border_mode": "reflect101"},
                "appearance": {},
                "occlusion": {},
                "filter": {
                    "min_bbox_area_ratio": 0.0,
                    "min_bbox_width_px": 1,
                    "min_bbox_height_px": 1,
                    "max_out_of_frame_ratio": 0.45,
                },
                "review": {
                    "enabled": False,
                    "per_template": 1,
                    "export_dir": "analysis/review",
                },
                "templates": {},
            }
            builder = DatasetBuilder(config, dry_run=False, limit=None, visualize=False)
            builder.write_dataset_yamls()

            written = load_yaml_file(root / "out" / "data.yaml")
            raw_eval = load_yaml_file(root / "out" / "data.raw_eval.yaml")
            self.assertEqual(written["val"], "none")
            self.assertEqual(written["test"], "none")
            self.assertEqual(raw_eval["val"], "none")
            self.assertEqual(raw_eval["test"], "none")

    def test_area_balance_builder_hits_exact_generated_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            (source_root / "train" / "images").mkdir(parents=True)
            (source_root / "train" / "labels").mkdir(parents=True)

            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[20:44, 20:44] = np.array([180, 180, 180], dtype=np.uint8)
            image_path = source_root / "train" / "images" / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            keypoints = []
            for index in range(33):
                if index < 20:
                    keypoints.extend([0.4 + index * 0.001, 0.4 + index * 0.001, 2])
                else:
                    keypoints.extend([0, 0, 0])
            label_values = ["0", "0.5", "0.5", "0.375", "0.375", *[str(value) for value in keypoints]]
            (source_root / "train" / "labels" / "sample.txt").write_text(" ".join(label_values) + "\n", encoding="utf-8")
            (source_root / "dataset.yaml").write_text(
                "\n".join(
                    [
                        "train: train",
                        "val: none",
                        "test: none",
                        "nc: 1",
                        "names:",
                        "- tech_core_mark",
                        "kpt_shape:",
                        "- 33",
                        "- 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = {
                "source": {"root": str(source_root), "data_yaml": str(source_root / "dataset.yaml")},
                "output": {"root": str(root / "balanced"), "clean_output": True},
                "runtime": {
                    "seed": 52,
                    "keep_original_train": True,
                    "generated_multiplier": 2,
                    "max_attempts_per_bin": 6,
                    "image_suffix": ".jpg",
                    "label_suffix": ".txt",
                    "strategy_preference": "object_scale",
                },
                "metric": {
                    "imgsz": 64,
                    "min_area_ratio": 0.01,
                    "max_area_ratio": 0.03,
                    "bin_count": 2,
                    "target_margin_ratio": 0.1,
                },
                "geometry": {
                    "object_scale": {
                        "enabled": True,
                        "input_size_px": 64,
                        "context_scale_range": [1.1, 1.2],
                        "center_jitter_ratio": 0.0,
                        "exclusion_margin_ratio": 0.12,
                        "feather_px": 4,
                        "min_source_context_px": 12,
                    },
                    "video_reframe": {
                        "enabled": False,
                        "output_width": 128,
                        "output_height": 72,
                        "input_size_px": 64,
                        "context_scale_range": [1.1, 1.2],
                        "center_x_range": [0.5, 0.6],
                        "center_y_range": [0.6, 0.7],
                        "exclusion_margin_ratio": 0.12,
                        "feather_px": 4,
                        "min_source_context_px": 12,
                        "target_area_tolerance_ratio": 0.02,
                    },
                },
                "appearance": {
                    "brightness_contrast": {"enabled": False},
                    "gamma": {"enabled": False},
                    "hsv": {"enabled": False},
                    "temperature_tint": {"enabled": False},
                    "rgb_shift": {"enabled": False},
                    "dominant_channel": {"enabled": False},
                    "channel_shuffle": {"enabled": False},
                    "channel_gain": {"enabled": False},
                    "highlight_exposure": {"enabled": False},
                    "blur": {"enabled": False},
                    "motion_blur": {"enabled": False},
                    "noise": {"enabled": False},
                    "jpeg": {"enabled": False},
                    "clahe": {"enabled": False},
                    "unsharp_mask": {"enabled": False},
                    "shadow": {"enabled": False},
                },
                "occlusion": {
                    "cutout": {"enabled": False},
                    "edge_cutout": {"enabled": False},
                    "corner_cutout": {"enabled": False},
                },
                "filter": {
                    "analysis_imgsz": 64,
                    "min_bbox_area_ratio": 0.001,
                    "min_visible_keypoints_floor": 10,
                    "min_visible_keypoints_ratio": 0.5,
                    "min_bbox_width_px": 4,
                    "min_bbox_height_px": 4,
                    "max_out_of_frame_ratio": 0.35,
                    "max_out_of_frame_ratio_aggressive": 0.25,
                },
                "review": {"enabled": False, "per_bin": 1, "export_dir": "analysis/review"},
            }

            builder = AreaBalancedDatasetBuilder(config, dry_run=False, limit=None)
            summary = builder.build()
            self.assertEqual(summary["raw_train_count"], 1)
            self.assertEqual(summary["generated_count"], 2)
            self.assertEqual(summary["final_train_count"], 3)

            bin_summary = load_yaml_file(root / "balanced" / "analysis" / "bin_summary.json")
            self.assertEqual(bin_summary["generated_only_hist"]["[0.010,0.020)"], 1)
            self.assertEqual(bin_summary["generated_only_hist"]["[0.020,0.030)"], 1)


class RepartitionTests(unittest.TestCase):
    def test_largest_remainder_counts_preserves_total(self) -> None:
        counts = largest_remainder_counts(7, (0.7, 0.2, 0.1))
        self.assertEqual(sum(counts.values()), 7)
        self.assertEqual(counts, {"train": 5, "valid": 1, "test": 1})

    def test_detects_output_name_collisions(self) -> None:
        from pose_dataset_build_utils import PoseDatasetSample

        sample_a = PoseDatasetSample(
            dataset_name="demo",
            split="train",
            stem="a",
            image_path=Path("/tmp/image.jpg"),
            label_path=Path("/tmp/image.txt"),
        )
        sample_b = PoseDatasetSample(
            dataset_name="demo",
            split="train",
            stem="b",
            image_path=Path("/tmp/image.jpg"),
            label_path=Path("/tmp/image.txt"),
        )
        with self.assertRaises(ValueError):
            ensure_unique_target_names({"train": [sample_a, sample_b], "valid": [], "test": []})


if __name__ == "__main__":
    unittest.main()
