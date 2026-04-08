from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from pose_offline_aug.io import object_annotation_to_yolo_pose_line
from pose_offline_aug.labels import transform_keypoints
from pose_offline_aug.structures import BBox, Keypoint, ObjectAnnotation, OcclusionRegion
from repartition_pose_dataset import ensure_unique_target_names, largest_remainder_counts


class OfflineAugTests(unittest.TestCase):
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
