#!/usr/bin/env python3
"""Shared grayscale preprocessing helpers for training and evaluation."""

from __future__ import annotations

import cv2
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_grayscale_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale and expand back to 3 BGR channels."""
    if image is None:
        raise ValueError("Image can not be None.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[..., 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@contextmanager
def patch_ultralytics_dataset_grayscale(enabled: bool) -> Iterator[None]:
    """Temporarily patch Ultralytics dataset loading to return grayscale BGR images."""
    if not enabled:
        yield
        return

    from ultralytics.data.base import BaseDataset

    original_load_image = BaseDataset.load_image

    def patched_load_image(self, i: int, rect_mode: bool = True):
        image, hw_original, hw_resized = original_load_image(self, i, rect_mode)
        return to_grayscale_bgr(image), hw_original, hw_resized

    BaseDataset.load_image = patched_load_image
    try:
        yield
    finally:
        BaseDataset.load_image = original_load_image


def collect_image_sources(source_path: Path) -> list[Path]:
    """Collect image files from a file or directory source."""
    if source_path.is_file():
        return [source_path.resolve()]
    if source_path.is_dir():
        return sorted(path.resolve() for path in source_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"Prediction source does not exist: {source_path}")


@contextmanager
def grayscale_prediction_sources(source_paths: list[Path], enabled: bool) -> Iterator[list[str]]:
    """Create temporary grayscale copies of prediction sources when requested."""
    if not enabled:
        yield [str(path) for path in source_paths]
        return

    with TemporaryDirectory(prefix="tech_core_gray_predict_", dir="/tmp") as temp_dir:
        temp_root = Path(temp_dir)
        temp_paths: list[str] = []
        for source_path in source_paths:
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to read image for grayscale preprocessing: {source_path}")
            target_path = temp_root / source_path.name
            if not cv2.imwrite(str(target_path), to_grayscale_bgr(image)):
                raise IOError(f"Failed to save grayscale temporary image: {target_path}")
            temp_paths.append(str(target_path))
        yield temp_paths
