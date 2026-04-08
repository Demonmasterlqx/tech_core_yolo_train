from __future__ import annotations

import cv2
import numpy as np


def adjust_brightness_contrast(image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    adjusted = image.astype(np.float32) * contrast + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(1e-6, gamma)
    lookup = np.array([((index / 255.0) ** (1.0 / gamma)) * 255.0 for index in range(256)], dtype=np.float32)
    table = np.clip(lookup, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def adjust_hsv(image: np.ndarray, hue_delta: float, sat_scale: float, val_scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return image.copy()
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0.0)


def add_gaussian_noise(image: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0.0:
        return image.copy()
    noise = rng.normal(0.0, std, size=image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    quality = min(100, max(5, int(quality)))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG during compression augmentation.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Failed to decode JPEG during compression augmentation.")
    return decoded


def apply_shadow(
    image: np.ndarray,
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
    darkness: float,
    blur_kernel: int,
) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        (max(1, int(round(axes[0]))), max(1, int(round(axes[1])))),
        angle_deg,
        0.0,
        360.0,
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    blur_kernel = max(1, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if blur_kernel > 1:
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), sigmaX=0.0)
    alpha = (mask.astype(np.float32) / 255.0) * np.clip(darkness, 0.0, 1.0)
    shadow = image.astype(np.float32)
    shadow *= 1.0 - alpha[..., None]
    return np.clip(shadow, 0, 255).astype(np.uint8)

