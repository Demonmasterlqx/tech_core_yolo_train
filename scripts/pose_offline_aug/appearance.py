from __future__ import annotations

import cv2
import numpy as np


def normalize_kernel_size(kernel_size: int) -> int:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


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


def apply_temperature_tint(image: np.ndarray, temperature_shift: float, tint_shift: float) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] = np.clip(lab[..., 1] + float(tint_shift), 0.0, 255.0)
    lab[..., 2] = np.clip(lab[..., 2] + float(temperature_shift), 0.0, 255.0)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_highlight_exposure(
    image: np.ndarray,
    gain: float,
    threshold: int,
    rolloff: float,
) -> np.ndarray:
    threshold = min(255, max(0, int(threshold)))
    rolloff = max(1e-6, float(rolloff))
    highlight = image.astype(np.float32)
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    alpha = np.clip((luminance - threshold) / rolloff, 0.0, 1.0)
    boosted = highlight * float(gain)
    mixed = highlight * (1.0 - alpha[..., None]) + boosted * alpha[..., None]
    return np.clip(mixed, 0, 255).astype(np.uint8)


def apply_unsharp_mask(image: np.ndarray, kernel_size: int, amount: float) -> np.ndarray:
    kernel_size = normalize_kernel_size(kernel_size)
    if kernel_size <= 1 or amount <= 0.0:
        return image.copy()
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0.0)
    sharpened = image.astype(np.float32) * (1.0 + float(amount)) - blurred.astype(np.float32) * float(amount)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = normalize_kernel_size(kernel_size)
    if kernel_size <= 1:
        return image.copy()
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0.0)


def motion_blur(image: np.ndarray, kernel_size: int, angle_deg: float) -> np.ndarray:
    kernel_size = normalize_kernel_size(kernel_size)
    if kernel_size <= 1:
        return image.copy()

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    center = ((kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        return image.copy()
    kernel /= kernel_sum
    return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)


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


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    tile_grid_size = max(1, int(tile_grid_size))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip_limit)), tileGridSize=(tile_grid_size, tile_grid_size))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_channel_gain(image: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    adjusted = image.astype(np.float32).copy()
    for channel_index, gain in enumerate(gains):
        adjusted[..., channel_index] *= float(gain)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_rgb_shift(image: np.ndarray, shifts: tuple[float, float, float]) -> np.ndarray:
    shifted = image.astype(np.float32).copy()
    for channel_index, shift in enumerate(shifts):
        shifted[..., channel_index] += float(shift)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_dominant_channel(
    image: np.ndarray,
    dominant_index: int,
    dominant_gain: float,
    other_gain: float,
    bias: float,
) -> np.ndarray:
    adjusted = image.astype(np.float32).copy()
    for channel_index in range(3):
        gain = dominant_gain if channel_index == int(dominant_index) else other_gain
        adjusted[..., channel_index] = adjusted[..., channel_index] * float(gain) + float(bias)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_channel_shuffle(image: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    return image[..., list(order)].copy()


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
    blur_kernel = normalize_kernel_size(blur_kernel)
    if blur_kernel > 1:
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), sigmaX=0.0)
    alpha = (mask.astype(np.float32) / 255.0) * np.clip(darkness, 0.0, 1.0)
    shadow = image.astype(np.float32)
    shadow *= 1.0 - alpha[..., None]
    return np.clip(shadow, 0, 255).astype(np.uint8)
