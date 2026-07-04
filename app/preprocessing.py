"""Reusable image preprocessing functions."""

from __future__ import annotations

import cv2
import numpy as np


def deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.2:
        return image

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def resize(image: np.ndarray, width: int | None = None) -> np.ndarray:
    if not width:
        return image
    height, current_width = image.shape[:2]
    ratio = width / current_width
    new_height = max(1, int(height * ratio))
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_CUBIC)


def threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        11,
    )
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(image, 3)


def contrast(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    channels[0] = cv2.equalizeHist(channels[0])
    merged = cv2.merge(channels)
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def rotate(image: np.ndarray, angle: float | None) -> np.ndarray:
    if angle is None:
        return image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def apply_preprocessing(
    image: np.ndarray,
    *,
    use_deskew: bool = False,
    resize_width: int | None = None,
    use_grayscale: bool = False,
    use_contrast: bool = False,
    use_threshold: bool = False,
    use_denoise: bool = False,
    rotate_angle: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Apply enabled preprocessing steps in a fixed, readable order."""

    out = image.copy()
    steps: list[str] = []

    if use_deskew:
        out = deskew(out)
        steps.append("deskew")
    if resize_width:
        out = resize(out, resize_width)
        steps.append("resize")
    if use_grayscale:
        out = to_grayscale(out)
        steps.append("grayscale")
    if use_contrast:
        out = contrast(out)
        steps.append("contrast")
    if use_threshold:
        out = threshold(out)
        steps.append("threshold")
    if use_denoise:
        out = denoise(out)
        steps.append("denoise")
    if rotate_angle is not None:
        out = rotate(out, rotate_angle)
        steps.append("rotate")

    return out, steps
