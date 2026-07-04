import cv2
import numpy as np
from pathlib import Path

import pytest

from app.preprocessing import apply_preprocessing, contrast, denoise, deskew, resize, threshold

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_images"


def _sample_image_paths() -> list[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
    return sorted([path for path in SAMPLE_DIR.iterdir() if path.is_file() and path.suffix.lower() in allowed])


def _image() -> np.ndarray:
    image = np.full((120, 240, 3), 255, dtype=np.uint8)
    cv2.putText(image, "OCR", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return image


def test_basic_preprocessing_functions() -> None:
    image = _image()

    assert deskew(image).shape == image.shape
    assert resize(image, 100).shape[1] == 100
    assert threshold(image).shape == image.shape
    assert denoise(image).shape == image.shape
    assert contrast(image).shape == image.shape


def test_apply_preprocessing_pipeline() -> None:
    image = _image()
    out, steps = apply_preprocessing(
        image,
        use_deskew=True,
        resize_width=120,
        use_grayscale=True,
        use_contrast=True,
        use_threshold=True,
        use_denoise=True,
        rotate_angle=3.0,
    )

    assert out.shape[1] == 120
    assert "deskew" in steps
    assert "resize" in steps
    assert "grayscale" in steps
    assert "contrast" in steps
    assert "threshold" in steps
    assert "denoise" in steps
    assert "rotate" in steps


@pytest.mark.parametrize("image_path", _sample_image_paths(), ids=lambda p: p.name)
def test_preprocessing_with_real_sample_images(image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    assert image is not None

    out, steps = apply_preprocessing(
        image,
        use_deskew=True,
        resize_width=400,
        use_grayscale=True,
        use_contrast=True,
        use_threshold=True,
        use_denoise=True,
        rotate_angle=2.0,
    )

    assert out is not None
    assert out.shape[2] == 3
    assert len(steps) >= 4
