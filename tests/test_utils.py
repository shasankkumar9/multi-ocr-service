from pathlib import Path

import cv2
import numpy as np
import pytest

from app.utils import (
    FileDecodeError,
    decode_image_bytes,
    encode_png,
    image_dimensions,
    load_upload_image,
    normalize_content_type,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_images"


def _sample_image_paths() -> list[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
    return sorted([path for path in SAMPLE_DIR.iterdir() if path.is_file() and path.suffix.lower() in allowed])


def _content_type_for(path: Path) -> str:
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mapping[path.suffix.lower()]


def test_normalize_content_type() -> None:
    assert normalize_content_type(" IMAGE/PNG ") == "image/png"
    assert normalize_content_type(None) == ""


def test_decode_image_bytes_rejects_empty_payload() -> None:
    with pytest.raises(FileDecodeError):
        decode_image_bytes(b"")


def test_encode_png_and_decode_roundtrip() -> None:
    image = np.full((50, 100, 3), 255, dtype=np.uint8)
    cv2.putText(image, "A", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    encoded = encode_png(image)
    decoded = decode_image_bytes(encoded)

    assert decoded.shape == image.shape


@pytest.mark.parametrize("image_path", _sample_image_paths(), ids=lambda p: p.name)
def test_load_upload_image_from_real_samples(image_path: Path) -> None:
    with image_path.open("rb") as handle:
        payload = handle.read()

    image = load_upload_image(payload, _content_type_for(image_path))
    width, height = image_dimensions(image)

    assert image is not None
    assert width > 0
    assert height > 0


def test_load_upload_image_rejects_unknown_content_type() -> None:
    with pytest.raises(FileDecodeError):
        load_upload_image(b"123", "application/octet-stream")
