"""Utility helpers without business logic."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import SUPPORTED_IMAGE_TYPES, SUPPORTED_PDF_TYPES


class FileDecodeError(ValueError):
    """Raised when an uploaded file cannot be decoded into an image."""


def normalize_content_type(content_type: str | None) -> str:
    """Normalize content type for branching logic."""

    return (content_type or "").lower().strip()


def decode_image_bytes(payload: bytes) -> np.ndarray:
    """Decode encoded image bytes into BGR OpenCV image."""

    if not payload:
        raise FileDecodeError("Empty file payload")

    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise FileDecodeError("Failed to decode image bytes")
    return image


def encode_png(image: np.ndarray) -> bytes:
    """Encode BGR image to PNG bytes."""

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise FileDecodeError("Failed to encode image")
    return encoded.tobytes()


def pdf_first_page_to_image(payload: bytes) -> np.ndarray:
    """Render the first page of a PDF payload into BGR OpenCV image."""

    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as exc:
        raise FileDecodeError("PDF support requires pypdfium2") from exc

    try:
        document = pdfium.PdfDocument(payload)
        if len(document) == 0:
            raise FileDecodeError("PDF has no pages")

        page = document[0]
        rendered = page.render(scale=2.0)
        pil_image = rendered.to_pil().convert("RGB")
        rgb = np.asarray(pil_image)
        page.close()
        document.close()

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except FileDecodeError:
        raise
    except Exception as exc:
        raise FileDecodeError(f"Failed to render PDF: {exc}") from exc


def load_upload_image(payload: bytes, content_type: str | None) -> np.ndarray:
    """Decode upload payload into image from supported image/PDF content types."""

    normalized = normalize_content_type(content_type)

    if not normalized or normalized in SUPPORTED_IMAGE_TYPES:
        return decode_image_bytes(payload)
    if normalized in SUPPORTED_PDF_TYPES:
        return pdf_first_page_to_image(payload)

    raise FileDecodeError(f"Unsupported content type: {content_type}")


def image_dimensions(image: np.ndarray) -> tuple[int, int]:
    """Return image dimensions as (width, height)."""

    height, width = image.shape[:2]
    return width, height
