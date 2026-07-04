"""EasyOCR engine adapter."""

from __future__ import annotations

from functools import lru_cache

from app.engines.base import AbstractOCREngine, OCRLine, OCRResult
from app.utils import decode_image_bytes


@lru_cache(maxsize=16)
def _build_client(languages_key: tuple[str, ...]):
    import easyocr  # type: ignore

    return easyocr.Reader(list(languages_key), gpu=False, verbose=False)


class EasyOCREngine(AbstractOCREngine):
    @property
    def name(self) -> str:
        return "easyocr"

    def supported_languages(self) -> list[str]:
        return ["en", "fr", "de", "es", "it", "pt"]

    def is_available(self) -> bool:
        try:
            import easyocr  # type: ignore

            return easyocr is not None
        except Exception:
            return False

    def availability_details(self) -> str:
        return "EasyOCR available" if self.is_available() else "EasyOCR not installed"

    def extract_text(self, payload: bytes, *, language: str | None = None) -> OCRResult:
        image = decode_image_bytes(payload)
        selected_language = language if language in self.supported_languages() else "en"
        client = _build_client((selected_language,))
        raw_result = client.readtext(image)

        lines: list[OCRLine] = []
        boxes: list[list[list[float]]] = []
        confidences: list[float] = []

        for item in raw_result or []:
            box_points, text, confidence = item
            normalized_box = [[float(point[0]), float(point[1])] for point in box_points]
            normalized_text = str(text).strip()
            normalized_confidence = float(confidence)
            if not normalized_text:
                continue

            lines.append(
                OCRLine(
                    text=normalized_text,
                    confidence=normalized_confidence,
                    bounding_box=normalized_box,
                )
            )
            boxes.append(normalized_box)
            confidences.append(normalized_confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        height, width = image.shape[:2]
        return OCRResult(
            text="\n".join(line.text for line in lines),
            lines=lines,
            average_confidence=avg_confidence,
            bounding_boxes=boxes,
            width=width,
            height=height,
        )
