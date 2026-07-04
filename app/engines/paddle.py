"""PaddleOCR engine adapter."""

from __future__ import annotations

from functools import lru_cache

from app.engines.base import AbstractOCREngine, OCRLine, OCRResult
from app.utils import decode_image_bytes


@lru_cache(maxsize=8)
def _build_client(language: str):
    from paddleocr import PaddleOCR  # type: ignore

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        lang=language,
        device='cpu'
    )


class PaddleOCREngine(AbstractOCREngine):
    @property
    def name(self) -> str:
        return "paddle"

    def supported_languages(self) -> list[str]:
        return ["en", "ch", "fr", "de", "es"]

    def is_available(self) -> bool:
        try:
            import paddleocr  # type: ignore
            import paddle  # type: ignore

            return paddleocr is not None and paddle is not None
        except Exception:
            return False

    def availability_details(self) -> str:
        try:
            import paddleocr  # type: ignore
        except Exception:
            return "PaddleOCR not installed"

        try:
            import paddle  # type: ignore
        except Exception:
            return "PaddleOCR requires paddlepaddle (not installed)"

        return "PaddleOCR available"

    def _resolve_language(self, language: str | None) -> str:
        """Map user input to a supported Paddle language code."""

        candidate = (language or "en").strip().lower()
        if candidate in {"", "string", "none", "null"}:
            return "en"
        if candidate in self.supported_languages():
            return candidate
        return "en"

    @staticmethod
    def _to_float(value: object) -> float | None:
        """Convert value to float safely."""

        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                return float(candidate)
            except Exception:
                return None

        try:
            return float(str(value))
        except Exception:
            return None

    def extract_text(self, payload: bytes, *, language: str | None = None) -> OCRResult:
        image = decode_image_bytes(payload)
        selected_language = self._resolve_language(language)
        client = _build_client(selected_language)
        raw_result = client.predict(image)

        lines: list[OCRLine] = []
        boxes: list[list[list[float]]] = []
        confidences: list[float] = []

        # PaddleOCR result shapes vary between versions.
        # Support both:
        # 1) New dict style from predict(): [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
        # 2) Older nested tuple/list style from ocr().
        for block in raw_result or []:
            if isinstance(block, dict):
                texts = block.get("rec_texts") or []
                scores = block.get("rec_scores") or []
                polys = block.get("rec_polys") or []

                for index, text_value in enumerate(texts):
                    text = str(text_value).strip()
                    if not text:
                        continue

                    score_value = scores[index] if index < len(scores) else None
                    confidence = self._to_float(score_value)

                    poly_value = polys[index] if index < len(polys) else None
                    normalized_box: list[list[float]] | None = None
                    if poly_value is not None:
                        try:
                            normalized_box = [
                                [float(point[0]), float(point[1])] for point in poly_value
                            ]
                        except Exception:
                            normalized_box = None

                    lines.append(OCRLine(text=text, confidence=confidence, bounding_box=normalized_box))
                    if normalized_box:
                        boxes.append(normalized_box)
                    if confidence is not None:
                        confidences.append(confidence)
                continue

            for entry in block or []:
                box = [[float(point[0]), float(point[1])] for point in entry[0]]
                text = str(entry[1][0]).strip()
                confidence = self._to_float(entry[1][1])
                if not text:
                    continue
                lines.append(OCRLine(text=text, confidence=confidence, bounding_box=box))
                boxes.append(box)
                if confidence is not None:
                    confidences.append(confidence)

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
