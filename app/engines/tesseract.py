"""Tesseract OCR engine adapter."""

from __future__ import annotations

from collections import defaultdict

from app.engines.base import AbstractOCREngine, OCRLine, OCRResult
from app.utils import decode_image_bytes


class TesseractOCREngine(AbstractOCREngine):
    @property
    def name(self) -> str:
        return "tesseract"

    def supported_languages(self) -> list[str]:
        return ["eng", "fra", "deu", "spa", "ita", "por"]

    def is_available(self) -> bool:
        try:
            import pytesseract  # type: ignore

            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def availability_details(self) -> str:
        return "Tesseract available" if self.is_available() else "Tesseract runtime not available"

    def extract_text(self, payload: bytes, *, language: str | None = None) -> OCRResult:
        image = decode_image_bytes(payload)
        selected_language = language if language in self.supported_languages() else "eng"

        import pytesseract  # type: ignore

        data = pytesseract.image_to_data(
            image,
            lang=selected_language,
            output_type=pytesseract.Output.DICT,
        )

        line_map: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
        confidences: list[float] = []
        all_boxes: list[list[list[float]]] = []

        total = len(data.get("text", []))
        for index in range(total):
            text = str(data["text"][index]).strip()
            if not text:
                continue

            conf = float(data["conf"][index]) if str(data["conf"][index]).strip() else -1.0
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])

            box = [
                [float(left), float(top)],
                [float(left + width), float(top)],
                [float(left + width), float(top + height)],
                [float(left), float(top + height)],
            ]
            all_boxes.append(box)
            if conf >= 0:
                confidences.append(conf)

            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            line_map[key].append({"text": text, "conf": conf, "box": box})

        lines: list[OCRLine] = []
        for _, words in sorted(line_map.items()):
            line_text = " ".join(str(item["text"]) for item in words).strip()
            line_conf_values = [float(item["conf"]) for item in words if float(item["conf"]) >= 0]
            line_confidence = sum(line_conf_values) / len(line_conf_values) if line_conf_values else None

            boxes = [item["box"] for item in words]
            x_values = [point[0] for box in boxes for point in box]
            y_values = [point[1] for box in boxes for point in box]
            merged_box = [
                [min(x_values), min(y_values)],
                [max(x_values), min(y_values)],
                [max(x_values), max(y_values)],
                [min(x_values), max(y_values)],
            ]
            lines.append(OCRLine(text=line_text, confidence=line_confidence, bounding_box=merged_box))

        average_confidence = sum(confidences) / len(confidences) if confidences else None
        height, width = image.shape[:2]
        return OCRResult(
            text="\n".join(line.text for line in lines),
            lines=lines,
            average_confidence=average_confidence,
            bounding_boxes=all_boxes,
            width=width,
            height=height,
        )
