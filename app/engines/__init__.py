"""OCR engine adapters."""

from app.engines.base import AbstractOCREngine, OCRLine, OCRResult
from app.engines.easyocr import EasyOCREngine
from app.engines.paddle import PaddleOCREngine
from app.engines.tesseract import TesseractOCREngine

__all__ = [
    "AbstractOCREngine",
    "OCRLine",
    "OCRResult",
    "PaddleOCREngine",
    "TesseractOCREngine",
    "EasyOCREngine",
]
