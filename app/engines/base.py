"""OCR engine abstraction and normalized OCR result models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class OCRLine:
    text: str
    confidence: float | None = None
    bounding_box: list[list[float]] | None = None


@dataclass(slots=True)
class OCRResult:
    text: str
    lines: list[OCRLine] = field(default_factory=list)
    average_confidence: float | None = None
    bounding_boxes: list[list[list[float]]] = field(default_factory=list)
    width: int | None = None
    height: int | None = None


class AbstractOCREngine(ABC):
    """Contract for all OCR adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def supported_languages(self) -> list[str]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def availability_details(self) -> str:
        ...

    @abstractmethod
    def extract_text(self, payload: bytes, *, language: str | None = None) -> OCRResult:
        ...
