"""Business services and engine factory."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from app import preprocessing
from app.engines import AbstractOCREngine, EasyOCREngine, PaddleOCREngine, TesseractOCREngine
from app.utils import FileDecodeError, encode_png, image_dimensions, load_upload_image


class ServiceError(Exception):
    """Base service-layer error with stable API code and status."""

    def __init__(self, code: str, message: str, status_code: int, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class EngineNotFoundError(ServiceError):
    def __init__(self, engine: str, available: list[str]) -> None:
        super().__init__(
            code="ENGINE_NOT_FOUND",
            message=f"Unsupported OCR engine: {engine}",
            status_code=404,
            details={"engine": engine, "available_engines": available},
        )


class EngineUnavailableError(ServiceError):
    def __init__(self, engine: str, reason: str) -> None:
        super().__init__(
            code="ENGINE_UNAVAILABLE",
            message=f"OCR engine '{engine}' is unavailable",
            status_code=503,
            details={"engine": engine, "reason": reason},
        )


class OCRExecutionError(ServiceError):
    def __init__(self, engine: str, reason: str) -> None:
        super().__init__(
            code="OCR_EXECUTION_FAILED",
            message=f"OCR execution failed for engine '{engine}'",
            status_code=500,
            details={"engine": engine, "reason": reason},
        )


class PayloadError(ServiceError):
    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(
            code="INVALID_PAYLOAD",
            message=message,
            status_code=400,
            details=details or {},
        )


@dataclass(slots=True)
class OCRRunResult:
    engine: str
    processing_time_ms: float
    confidence: float | None
    detected_text: str
    bounding_boxes: list[list[list[float]]]
    detected_lines: list[dict[str, object]]
    width: int | None
    height: int | None
    preprocessing_steps: list[str]


@dataclass(slots=True)
class BenchmarkResult:
    engine: str
    success: bool
    execution_time_ms: float
    average_confidence: float | None
    extracted_characters: int
    detected_words: int
    detected_lines: int
    error: str | None = None


class OCRFactory:
    """Simple factory with explicit engine registry."""

    def __init__(self) -> None:
        self._registry: dict[str, Callable[[], AbstractOCREngine]] = {
            "paddle": PaddleOCREngine,
            "tesseract": TesseractOCREngine,
            "easyocr": EasyOCREngine,
        }

    def register(self, name: str, constructor: Callable[[], AbstractOCREngine]) -> None:
        self._registry[name.strip().lower()] = constructor

    def create(self, name: str) -> AbstractOCREngine:
        key = name.strip().lower()
        constructor = self._registry.get(key)
        if constructor is None:
            raise EngineNotFoundError(name, self.available_engines())
        return constructor()

    def available_engines(self) -> list[str]:
        return sorted(self._registry.keys())


class OCRService:
    """Coordinates parsing, preprocessing, and OCR execution."""

    def __init__(self, factory: OCRFactory) -> None:
        self._factory = factory

    def list_engines(self) -> list[dict[str, object]]:
        engines: list[dict[str, object]] = []
        for name in self._factory.available_engines():
            engine = self._factory.create(name)
            engines.append(
                {
                    "name": engine.name,
                    "available": engine.is_available(),
                    "details": engine.availability_details(),
                    "supported_languages": engine.supported_languages(),
                }
            )
        return engines

    def run_ocr(
        self,
        *,
        payload: bytes,
        content_type: str | None,
        engine_name: str,
        language: str | None,
        preprocessing_enabled: bool,
        use_deskew: bool,
        resize_width: int | None,
        use_grayscale: bool,
        use_contrast: bool,
        use_threshold: bool,
        use_denoise: bool,
        rotate_angle: float | None,
    ) -> OCRRunResult:
        if not payload:
            raise PayloadError("Uploaded file is empty")

        try:
            image = load_upload_image(payload, content_type)
        except FileDecodeError as exc:
            raise PayloadError(str(exc), {"content_type": content_type}) from exc

        steps: list[str] = []
        if preprocessing_enabled:
            image, steps = preprocessing.apply_preprocessing(
                image,
                use_deskew=use_deskew,
                resize_width=resize_width,
                use_grayscale=use_grayscale,
                use_contrast=use_contrast,
                use_threshold=use_threshold,
                use_denoise=use_denoise,
                rotate_angle=rotate_angle,
            )

        encoded_image = encode_png(image)
        width, height = image_dimensions(image)

        engine = self._factory.create(engine_name)
        if not engine.is_available():
            raise EngineUnavailableError(engine.name, engine.availability_details())

        started = time.perf_counter()
        try:
            result = engine.extract_text(encoded_image, language=language)
        except ServiceError:
            raise
        except Exception as exc:
            raise OCRExecutionError(engine.name, str(exc)) from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        line_payload = [
            {
                "text": line.text,
                "confidence": line.confidence,
                "bounding_box": line.bounding_box,
            }
            for line in result.lines
        ]

        return OCRRunResult(
            engine=engine.name,
            processing_time_ms=elapsed_ms,
            confidence=result.average_confidence,
            detected_text=result.text,
            bounding_boxes=result.bounding_boxes,
            detected_lines=line_payload,
            width=result.width or width,
            height=result.height or height,
            preprocessing_steps=steps,
        )


class BenchmarkService:
    """Runs OCR with each registered engine and returns comparable metrics."""

    def __init__(self, factory: OCRFactory, ocr_service: OCRService) -> None:
        self._factory = factory
        self._ocr_service = ocr_service

    def run(
        self,
        *,
        payload: bytes,
        content_type: str | None,
        language: str | None,
        preprocessing_enabled: bool,
        use_deskew: bool,
        resize_width: int | None,
        use_grayscale: bool,
        use_contrast: bool,
        use_threshold: bool,
        use_denoise: bool,
        rotate_angle: float | None,
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for engine_name in self._factory.available_engines():
            try:
                response = self._ocr_service.run_ocr(
                    payload=payload,
                    content_type=content_type,
                    engine_name=engine_name,
                    language=language,
                    preprocessing_enabled=preprocessing_enabled,
                    use_deskew=use_deskew,
                    resize_width=resize_width,
                    use_grayscale=use_grayscale,
                    use_contrast=use_contrast,
                    use_threshold=use_threshold,
                    use_denoise=use_denoise,
                    rotate_angle=rotate_angle,
                )
                results.append(
                    BenchmarkResult(
                        engine=engine_name,
                        success=True,
                        execution_time_ms=response.processing_time_ms,
                        average_confidence=response.confidence,
                        extracted_characters=len(response.detected_text),
                        detected_words=len(response.detected_text.split()),
                        detected_lines=len(response.detected_lines),
                    )
                )
            except ServiceError as exc:
                results.append(
                    BenchmarkResult(
                        engine=engine_name,
                        success=False,
                        execution_time_ms=0.0,
                        average_confidence=None,
                        extracted_characters=0,
                        detected_words=0,
                        detected_lines=0,
                        error=exc.message,
                    )
                )

        return results
