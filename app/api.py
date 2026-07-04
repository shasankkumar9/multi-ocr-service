"""HTTP API routes."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app import config
from app.schemas import (
    APIError,
    BenchmarkItemResponse,
    BenchmarkResponse,
    EngineInfo,
    EnginesResponse,
    ErrorResponse,
    HealthResponse,
    ImageDimensions,
    OCRLineResponse,
    OCRResponse,
    RootResponse,
)
from app.services import BenchmarkService, OCRFactory, OCRService, ServiceError

router = APIRouter()
_factory = OCRFactory()
_ocr_service = OCRService(_factory)
_benchmark_service = BenchmarkService(_factory, _ocr_service)


def _to_http_error(exc: ServiceError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail=ErrorResponse(error=APIError(code=exc.code, message=exc.message, details=exc.details)).model_dump()[
            "error"
        ],
    )


@router.get("/", response_model=RootResponse, tags=["System"])
def root() -> RootResponse:
    return RootResponse(service=config.APP_NAME, version=config.APP_VERSION, docs_url="/docs")


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", environment=config.ENVIRONMENT)


@router.get("/engines", response_model=EnginesResponse, tags=["OCR"])
def engines() -> EnginesResponse:
    return EnginesResponse(engines=[EngineInfo(**item) for item in _ocr_service.list_engines()])


@router.post("/ocr", response_model=OCRResponse, tags=["OCR"])
async def ocr(
    file: UploadFile = File(..., description="Image or PDF file"),
    engine: str = Form(config.DEFAULT_ENGINE),
    language: str | None = Form(None),
    preprocessing_enabled: bool = Form(False),
    deskew: bool = Form(False),
    resize_width: int | None = Form(None),
    grayscale: bool = Form(False),
    contrast: bool = Form(False),
    threshold: bool = Form(False),
    denoise: bool = Form(False),
    rotate_angle: float | None = Form(None),
) -> OCRResponse:
    payload = await file.read()
    try:
        result = _ocr_service.run_ocr(
            payload=payload,
            content_type=file.content_type,
            engine_name=engine,
            language=language,
            preprocessing_enabled=preprocessing_enabled,
            use_deskew=deskew,
            resize_width=resize_width,
            use_grayscale=grayscale,
            use_contrast=contrast,
            use_threshold=threshold,
            use_denoise=denoise,
            rotate_angle=rotate_angle,
        )
    except ServiceError as exc:
        raise _to_http_error(exc) from exc

    return OCRResponse(
        engine_used=result.engine,
        processing_time_ms=result.processing_time_ms,
        confidence=result.confidence,
        detected_text=result.detected_text,
        bounding_boxes=result.bounding_boxes,
        detected_lines=[OCRLineResponse(**line) for line in result.detected_lines],
        image_dimensions=ImageDimensions(width=result.width, height=result.height),
        preprocessing_steps=result.preprocessing_steps,
    )


@router.post("/benchmark", response_model=BenchmarkResponse, tags=["OCR"])
async def benchmark(
    file: UploadFile = File(..., description="Image or PDF file"),
    language: str | None = Form(None),
    preprocessing_enabled: bool = Form(False),
    deskew: bool = Form(False),
    resize_width: int | None = Form(None),
    grayscale: bool = Form(False),
    contrast: bool = Form(False),
    threshold: bool = Form(False),
    denoise: bool = Form(False),
    rotate_angle: float | None = Form(None),
) -> BenchmarkResponse:
    payload = await file.read()
    try:
        rows = _benchmark_service.run(
            payload=payload,
            content_type=file.content_type,
            language=language,
            preprocessing_enabled=preprocessing_enabled,
            use_deskew=deskew,
            resize_width=resize_width,
            use_grayscale=grayscale,
            use_contrast=contrast,
            use_threshold=threshold,
            use_denoise=denoise,
            rotate_angle=rotate_angle,
        )
    except ServiceError as exc:
        raise _to_http_error(exc) from exc

    return BenchmarkResponse(results=[BenchmarkItemResponse(**asdict(row)) for row in rows])
