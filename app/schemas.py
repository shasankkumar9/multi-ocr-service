"""API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    service: str
    version: str
    docs_url: str


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str


class EngineInfo(BaseModel):
    name: str
    available: bool
    details: str
    supported_languages: list[str]


class EnginesResponse(BaseModel):
    engines: list[EngineInfo]


class ImageDimensions(BaseModel):
    width: int | None
    height: int | None


class OCRLineResponse(BaseModel):
    text: str
    confidence: float | None = None
    bounding_box: list[list[float]] | None = None


class OCRResponse(BaseModel):
    engine_used: str
    processing_time_ms: float
    confidence: float | None
    detected_text: str
    bounding_boxes: list[list[list[float]]]
    detected_lines: list[OCRLineResponse]
    image_dimensions: ImageDimensions
    preprocessing_steps: list[str] = Field(default_factory=list)


class BenchmarkItemResponse(BaseModel):
    engine: str
    success: bool
    execution_time_ms: float
    average_confidence: float | None
    extracted_characters: int
    detected_words: int
    detected_lines: int
    error: str | None = None


class BenchmarkResponse(BaseModel):
    results: list[BenchmarkItemResponse]


class APIError(BaseModel):
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: APIError
