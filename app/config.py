"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

APP_NAME: str = os.getenv("MULTIOCR_APP_NAME", "MultiOCR Service")
APP_VERSION: str = os.getenv("MULTIOCR_APP_VERSION", "1.1.0")
APP_DESCRIPTION: str = os.getenv(
    "MULTIOCR_APP_DESCRIPTION",
    "Production-ready OCR backend with pluggable OCR engines.",
)
ENVIRONMENT: str = os.getenv("MULTIOCR_ENVIRONMENT", "local")
HOST: str = os.getenv("MULTIOCR_HOST", "0.0.0.0")
PORT: int = int(os.getenv("MULTIOCR_PORT", "8000"))
DEBUG: bool = os.getenv("MULTIOCR_DEBUG", "false").lower() == "true"
DEFAULT_ENGINE: str = os.getenv("MULTIOCR_DEFAULT_ENGINE", "paddle")

SUPPORTED_IMAGE_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
)
SUPPORTED_PDF_TYPES: tuple[str, ...] = ("application/pdf",)
