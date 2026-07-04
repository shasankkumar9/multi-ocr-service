import cv2
import numpy as np
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from app.main import app as fastapi_app
from app.services import BenchmarkResult, OCRRunResult

client = TestClient(fastapi_app)
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_images"


def _sample_image_paths() -> list[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
    return sorted([path for path in SAMPLE_DIR.iterdir() if path.is_file() and path.suffix.lower() in allowed])


def _content_type_for(path: Path) -> str:
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mapping.get(path.suffix.lower(), "application/octet-stream")


def _sample_png() -> bytes:
    image = np.full((80, 220, 3), 255, dtype=np.uint8)
    cv2.putText(image, "HELLO", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_root_health_engines() -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert "service" in root.json()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    engines = client.get("/engines")
    assert engines.status_code == 200
    assert isinstance(engines.json()["engines"], list)


def test_sample_images_exist() -> None:
    assert SAMPLE_DIR.exists()
    assert _sample_image_paths()


def test_ocr_endpoint_with_mocked_service(monkeypatch) -> None:
    def fake_run_ocr(**kwargs):
        _ = kwargs
        return OCRRunResult(
            engine="paddle",
            processing_time_ms=12.3,
            confidence=0.94,
            detected_text="hello",
            bounding_boxes=[[[0, 0], [10, 0], [10, 10], [0, 10]]],
            detected_lines=[{"text": "hello", "confidence": 0.94,
                             "bounding_box": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
            width=220,
            height=80,
            preprocessing_steps=["grayscale"],
        )

    monkeypatch.setattr(api_module._ocr_service, "run_ocr", fake_run_ocr)

    response = client.post(
        "/ocr",
        files={"file": ("sample.png", _sample_png(), "image/png")},
        data={"engine": "paddle", "preprocessing_enabled": "true", "grayscale": "true"},
    )
    assert response.status_code == 200
    assert response.json()["engine_used"] == "paddle"


def test_benchmark_endpoint_with_mocked_service(monkeypatch) -> None:
    def fake_benchmark(**kwargs):
        _ = kwargs
        return [
            BenchmarkResult(
                engine="paddle",
                success=True,
                execution_time_ms=10.2,
                average_confidence=0.9,
                extracted_characters=100,
                detected_words=20,
                detected_lines=4,
            )
        ]

    monkeypatch.setattr(api_module._benchmark_service, "run", fake_benchmark)

    response = client.post(
        "/benchmark",
        files={"file": ("sample.png", _sample_png(), "image/png")},
        data={"preprocessing_enabled": "false"},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["engine"] == "paddle"


@pytest.mark.parametrize("image_path", _sample_image_paths(), ids=lambda p: p.name)
def test_benchmark_endpoint_with_real_sample_images(image_path: Path) -> None:
    with image_path.open("rb") as handle:
        payload = handle.read()

    response = client.post(
        "/benchmark",
        files={"file": (image_path.name, payload, _content_type_for(image_path))},
        data={"preprocessing_enabled": "true", "grayscale": "true", "threshold": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert len(body["results"]) >= 1
    assert all("engine" in item and "success" in item for item in body["results"])


@pytest.mark.parametrize("image_path", _sample_image_paths(), ids=lambda p: p.name)
def test_ocr_endpoint_with_real_sample_images(image_path: Path) -> None:
    engines_response = client.get("/engines")
    assert engines_response.status_code == 200
    available_engines = [item["name"] for item in engines_response.json()["engines"] if item["available"]]

    if not available_engines:
        pytest.skip("No OCR engine available in this environment")

    selected_engine = available_engines[0]
    with image_path.open("rb") as handle:
        payload = handle.read()

    response = client.post(
        "/ocr",
        files={"file": (image_path.name, payload, _content_type_for(image_path))},
        data={
            "engine": selected_engine,
            "language": "en",
            "preprocessing_enabled": "true",
            "grayscale": "true",
            "threshold": "true",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["engine_used"] == selected_engine
    assert "detected_text" in body
    assert "detected_lines" in body
