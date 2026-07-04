from app.engines.base import AbstractOCREngine, OCRResult
from app.services import OCRFactory


class DummyEngine(AbstractOCREngine):
    @property
    def name(self) -> str:
        return "dummy"

    def supported_languages(self) -> list[str]:
        return ["en"]

    def is_available(self) -> bool:
        return True

    def availability_details(self) -> str:
        return "ok"

    def extract_text(self, payload: bytes, *, language: str | None = None) -> OCRResult:
        _ = payload
        _ = language
        return OCRResult(text="dummy")


def test_factory_lists_default_engines() -> None:
    factory = OCRFactory()
    names = factory.available_engines()

    assert "paddle" in names
    assert "tesseract" in names
    assert "easyocr" in names


def test_factory_registers_new_engine() -> None:
    factory = OCRFactory()
    factory.register("dummy", DummyEngine)

    engine = factory.create("dummy")
    assert engine.name == "dummy"
    assert engine.is_available() is True
