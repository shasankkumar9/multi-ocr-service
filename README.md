# MultiOCR Service

Simple OCR backend with FastAPI and pluggable engines.

## What It Does

- `POST /ocr` runs OCR with one engine.
- `POST /benchmark` runs OCR across all engines.
- Optional preprocessing is supported via form fields.

## Engines

- PaddleOCR
- Tesseract
- EasyOCR

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Docs: `http://localhost:8000/docs`

## Run With Docker

```bash
docker compose up --build
```

## Tests

```bash
pytest
```
