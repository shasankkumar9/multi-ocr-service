# Contributing

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Guidelines

- Keep changes small and focused.
- Add or update tests with behavior changes.
- Keep engine-specific code in `app/engines/`.
- Keep service logic in `app/services.py`.
