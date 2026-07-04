"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app import config
from app.api import router

app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.include_router(router)
