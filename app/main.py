"""tack45-caller application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api import APIError, router
from app.config import Settings
from app.db import Database
from app.prewarm import PrewarmPool
from app.realtime_ws import default_provider_factory
from app.twilio_gateway import TwilioGateway


def create_app(
    settings: Settings | None = None,
    http_transport: httpx.BaseTransport | None = None,
    provider_factory=None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
        database = Database(settings)
        await database.connect()
        client = httpx.AsyncClient(transport=http_transport, timeout=30)
        app.state.settings = settings
        app.state.db = database
        app.state.http = client
        app.state.gateway = TwilioGateway(settings, client)
        app.state.provider_factory = provider_factory or default_provider_factory
        app.state.prewarm = PrewarmPool()
        try:
            yield
        finally:
            await app.state.prewarm.aclose()
            await client.aclose()
            await database.dispose()

    app = FastAPI(title="tack45-caller", version=__version__, lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(APIError)
    async def on_api_error(_request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={"detail": exc.message, "code": exc.code},
        )

    return app


app = create_app()
