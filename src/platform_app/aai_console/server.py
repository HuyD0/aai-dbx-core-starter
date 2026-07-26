"""FastAPI application for the AAI platform console.

Serves a small server-rendered UI: full pages for navigation, HTML fragments for
in-place swaps, and a narrow JSON surface. There is no bundler and no third-party
client library — `scripts/cloud-verify.sh` performs an offline `uv sync --locked`, so
an npm lockfile ecosystem would be a change of security posture, not a dependency.

Responses are assembled field by field. Never serialise an SDK object wholesale:
`dataclasses.asdict()` recurses into `PlatformSettings.raw`, and `repr=False` does not
prevent it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__
from .checks import (
    PLATFORM_STATE_HEADING,
    PlatformCheck,
    assert_platform_state,
    run_checks,
)
from .config import ConfigError, ConsoleConfig, load_config
from .content import Track, inline_code, resolve_placeholders
from .generate import GenerateError, GenerateRequest, bundle_init
from .registry import TrackRegistry

PACKAGE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger("aai_console")


def _render_tracks(
    tracks: tuple[Track, ...], config: ConsoleConfig
) -> tuple[Track, ...]:
    """Substitute identifier placeholders in every code block."""
    rendered = []
    for track in tracks:
        steps = []
        for step in track.steps:
            blocks = tuple(
                type(block)(
                    lang=block.lang,
                    code=resolve_placeholders(block.code, config),
                )
                for block in step.blocks
            )
            steps.append(type(step)(**{**step.__dict__, "blocks": blocks}))
        rendered.append(type(track)(**{**track.__dict__, "steps": tuple(steps)}))
    return tuple(rendered)


def create_app(
    config: ConsoleConfig | None = None,
    *,
    probe=None,
    registry: TrackRegistry | None = None,
) -> FastAPI:
    app = FastAPI(title="AAI platform console", docs_url=None, redoc_url=None)

    app.state.config = config if config is not None else load_config()
    app.state.probe = probe
    app.state.registry = registry if registry is not None else TrackRegistry.default()

    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    # The only markup content may use. Escapes first, so it cannot inject an element.
    templates.env.filters["icode"] = inline_code
    app.mount(
        "/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static"
    )

    def tracks_for(request: Request) -> tuple[Track, ...]:
        return _render_tracks(
            request.app.state.registry.tracks(), request.app.state.config
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the type only. A traceback reaches the app's Logs tab, which anyone with
        # CAN MANAGE can read, and the process environment holds a live OAuth secret.
        logger.error(
            "unhandled error serving %s: %s", request.url.path, type(exc).__name__
        )
        return JSONResponse({"error": "internal error"}, status_code=500)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/session")
    async def session(request: Request) -> dict:
        config: ConsoleConfig = request.app.state.config
        return {
            "hosted": config.hosted,
            "app_name": config.app_name,
            "version": __version__,
            "capability": "guide-and-generate",
        }

    @app.get("/api/content")
    async def content(request: Request) -> dict:
        return {
            "tracks": [
                {
                    "id": track.id,
                    "title": track.title,
                    "subtitle": track.subtitle,
                    "glyph": track.glyph,
                    "steps": [
                        {"id": step.id, "title": step.title} for step in track.steps
                    ],
                }
                for track in tracks_for(request)
            ]
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        tracks = tracks_for(request)
        return templates.TemplateResponse(
            request,
            "index.html.j2",
            {
                "tracks": tracks,
                "active": tracks[0],
                "session": request.app.state.config,
                "platform_state_heading": PLATFORM_STATE_HEADING,
            },
        )

    @app.get("/track/{track_id}", response_class=HTMLResponse)
    async def track_page(request: Request, track_id: str) -> HTMLResponse:
        tracks = tracks_for(request)
        match = next((t for t in tracks if t.id == track_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail="unknown track")
        return templates.TemplateResponse(
            request,
            "index.html.j2",
            {
                "tracks": tracks,
                "active": match,
                "session": request.app.state.config,
                "platform_state_heading": PLATFORM_STATE_HEADING,
            },
        )

    @app.post("/api/checks/run", response_class=HTMLResponse)
    async def checks(request: Request) -> HTMLResponse:
        results: list[PlatformCheck] = run_checks(
            request.app.state.config, request.app.state.probe
        )
        # Raises if anyone ever tries to present these as the viewer's own access.
        assert_platform_state(results, PLATFORM_STATE_HEADING)
        return templates.TemplateResponse(
            request,
            "fragments/checks.html.j2",
            {"checks": results, "heading": PLATFORM_STATE_HEADING},
        )

    @app.post("/api/generate", response_class=HTMLResponse)
    async def generate(request: Request) -> HTMLResponse:
        payload = await request.json()
        try:
            blocks = bundle_init(
                GenerateRequest(
                    template=str(payload.get("template", "")),
                    project_name=str(payload.get("project_name") or "my-project"),
                ),
                request.app.state.config,
            )
        except GenerateError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ConfigError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return templates.TemplateResponse(
            request, "fragments/blocks.html.j2", {"blocks": blocks}
        )

    @app.get("/api/palette")
    async def palette(request: Request, q: str = "") -> dict:
        needle = q.strip().lower()
        hits = []
        for track in tracks_for(request):
            for step in track.steps:
                haystack = f"{track.title} {step.title} {step.body}".lower()
                if not needle or needle in haystack:
                    hits.append(
                        {
                            "track": track.id,
                            "track_title": track.title,
                            "step": step.id,
                            "title": step.title,
                        }
                    )
        return {"results": hits[:20]}

    return app


# The Apps runtime starts `uvicorn aai_console.server:app`, which needs an instance
# rather than a factory. Building it at import keeps a config failure loud and early.
app = create_app()
