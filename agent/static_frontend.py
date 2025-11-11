# backend/app/static_frontend.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

logger = logging.getLogger(__name__)

def mount_vite_frontend(
    app: FastAPI,
    dist_dir: Optional[Path] = None,
    *,
    base_path: str = "/",      # where to serve the SPA (usually "/")
    assets_path: str = "/assets",  # where hashed assets live
) -> None:
    """
    Mount a built Vite app (index.html + /assets/*) onto an existing FastAPI app.

    Args:
        app: your FastAPI app instance
        dist_dir: explicit path to the Vite dist directory. If None, we try the default
                  "../frontend/dist" relative to this file.
        base_path: URL path to serve the SPA root ("/" by default)
        assets_path: URL path for Vite assets ("/assets" by default)
    """
    # Resolve default dist path if not provided:
    if dist_dir is None:
        # this file: backend/app/static_frontend.py
        here = Path(__file__).resolve()
        dist_dir = (here.parent.parent / "frontend" / "dist").resolve()

    index_html = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"

    if not dist_dir.exists() or not index_html.exists():
        logger.warning("[STATIC] Frontend dist not found at %s. Run `npm run build` in frontend.", dist_dir)
        return

    # Mount /assets (fingerprinted js/css)
    if assets_dir.exists():
        app.mount(assets_path, StaticFiles(directory=str(assets_dir)), name="assets")
    else:
        logger.warning("[STATIC] assets dir not found at %s (continuing: index.html will still be served).", assets_dir)

    # Root index (exact match for base_path)
    # Note: include_in_schema=False so it doesn't appear in OpenAPI docs.
    @app.get(base_path, include_in_schema=False)
    async def vite_index_root():
        return FileResponse(str(index_html))

    # SPA fallback: any unknown path under base_path returns index.html (so client routing works)
    # Using a path param lets all nested routes resolve to the same index.html
    normalized = base_path.rstrip("/")
    if normalized == "":
        normalized = "/"

    @app.get(f"{normalized}" + "/{full_path:path}", include_in_schema=False)
    async def vite_spa_fallback(full_path: str):
        return FileResponse(str(index_html))

    logger.info("[STATIC] Vite frontend mounted: dist=%s base=%s assets=%s", dist_dir, base_path, assets_path)
