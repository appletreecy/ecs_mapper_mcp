#!/usr/bin/env python3
import os
import sys
import json
import argparse
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, List
import logging

# SQLAlchemy
from sqlalchemy import text

# DB helpers (now from SQLAlchemy-based db.py)
from .db import SessionLocal

# FastAPI
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Static frontend mounting (moved out)
from .static_frontend import mount_vite_frontend

# (Optional) CORS for local dev
try:
    from fastapi.middleware.cors import CORSMiddleware
except Exception:
    CORSMiddleware = None  # type: ignore

# dotenv (optional)
try:
    from dotenv import load_dotenv
    load_dotenv(".env.llm")
except Exception:
    pass

# OpenAI client (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# MCP (stdio) client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an expert ECS 8.10.0 field mapper.
Rules:
- Prefer ECS fields when a clear semantic match exists.
- If no ECS field fits, map to a non-ECS name with the prefix 'custom_prefix_' and preserve the original semantic.
- Respect prior mapping conventions shown in 'Prior Mapping Hints'.
- Return strict JSON with keys: mapping_type, mapped_field_name, ecs_version, rationale, confidence (0..1).
"""

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # change to logging.DEBUG for more verbosity
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Prompt & MCP helpers
# -----------------------------------------------------------------------------
def build_user_prompt(llm_context: str, sourcetype: str, field: str, description: str) -> str:
    return f"""{llm_context}

Task:
sourcetype={sourcetype}
field={field}
Describe: {description}

Return JSON:
{{
  "mapping_type": "ecs" | "non-ecs",
  "mapped_field_name": "<ecs.field or custom_prefix_{field}>",
  "ecs_version": "8.10.0",
  "rationale": "...",
  "confidence": 0.00
}}"""

def _unwrap_call_tool_result(result) -> Dict[str, Any]:
    """
    Normalize MCP CallToolResult into a plain dict.
    Handles json/object content, text (with or without code fences), or direct dicts.
    """
    if isinstance(result, dict):
        return result

    content = getattr(result, "content", None)

    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        # Prefer json/object items
        for item in content:
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t in ("json", "object"):
                val = getattr(item, "value", None) or (isinstance(item, dict) and item.get("value"))
                if isinstance(val, dict):
                    return val

        # Fallback to text; strip code fences and parse
        for item in content:
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t == "text":
                txt = getattr(item, "text", None) or (isinstance(item, dict) and item.get("text")) or ""
                s = txt.strip()
                if s.startswith("```"):
                    s = s[3:]
                    if "\n" in s:
                        s = s.split("\n", 1)[1]
                    s = s.rstrip("`")
                try:
                    val = json.loads(s)
                    if isinstance(val, dict):
                        return val
                except Exception:
                    pass

    for attr in ("json", "model_dump"):
        f = getattr(result, attr, None)
        if callable(f):
            try:
                val = f()
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    j = json.loads(val)
                    if isinstance(j, dict):
                        return j
            except Exception:
                pass

    return {}

async def call_mcp_suggest(
    sourcetype: str,
    field: str,
    limit: int = 5,
    spawn_cmd: Optional[str] = None,
    spawn_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Spawns the MCP server and calls the suggest_mappings tool.
    """
    here = os.path.abspath(os.path.dirname(__file__))        # .../app
    project_root = os.path.abspath(os.path.join(here, "..")) # .../
    env = dict(os.environ)
    env["PYTHONPATH"] = project_root + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    if spawn_cmd is None:
        spawn_cmd = sys.executable
    if spawn_args is None:
        spawn_args = ["-m", "app.mcp_server"]

    logger.debug(f"[MCP] Launching: {spawn_cmd} {' '.join(spawn_args)}")
    logger.debug(f"[MCP] PYTHONPATH={env.get('PYTHONPATH')}")

    server_params = StdioServerParameters(command=spawn_cmd, args=spawn_args, env=env)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            logger.debug(f"[MCP] initialize -> {getattr(init, 'capabilities', None)}")

            tools_resp = await session.list_tools()
            tool_names = [t.name for t in getattr(tools_resp, "tools", [])]
            logger.debug(f"[MCP] tools -> {tool_names}")

            if "suggest_mappings" not in tool_names:
                raise RuntimeError(
                    f"MCP tool 'suggest_mappings' not found. Tools: {tool_names}. "
                    "Check app.mcp_server exports."
                )

            result = await session.call_tool(
                "suggest_mappings",
                {"args": {"sourcetype": sourcetype, "field": field, "limit": limit}}
            )
            logger.debug(f"[MCP] raw CallToolResult: {result}")
            unwrapped = _unwrap_call_tool_result(result)
            logger.debug(f"[MCP] unwrapped -> {unwrapped}")
            return unwrapped or {}

def ask_llm_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key_env: str = "OPENAI_API_KEY",
) -> Dict[str, Any]:
    """
    Calls OpenAI and enforces JSON parsing with a light fallback.
    """
    if OpenAI is None:
        return {"error": "openai python package not available"}
    api_key = os.getenv(api_key_env)
    if not api_key:
        return {"error": f"{api_key_env} not set in environment"}

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    text_resp = resp.choices[0].message.content if resp.choices else ""

    try:
        return json.loads(text_resp)
    except Exception:
        start = text_resp.find("{")
        end = text_resp.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text_resp[start:end+1])
            except Exception:
                pass
        return {"raw": text_resp, "error": "Model did not return valid JSON"}

def validate_mapping(payload: Dict[str, Any], field: str) -> Dict[str, Any]:
    """
    Minimal validator/normalizer for the LLM response.
    """
    out: Dict[str, Any] = dict(payload) if isinstance(payload, dict) else {"raw": payload}
    mt = out.get("mapping_type")
    name = out.get("mapped_field_name")
    conf = out.get("confidence")

    if mt not in {"ecs", "non-ecs"}:
        out["mapping_type"] = "non-ecs" if (isinstance(name, str) and name.startswith("custom_prefix_")) else "ecs"

    if not name or not isinstance(name, str):
        out["mapped_field_name"] = f"custom_prefix_{field}"

    if not out.get("ecs_version"):
        out["ecs_version"] = "8.10.0"

    try:
        c = float(conf)
        c = max(0.0, min(1.0, c))
        out["confidence"] = c
    except Exception:
        out["confidence"] = 0.50

    if not out.get("rationale"):
        out["rationale"] = "Auto-filled rationale; please review."

    return out

async def map_field(
    sourcetype: str,
    field: str,
    description: str,
    limit: int,
    model: str,
) -> Dict[str, Any]:
    mcp_result = await call_mcp_suggest(sourcetype, field, limit=limit)
    logger.info(f"[MAP] mcp_result -> {mcp_result!r}")

    llm_context = (mcp_result or {}).get("llm_context", "")
    hints = (mcp_result or {}).get("top", (mcp_result or {}).get("top5", []))

    user_prompt = build_user_prompt(llm_context, sourcetype, field, description)

    llm_raw = ask_llm_openai(model=model, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

    llm_decision = validate_mapping(llm_raw, field)

    return {
        "query": {"sourcetype": sourcetype, "field": field, "limit": limit, "model": model},
        "hints": hints,
        "llm_prompt": user_prompt,
        "llm_decision": llm_decision
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECS Mapper Agent using MCP tool for retrieval")
    p.add_argument("--sourcetype", required=False, default="pan_threat", help="e.g., pan_threat")
    p.add_argument("--field", required=False, default="department_address", help="e.g., department_address")
    p.add_argument("--description", default="postal address of the department initiating the event",
                   help="semantic description for the field")
    p.add_argument("--limit", type=int, default=5, help="how many similar hints to fetch")
    p.add_argument("--model", default=MODEL, help="LLM model name")
    return p.parse_args()

# =============================================================================
# SQLAlchemy-based DB helpers (replacing mysql-connector code)
# =============================================================================
def mapping_exists(sourcetype: str, source_field: str, mapped_field_name: str) -> bool:
    with SessionLocal() as session:
        row = session.execute(
            text(
                """
                SELECT 1 FROM field_mapping
                WHERE sourcetype = :s AND source_field = :f AND mapped_field_name = :m
                LIMIT 1
                """
            ),
            {"s": sourcetype, "f": source_field, "m": mapped_field_name},
        ).first()
        return row is not None

def upsert_mapping(
    sourcetype: str,
    source_field: str,
    mapping_type: str,
    mapped_field_name: str,
    rationale: Optional[str] = None,
    confidence: float = 0.80,
):
    with SessionLocal() as session:
        session.execute(
            text(
                """
                INSERT IGNORE INTO field_mapping (
                    sourcetype, source_field, mapping_type, mapped_field_name,
                    rationale, confidence
                )
                VALUES (:s, :f, :t, :m, :r, :c)
                """
            ),
            {
                "s": sourcetype,
                "f": source_field,
                "t": mapping_type,
                "m": mapped_field_name,
                "r": rationale,
                "c": confidence,
            },
        )
        session.commit()

# =============================================================================
# FastAPI service
# =============================================================================

class SourceFieldIn(BaseModel):
    sourcetype: str
    field: str
    description: str

class MappingUpdateIn(BaseModel):
    human_verified: Optional[bool] = None
    mapped_field_name: Optional[str] = None

app = FastAPI(title="ECS Mapper MCP Service")

# CORS in dev (optional)
if CORSMiddleware:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# GET /mappings  (list with search + pagination)
# -----------------------------------------------------------------------------
@app.get("/mappings")
def get_mappings(
    search: str = Query("", description="Optional search filter"),
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=200),
):
    """
    Returns paginated list of field_mapping records using SQLAlchemy.
    - Case-insensitive search across sourcetype/source_field/mapped_field_name
    - Casts confidence to CHAR to avoid Decimal JSON serialization issues
    - Uses updated_at as created_at for the UI if created_at isn't present
    - Includes human_verified (as bool)
    """
    try:
        offset = (page - 1) * pageSize

        params: Dict[str, Any] = {"limit": pageSize, "offset": offset}
        where = ""
        if search:
            where = """
                WHERE
                    LOWER(sourcetype) LIKE :q OR
                    LOWER(source_field) LIKE :q OR
                    LOWER(mapped_field_name) LIKE :q
            """
            params["q"] = f"%{search.lower()}%"

        data_sql = text(
            f"""
            SELECT
                id,
                sourcetype,
                source_field,
                mapped_field_name,
                mapping_type,
                rationale,
                COALESCE(CAST(confidence AS CHAR), NULL) AS confidence,
                COALESCE(updated_at, NOW()) AS created_at,
                COALESCE(human_verified, 0) AS human_verified
            FROM field_mapping
            {where}
            ORDER BY id DESC
            LIMIT :limit OFFSET :offset
            """
        )

        count_sql = text(
            f"""
            SELECT COUNT(*) AS total
            FROM field_mapping
            {where}
            """
        )

        with SessionLocal() as session:
            rows = session.execute(data_sql, params).mappings().all()
            total_row = session.execute(count_sql, params).mappings().first()
            total = int(total_row["total"]) if total_row else 0

        # Convert RowMapping -> dict and normalize
        items: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)  # <-- make it mutable
            ca = d.get("created_at")
            if isinstance(ca, datetime):
                d["created_at"] = ca.isoformat()
            # ensure boolean for frontend
            d["human_verified"] = bool(d.get("human_verified"))
            items.append(d)

        return {"items": items, "total": total}

    except Exception as e:
        import traceback
        logger.error("Error in GET /mappings:\n%s", "".join(traceback.format_exception(e)))
        raise HTTPException(status_code=500, detail="Internal error in /mappings")


# -----------------------------------------------------------------------------
# PATCH /mappings/{id}  (update human_verified and/or mapped_field_name)
# -----------------------------------------------------------------------------
@app.patch("/mappings/{mapping_id}")
def update_mapping(mapping_id: int, payload: MappingUpdateIn):
    sets: List[str] = []
    params: Dict[str, Any] = {"id": mapping_id}

    if payload.human_verified is not None:
        sets.append("human_verified = :hv")
        params["hv"] = 1 if payload.human_verified else 0

    if payload.mapped_field_name is not None:
        name = payload.mapped_field_name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="mapped_field_name cannot be empty")
        sets.append("mapped_field_name = :name")
        params["name"] = name

    if not sets:
        raise HTTPException(status_code=400, detail="No updatable fields provided")

    sets.append("updated_at = NOW()")

    try:
        with SessionLocal() as session:
            res = session.execute(
                text(f"UPDATE field_mapping SET {', '.join(sets)} WHERE id = :id"),
                params,
            )
            session.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Mapping not found")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error("Error in PATCH /mappings/{id}:\n%s", "".join(traceback.format_exception(e)))
        raise HTTPException(status_code=500, detail="Internal error in PATCH /mappings")

# =========================
# /map-batch (POST)
# =========================
class BatchInputItem(BaseModel):
    sourcetype: str
    field: str
    description: str

class MapBatchResponse(BaseModel):
    results: list

@app.post("/map-batch", response_model=MapBatchResponse)
async def map_batch(
    items: List[BatchInputItem],
    limit: int = Query(5, ge=1, le=20),
    model: str = Query(MODEL)
):
    """
    Accepts a list of {sourcetype, field, description} and returns LLM mapping
    decisions for each. Also writes to DB (INSERT IGNORE) and reports db_status.
    """
    if not items:
        raise HTTPException(status_code=400, detail="Payload must be a non-empty array")

    tasks = [
        map_field(
            sourcetype=i.sourcetype,
            field=i.field,
            description=i.description,
            limit=limit,
            model=model,
        )
        for i in items
    ]
    mapped = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, res in enumerate(mapped):
        item = items[i]
        if isinstance(res, Exception):
            results.append({
                "query": {"sourcetype": item.sourcetype, "field": item.field, "limit": limit, "model": model},
                "hints": [],
                "llm_prompt": "",
                "llm_decision": {
                    "mapping_type": "non-ecs",
                    "mapped_field_name": f"custom_prefix_{item.field}",
                    "ecs_version": "8.10.0",
                    "rationale": f"Error during mapping: {type(res).__name__}",
                    "confidence": 0.0,
                },
                "db_status": "error",
            })
            continue

        # Upsert to DB
        try:
            sourcetype = res["query"]["sourcetype"]
            field = res["query"]["field"]
            decision = res["llm_decision"]

            mapped_field_name = decision["mapped_field_name"]
            mapping_type = decision["mapping_type"]
            rationale = decision.get("rationale")
            confidence = float(decision.get("confidence", 0.5))

            if mapping_exists(sourcetype, field, mapped_field_name):
                db_status = "exists"
            else:
                upsert_mapping(
                    sourcetype, field, mapping_type, mapped_field_name, rationale, confidence
                )
                db_status = "inserted"

            res["db_status"] = db_status
        except Exception:
            res["db_status"] = "error"

        results.append(res)

    return {"results": results}

# =========================
# Mount static frontend (Vite) from separate module
# =========================
mount_vite_frontend(app)

# =============================================================================
# CLI entry (manual one-off)
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    source_fields = [{
        "sourcetype": args.sourcetype,
        "field": args.field,
        "description": args.description,
    }]

    for source_field in source_fields:
        print(source_field)
        result = asyncio.run(
            map_field(
                sourcetype=source_field["sourcetype"],
                field=source_field["field"],
                description=source_field["description"],
                limit=args.limit,
                model=args.model,
            )
        )
        sourcetype = result["query"]["sourcetype"]
        field = result["query"]["field"]
        mapped_field_name = result["llm_decision"]["mapped_field_name"]
        mapping_type = result["llm_decision"]["mapping_type"]
        rationale = result["llm_decision"]["rationale"]
        confidence = result["llm_decision"]["confidence"]

        if mapping_exists(sourcetype, field, mapped_field_name):
            print("The row exists in DB already")
        else:
            print(
                "write this record into DB:",
                sourcetype, field, mapped_field_name, mapping_type, rationale, confidence
            )
            upsert_mapping(sourcetype, field, mapping_type, mapped_field_name, rationale, confidence)
