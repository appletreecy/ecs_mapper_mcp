#!/usr/bin/env python3
import os
import sys
import json
import argparse
import asyncio
from typing import Any, Dict, Optional, List  # ✅ ADDED
from .db import get_conn

# ✅ ADDED
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load LLM key if present
try:
    from dotenv import load_dotenv
    load_dotenv(".env.llm")
except Exception:
    pass

# OpenAI client (swap for your provider if desired)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # We'll check at runtime

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
    Prefer json/object content; fallback to parsing text as JSON.
    """
    if isinstance(result, dict):
        return result

    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            t = getattr(item, "type", None)
            if t in ("json", "object"):
                val = getattr(item, "value", None)
                if isinstance(val, dict):
                    return val
            if t == "text":
                txt = getattr(item, "text", "")
                try:
                    return json.loads(txt)
                except Exception:
                    pass
    return {}

async def call_mcp_suggest(
    sourcetype: str,
    field: str,
    limit: int = 5,
    spawn_cmd: Optional[str] = None,
    spawn_args: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Spawns the MCP server and calls the suggest_mappings tool.
    Works whether you run from project root or agent/ by fixing PYTHONPATH.
    """
    # Ensure app.* is importable for the spawned process
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = dict(os.environ)
    env["PYTHONPATH"] = project_root + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    if spawn_cmd is None:
        spawn_cmd = sys.executable  # use same venv interpreter
    if spawn_args is None:
        spawn_args = ["-m", "app.mcp_server"]

    server_params = StdioServerParameters(command=spawn_cmd, args=spawn_args, env=env)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # optional: ensure tool exists
            tools = await session.list_tools()
            names = [t.name for t in getattr(tools, "tools", [])]
            if "suggest_mappings" not in names:
                raise RuntimeError("MCP tool 'suggest_mappings' not found. Is app.mcp_server correct?")
            result = await session.call_tool(
                "suggest_mappings",
                {"sourcetype": sourcetype, "field": field, "limit": limit}
            )
            return _unwrap_call_tool_result(result)

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
    api_key = os.getenv("OPENAI_API_KEY")
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
    text = resp.choices[0].message.content if resp.choices else ""

    # Try strict JSON parse, then try extracting a JSON object
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        return {"raw": text, "error": "Model did not return valid JSON"}

def validate_mapping(payload: Dict[str, Any], field: str) -> Dict[str, Any]:
    """
    Minimal validator/normalizer for the LLM response.
    """
    out = dict(payload) if isinstance(payload, dict) else {"raw": payload}
    mt = out.get("mapping_type")
    name = out.get("mapped_field_name")
    conf = out.get("confidence")

    # mapping_type
    if mt not in {"ecs", "non-ecs"}:
        out["mapping_type"] = "non-ecs" if (isinstance(name, str) and name.startswith("custom_prefix_")) else "ecs"

    # mapped_field_name
    if not name or not isinstance(name, str):
        out["mapped_field_name"] = f"custom_prefix_{field}"

    # ecs_version
    if not out.get("ecs_version"):
        out["ecs_version"] = "8.10.0"

    # confidence
    try:
        c = float(conf)
        c = max(0.0, min(1.0, c))
        out["confidence"] = c
    except Exception:
        out["confidence"] = 0.50

    # rationale
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
    # 1) retrieve prior mapping hints via MCP
    mcp_result = await call_mcp_suggest(sourcetype, field, limit=limit)
    llm_context = mcp_result.get("llm_context", "")
    hints = mcp_result.get("top", mcp_result.get("top5", []))

    # 2) build user prompt
    user_prompt = build_user_prompt(llm_context, sourcetype, field, description)

    # 3) call LLM
    llm_raw = ask_llm_openai(model=model, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

    # 4) validate/normalize
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

def mapping_exists(sourcetype:str, source_field: str, mapped_field_name: str) -> bool:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM field_mapping
            WHERE sourcetype=%s AND source_field=%s AND mapped_field_name=%s
            LIMIT 1
            """,
            (sourcetype, source_field, mapped_field_name),
        )
        return cursor.fetchone() is not None
    finally:
        cursor.close()
        conn.close()

def upsert_mapping(
    sourcetype: str,
    source_field: str,
    mapping_type: str,
    mapped_field_name: str,
    rationale: str = None,
    confidence: float = 0.80,
):
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT IGNORE INTO field_mapping (
                sourcetype, source_field, mapping_type, mapped_field_name,
                rationale, confidence
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (sourcetype, source_field, mapping_type, mapped_field_name, rationale, confidence),
        )
    finally:
        cursor.close()
        conn.close()

# =====================================================================================
# ✅ ADDED: FastAPI service that accepts a batch of fields and performs mapping + upsert
# =====================================================================================

class SourceFieldIn(BaseModel):  # ✅ ADDED
    sourcetype: str
    field: str
    description: str

app = FastAPI(title="ECS Mapper MCP Service")  # ✅ ADDED

@app.get("/health")  # ✅ ADDED
async def health():
    return {"status": "ok"}

@app.post("/map-batch")  # ✅ ADDED
async def map_batch(items: List[SourceFieldIn], limit: int = 5, model: str = MODEL):
    """
    Accepts a list of {sourcetype, field, description}, performs mapping for each,
    and writes non-duplicate results into MySQL. Returns all results.
    """
    try:
        # Run the MCP+LLM mapping concurrently for speed
        tasks = [map_field(i.sourcetype, i.field, i.description, limit, model) for i in items]
        results = await asyncio.gather(*tasks)

        out = []
        for res in results:
            sourcetype = res["query"]["sourcetype"]
            source_field = res["query"]["field"]
            decision = res["llm_decision"]

            mapped_field_name = decision["mapped_field_name"]
            mapping_type = decision["mapping_type"]
            rationale = decision.get("rationale", "")
            confidence = float(decision.get("confidence", 0.5))

            existed = mapping_exists(sourcetype, source_field, mapped_field_name)
            if not existed:
                upsert_mapping(sourcetype, source_field, mapping_type, mapped_field_name, rationale, confidence)

            out.append({
                "query": res["query"],
                "hints": res["hints"],
                "llm_decision": decision,
                "db_status": "exists" if existed else "inserted"
            })

        return {"results": out}

    except Exception as e:
        # Surface the error text for quick debugging
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================================
# The old CLI entry stays; it won't run when serving via uvicorn.
# =====================================================================================

if __name__ == "__main__":
    # Old hardcoded demo loop kept for backward-compat (won’t run under uvicorn)
    source_fields = [{
        "sourcetype": "pan_traffic",
        "field": "department_phone_number",
        "description": "the phone number of this department"
    }, {
        "sourcetype": "pan_traffic",
        "field": "department_abn",
        "description": "the abn number of this department"
    }]

    for source_field in source_fields:
        print(source_field)
        result = asyncio.run(
            map_field(
                sourcetype=source_field["sourcetype"],
                field=source_field["field"],
                description=source_field["description"],
                limit=5,
                model="gpt-4.1-mini"
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
            print(f"write this record into DB: {sourcetype} {field} {mapped_field_name} {mapping_type} {rationale} {confidence}")
            upsert_mapping(sourcetype, field, mapping_type, mapped_field_name, rationale, confidence)
