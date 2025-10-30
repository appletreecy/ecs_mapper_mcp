import asyncio
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from .similar_mapper import find_top5_similar, format_llm_context

mcp = FastMCP("ecs-mapper-mcp")

class SuggestArgs(BaseModel):
    sourcetype: str = Field(..., description="Source type, e.g., pan_threat")
    field: str = Field(..., description="Field name to map, e.g., department_address")
    limit: int = Field(5, ge=1, le=10, description="How many hints to return (default 5)")

@mcp.tool()
def suggest_mappings(args: SuggestArgs) -> Dict[str, Any]:
    """
    Return top-N similar prior mappings from MySQL and a compact context string for LLM prompting.
    """
    recs: List[Dict[str, Any]] = find_top5_similar(args.sourcetype, args.field, args.limit)
    return {
        "query": {"sourcetype": args.sourcetype, "field": args.field, "limit": args.limit},
        "top": recs,
        "llm_context": format_llm_context(recs)
    }

if __name__ == "__main__":
    mcp.run()
