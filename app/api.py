from fastapi import FastAPI, Query
from .similar_mapper import find_top5_similar, format_llm_context

app = FastAPI(title="ECS Mapper Helper")

@app.get("/suggest-mappings")
def suggest_mappings(
    sourcetype: str = Query(..., example="pan_threat"),
    field: str = Query(..., example="department_address")
):
    recs = find_top5_similar(sourcetype, field)
    return {
        "query": {"sourcetype": sourcetype, "field": field},
        "top5": recs,
        "llm_context": format_llm_context(recs)
    }
