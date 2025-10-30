from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz, process
from .db import get_conn

# Lightweight candidate fetcher:
#  - Prefer same sourcetype first.
#  - Also allow cross-sourcetype fallbacks, but score them lower.
#  - You can add WHERE predicates to limit by simple LIKE heuristics for speed.
CANDIDATE_SQL = """
SELECT id, sourcetype, source_field, mapping_type, mapped_field_name, rationale, confidence, updated_at
FROM field_mapping
"""

def _score(
    target_sourcetype: str, target_field: str,
    row: Dict[str, Any]
) -> float:
    # 0..100 fuzz scores
    st_score = fuzz.token_set_ratio(target_sourcetype, row["sourcetype"])
    sf_score = fuzz.token_set_ratio(target_field, row["source_field"])
    # Heuristic weighting: field name is more important than sourcetype
    # Bonus if same sourcetype
    bonus = 10.0 if target_sourcetype == row["sourcetype"] else 0.0
    return 0.35 * st_score + 0.65 * sf_score + bonus

def find_top5_similar(
    sourcetype: str,
    field_name: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(CANDIDATE_SQL)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    # Score all candidates
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        s = _score(sourcetype, field_name, r)
        r["_similarity"] = s
        scored.append((s, r))

    # Sort and take top N
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:limit]
    return [r for _, r in top]

def format_llm_context(records: List[Dict[str, Any]]) -> str:
    """
    Compact context for few-shot style prompting.
    """
    lines = ["### Prior Mapping Hints (Top 5 Similar)"]
    for i, r in enumerate(records, 1):
        lines.append(
            f"{i}. sourcetype={r['sourcetype']} | src_field={r['source_field']} "
            f"→ [{r['mapping_type']}] {r['mapped_field_name']} "
            f"(conf={r['confidence']}, sim={r['_similarity']:.1f})"
        )
        if r.get("rationale"):
            lines.append(f"   rationale: {r['rationale']}")
    lines.append("### End Hints")
    return "\n".join(lines)
