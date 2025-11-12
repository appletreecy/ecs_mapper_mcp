# app/similar_mapper.py
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz
from sqlalchemy import text

from .db import SessionLocal  # SQLAlchemy session

# Lightweight candidate fetcher:
#  - Prefer same sourcetype first.
#  - Also allow cross-sourcetype fallbacks, but score them lower.
CANDIDATE_SQL = """
SELECT
  id,
  sourcetype,
  source_field,
  mapping_type,
  mapped_field_name,
  rationale,
  confidence,
  updated_at
FROM field_mapping
"""

def _score(target_sourcetype: str, target_field: str, row: Dict[str, Any]) -> float:
    # 0..100 fuzz scores
    st_score = fuzz.token_set_ratio(target_sourcetype, row["sourcetype"])
    sf_score = fuzz.token_set_ratio(target_field, row["source_field"])
    # Heuristic weighting: field name is more important than sourcetype
    bonus = 10.0 if target_sourcetype == row["sourcetype"] else 0.0
    return 0.35 * st_score + 0.65 * sf_score + bonus

def find_top5_similar(
    sourcetype: str,
    field_name: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    # Fetch candidates via SQLAlchemy
    with SessionLocal() as session:
        rows = session.execute(text(CANDIDATE_SQL)).mappings().all()

    # Score all candidates
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        d = dict(r)  # RowMapping -> mutable dict
        s = _score(sourcetype, field_name, d)
        d["_similarity"] = s
        scored.append((s, d))

    # Sort and take top N
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:limit]
    return [r for _, r in top]

def format_llm_context(records: List[Dict[str, Any]]) -> str:
    """
    Compact context for few-shot style prompting.
    """
    lines = ["### Prior Mapping Hints (Top 5 Similar)"]
    for i, r in enumerate(records, 1):
        # Confidence may be Decimal/str/float; render neatly
        conf = r.get("confidence")
        try:
            conf_str = f"{float(conf):.2f}"
        except Exception:
            conf_str = str(conf) if conf is not None else "null"

        lines.append(
            f"{i}. sourcetype={r['sourcetype']} | src_field={r['source_field']} "
            f"→ [{r['mapping_type']}] {r['mapped_field_name']} "
            f"(conf={conf_str}, sim={r['_similarity']:.1f})"
        )
        if r.get("rationale"):
            lines.append(f"   rationale: {r['rationale']}")
    lines.append("### End Hints")
    return "\n".join(lines)
