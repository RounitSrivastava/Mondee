"""
core/recommender.py

Hybrid Recommender
RoBERTa + LDA + FAISS

Responsibilities:
- Persona -> semantic query
- Embed query
- Search FAISS
- Rank recommendations
- Return structured response
"""

from datetime import (
    datetime,
    timezone,
    timedelta
)

import faiss

from core.vectorizer import embed_query
from core.indexer import load_index

# =====================================================
# GLOBAL CACHE
# =====================================================

_index = None
_meta = None

TOP_K = 30

# =====================================================
# LOAD INDEX ONCE
# =====================================================

def _load_once():

    global _index
    global _meta

    if _index is None:

        _index, _meta = load_index()

        if hasattr(_index, "nprobe"):

            _index.nprobe = 40

        print(
            f"[recommender] Loaded "
            f"{_index.ntotal} vectors"
        )

# =====================================================
# PERSONA -> QUERY
# =====================================================

def _flatten_persona(
    persona: dict
) -> str:

    parts = []

    persona_name = (
        persona.get(
            "persona",
            ""
        )
        .replace("_", " ")
        .lower()
        .strip()
    )

    if persona_name:

        parts.append(persona_name)

    # signals

    for signal in persona.get(
        "signals",
        []
    ):

        clean = (
            signal
            .replace("_", " ")
            .lower()
            .strip()
        )

        parts.append(clean)

    # query string

    query_string = persona.get(
        "query_string",
        ""
    )

    if query_string:

        parts.append(
            query_string.lower()
        )

    # destination

    destination = persona.get(
        "destination",
        ""
    )

    if destination:

        parts.append(
            destination.lower()
        )

    # persona expansion

    expansions = {

        "budget shopper": [
            "cheap",
            "discount",
            "affordable",
            "deal"
        ],

        "premium buyer": [
            "luxury",
            "premium",
            "high quality"
        ],

        "sports enthusiast": [
            "fitness",
            "gym",
            "athletic",
            "performance"
        ],

        "health conscious": [
            "healthy",
            "organic",
            "nutrition",
            "wellness"
        ],

        "gift buyer": [
            "gift",
            "birthday",
            "present"
        ],

        "family buyer": [
            "family",
            "kids",
            "children",
            "household"
        ]
    }

    if persona_name in expansions:

        parts.extend(
            expansions[persona_name]
        )

    query = " ".join(parts)

    return query.strip()

# =====================================================
# MAIN RECOMMENDATION
# =====================================================

def recommend(
    persona: dict,
    top_k: int = TOP_K
) -> dict:

    _load_once()

    user_id = persona.get(
        "user_id",
        "unknown"
    )

    now = datetime.now(
        timezone.utc
    )

    refresh_due = (
        now + timedelta(hours=12)
    )

    # -------------------------------------------------
    # QUERY
    # -------------------------------------------------

    query_string = _flatten_persona(
        persona
    )

    if not query_string:

        raise ValueError(
            "Empty query generated."
        )

    print(
        f"[recommender] Query: "
        f"{query_string}"
    )

    # -------------------------------------------------
    # EMBED QUERY
    # -------------------------------------------------

    query_vector = embed_query(
        query_string
    ).astype("float32")

    faiss.normalize_L2(
        query_vector
    )

    # -------------------------------------------------
    # SEARCH
    # -------------------------------------------------

    search_k = min(
        top_k * 10,
        _index.ntotal
    )

    scores, indices = _index.search(
        query_vector,
        search_k
    )

    # -------------------------------------------------
    # BUILD RESULTS
    # -------------------------------------------------

    recommendations = []

    used_exp_ids = set()

    rank = 1

    for idx, score in zip(
        indices[0],
        scores[0]
    ):

        if idx == -1:

            continue

        if idx >= len(_meta):

            continue

        item = _meta[idx]

        exp_id = item.get(
            "exp_id"
        )

        if exp_id in used_exp_ids:

            continue

        used_exp_ids.add(
            exp_id
        )

        recommendations.append({

            "rank":
                rank,

            "exp_id":
                exp_id,

            "name":
                item.get(
                    "name",
                    ""
                ),

            "details":
                item.get(
                    "details",
                    ""
                )[:200],

            "score":
                round(
                    float(score) * 100,
                    2
                )
        })

        rank += 1

        if len(
            recommendations
        ) >= top_k:

            break

    # -------------------------------------------------
    # RESPONSE
    # -------------------------------------------------

    return {

        "user_id":
            user_id,

        "persona":
            persona.get(
                "persona",
                "unknown"
            ),

        "query_string_used":
            query_string,

        "generated_at":
            now.isoformat(),

        "refresh_due_at":
            refresh_due.isoformat(),

        "recommendations":
            recommendations
    }