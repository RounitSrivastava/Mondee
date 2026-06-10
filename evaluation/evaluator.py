"""
evaluation/evaluator.py

Advanced Evaluator for Hybrid Retrieval System

Metrics:
1. Score Distribution
2. Score Std Dev
3. Retrieval Confidence
4. Diversity
5. Persona Separation
6. Jaccard Similarity
7. Global Evaluation Summary
"""

import os
import json
import csv
import statistics

from itertools import combinations
from datetime import datetime

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

# =====================================================
# RUN ALL PERSONAS
# =====================================================

def run_all_personas(
    personas: list[dict],
    top_k: int = 30
) -> list[dict]:

    from core.recommender import recommend

    all_results = []

    for persona in personas:

        print(
            f"[evaluator] Running "
            f"{persona['persona']}"
        )

        result = recommend(
            persona,
            top_k=top_k
        )

        all_results.append(
            result
        )

    return all_results


# =====================================================
# SCORE DISTRIBUTION
# =====================================================

def score_distribution(
    result: dict
) -> dict:

    scores = [

        r["score"]

        for r in result.get(
            "recommendations",
            []
        )
    ]

    if not scores:

        return {}

    top5_avg = (
        sum(scores[1:6])
        /
        max(
            1,
            min(5, len(scores) - 1)
        )
    )

    confidence = (
        scores[0] - top5_avg
    )

    unique_ids = {

        r["exp_id"]

        for r in result[
            "recommendations"
        ]
    }

    diversity = (
        len(unique_ids)
        /
        len(scores)
    )

    return {

        "persona":
            result["persona"],

        "top_1_score":
            round(scores[0], 4),

        "top_10_avg":
            round(
                sum(scores[:10])
                /
                min(
                    10,
                    len(scores)
                ),
                4
            ),

        "top_30_avg":
            round(
                sum(scores)
                /
                len(scores),
                4
            ),

        "score_spread":
            round(
                scores[0]
                -
                scores[-1],
                4
            ),

        "std_dev":
            round(
                statistics.pstdev(
                    scores
                ),
                4
            ),

        "confidence":
            round(
                confidence,
                4
            ),

        "diversity":
            round(
                diversity,
                4
            ),

        "result_count":
            len(scores)
    }


# =====================================================
# JACCARD SIMILARITY
# =====================================================

def overlap_between(
    result_a: dict,
    result_b: dict,
    top_n: int = 10
) -> float:

    ids_a = {

        r["exp_id"]

        for r in result_a[
            "recommendations"
        ][:top_n]
    }

    ids_b = {

        r["exp_id"]

        for r in result_b[
            "recommendations"
        ][:top_n]
    }

    if not ids_a or not ids_b:

        return 0.0

    intersection = len(
        ids_a & ids_b
    )

    union = len(
        ids_a | ids_b
    )

    if union == 0:

        return 0.0

    return round(
        intersection / union,
        4
    )


# =====================================================
# OVERLAP MATRIX
# =====================================================

def overlap_matrix(
    all_results: list[dict],
    top_n: int = 10
) -> list[dict]:

    rows = []

    for a, b in combinations(
        all_results,
        2
    ):

        overlap = overlap_between(
            a,
            b,
            top_n
        )

        rows.append({

            "persona_a":
                a["persona"],

            "persona_b":
                b["persona"],

            "jaccard_similarity":
                overlap,

            "separation":
                round(
                    1 - overlap,
                    4
                )
        })

    return rows


# =====================================================
# GLOBAL SUMMARY
# =====================================================

def build_summary(
    dist_rows: list[dict],
    overlap_rows: list[dict]
) -> dict:

    if not dist_rows:

        return {}

    avg_top1 = sum(
        r["top_1_score"]
        for r in dist_rows
    ) / len(dist_rows)

    avg_top10 = sum(
        r["top_10_avg"]
        for r in dist_rows
    ) / len(dist_rows)

    avg_spread = sum(
        r["score_spread"]
        for r in dist_rows
    ) / len(dist_rows)

    avg_std = sum(
        r["std_dev"]
        for r in dist_rows
    ) / len(dist_rows)

    avg_confidence = sum(
        r["confidence"]
        for r in dist_rows
    ) / len(dist_rows)

    avg_diversity = sum(
        r["diversity"]
        for r in dist_rows
    ) / len(dist_rows)

    if overlap_rows:

        avg_similarity = sum(
            r["jaccard_similarity"]
            for r in overlap_rows
        ) / len(overlap_rows)

        avg_separation = sum(
            r["separation"]
            for r in overlap_rows
        ) / len(overlap_rows)

    else:

        avg_similarity = 0
        avg_separation = 0

    return {

        "persona_count":
            len(dist_rows),

        "avg_top1_score":
            round(avg_top1, 4),

        "avg_top10_score":
            round(avg_top10, 4),

        "avg_score_spread":
            round(avg_spread, 4),

        "avg_std_dev":
            round(avg_std, 4),

        "avg_confidence":
            round(avg_confidence, 4),

        "avg_diversity":
            round(avg_diversity, 4),

        "avg_similarity":
            round(avg_similarity, 4),

        "avg_separation":
            round(avg_separation, 4)
    }


# =====================================================
# SAVE RESULTS
# =====================================================

def save_results(
    all_results: list[dict],
    dist_rows: list[dict],
    overlap_rows: list[dict]
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    summary = build_summary(
        dist_rows,
        overlap_rows
    )

    # -----------------------------------------
    # FULL JSON
    # -----------------------------------------

    full_path = os.path.join(
        RESULTS_DIR,
        f"recommendations_{ts}.json"
    )

    with open(
        full_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"[evaluator] Results → "
        f"{full_path}"
    )

    # -----------------------------------------
    # DISTRIBUTION CSV
    # -----------------------------------------

    dist_path = os.path.join(
        RESULTS_DIR,
        f"score_distribution_{ts}.csv"
    )

    if dist_rows:

        with open(
            dist_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=dist_rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(
                dist_rows
            )

    # -----------------------------------------
    # OVERLAP CSV
    # -----------------------------------------

    overlap_path = os.path.join(
        RESULTS_DIR,
        f"overlap_matrix_{ts}.csv"
    )

    if overlap_rows:

        with open(
            overlap_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=overlap_rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(
                overlap_rows
            )

    # -----------------------------------------
    # SUMMARY JSON
    # -----------------------------------------

    summary_path = os.path.join(
        RESULTS_DIR,
        f"summary_{ts}.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"[evaluator] Summary → "
        f"{summary_path}"
    )

    return (
        full_path,
        dist_path,
        overlap_path,
        summary_path
    )