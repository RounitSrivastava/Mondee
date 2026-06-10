"""
scripts/report.py

Evaluation Report Viewer

Displays:

1. Top Recommendations
2. Score Distribution
3. Persona Separation Matrix
4. Global Summary
5. Diagnosis

Compatible with:
- E5 / BGE Embeddings
- LDA Hybrid Retrieval
- FAISS
"""

import os
import sys
import json
import glob

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "evaluation",
    "results"
)

# =====================================================
# COLOR SUPPORT
# =====================================================

try:

    from colorama import (
        Fore,
        Style,
        init
    )

    init(
        autoreset=True
    )

except ImportError:

    class Fore:

        GREEN = ""
        YELLOW = ""
        RED = ""
        CYAN = ""
        WHITE = ""
        MAGENTA = ""

    class Style:

        RESET_ALL = ""
        BRIGHT = ""

# =====================================================
# HELPERS
# =====================================================

def color_score(
    score: float
):

    if score >= 80:

        return (
            f"{Fore.GREEN}"
            f"{score:.2f}"
            f"{Style.RESET_ALL}"
        )

    elif score >= 60:

        return (
            f"{Fore.YELLOW}"
            f"{score:.2f}"
            f"{Style.RESET_ALL}"
        )

    return (
        f"{Fore.RED}"
        f"{score:.2f}"
        f"{Style.RESET_ALL}"
    )


def color_separation(
    sep: float
):

    if sep >= 0.70:

        return (
            f"{Fore.GREEN}"
            f"{sep:.3f}"
            f"{Style.RESET_ALL}"
        )

    elif sep >= 0.50:

        return (
            f"{Fore.YELLOW}"
            f"{sep:.3f}"
            f"{Style.RESET_ALL}"
        )

    return (
        f"{Fore.RED}"
        f"{sep:.3f}"
        f"{Style.RESET_ALL}"
    )


def latest_file(
    pattern: str
):

    files = sorted(

        glob.glob(
            os.path.join(
                RESULTS_DIR,
                pattern
            )
        )

    )

    return files[-1] if files else None

# =====================================================
# TOP RESULTS
# =====================================================

def print_top_results(
    all_results,
    top_n=5
):

    print("\n" + "=" * 80)
    print(
        f"TOP {top_n} RECOMMENDATIONS"
    )
    print("=" * 80)

    for result in all_results:

        persona = (

            result["persona"]
            .upper()
            .replace("_", " ")

        )

        print(
            f"\n{Fore.CYAN}"
            f"{Style.BRIGHT}"
            f"{persona}"
            f"{Style.RESET_ALL}"
        )

        query = result.get(
            "query_string_used",
            ""
        )

        print(
            f"Query: {query[:120]}"
        )

        print(
            f"\n{'Rank':<8}"
            f"{'Score':<12}"
            f"{'Experience'}"
        )

        print("-" * 70)

        for rec in result[
            "recommendations"
        ][:top_n]:

            print(

                f"{rec['rank']:<8}"

                f"{color_score(rec['score']):<12}"

                f"{rec['name'][:60]}"

            )

# =====================================================
# SCORE TABLE
# =====================================================

def print_score_distribution(
    path
):

    import csv

    print("\n" + "=" * 110)
    print(
        "SCORE DISTRIBUTION"
    )
    print("=" * 110)

    print(

        f"{'Persona':<22}"

        f"{'Top1':>10}"

        f"{'Top10':>10}"

        f"{'Spread':>10}"

        f"{'StdDev':>10}"

        f"{'Conf':>10}"

        f"{'Div':>10}"

    )

    print("-" * 110)

    with open(
        path,
        encoding="utf-8"
    ) as f:

        for row in csv.DictReader(f):

            print(

                f"{row['persona'][:20]:<22}"

                f"{color_score(float(row['top_1_score'])):>10}"

                f"{color_score(float(row['top_10_avg'])):>10}"

                f"{float(row['score_spread']):>10.2f}"

                f"{float(row['std_dev']):>10.2f}"

                f"{float(row['confidence']):>10.2f}"

                f"{float(row['diversity']):>10.2f}"

            )

# =====================================================
# OVERLAP
# =====================================================

def print_overlap_matrix(
    overlap_path
):

    import csv

    print("\n" + "=" * 90)
    print(
        "PERSONA SEPARATION"
    )
    print("=" * 90)

    print(

        f"{'Persona A':<25}"
        f"{'Persona B':<25}"
        f"{'Similarity':>12}"
        f"{'Separation':>15}"

    )

    print("-" * 90)

    with open(
        overlap_path,
        encoding="utf-8"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    rows = sorted(

        rows,

        key=lambda r:
        float(
            r["separation"]
        )

    )

    for row in rows:

        print(

            f"{row['persona_a'][:24]:<25}"

            f"{row['persona_b'][:24]:<25}"

            f"{float(row['jaccard_similarity']):>12.4f}"

            f"{color_separation(float(row['separation'])):>15}"

        )

# =====================================================
# SUMMARY
# =====================================================

def print_summary(
    summary_path
):

    print("\n" + "=" * 60)
    print(
        "GLOBAL MODEL SUMMARY"
    )
    print("=" * 60)

    with open(
        summary_path,
        encoding="utf-8"
    ) as f:

        summary = json.load(f)

    for key, value in summary.items():

        print(
            f"{key:<25}: {value}"
        )

# =====================================================
# DIAGNOSIS
# =====================================================

def print_diagnosis(
    summary_path
):

    with open(
        summary_path,
        encoding="utf-8"
    ) as f:

        summary = json.load(f)

    avg_spread = summary.get(
        "avg_score_spread",
        0
    )

    avg_sep = summary.get(
        "avg_separation",
        0
    )

    avg_conf = summary.get(
        "avg_confidence",
        0
    )

    print("\n" + "=" * 60)
    print(
        "DIAGNOSIS"
    )
    print("=" * 60)

    print(
        f"\nAverage Spread      : {avg_spread:.2f}"
    )

    print(
        f"Average Separation : {avg_sep:.3f}"
    )

    print(
        f"Average Confidence : {avg_conf:.2f}"
    )

    issues = []

    if avg_spread < 3:

        issues.append(
            "Low score spread"
        )

    if avg_sep < 0.50:

        issues.append(
            "Low persona separation"
        )

    if avg_conf < 1.5:

        issues.append(
            "Weak recommendation confidence"
        )

    if (
    avg_sep >= 0.70
    and avg_conf >= 1.5
):

        print(
            f"\n{Fore.GREEN}"
            f"✓ Retrieval quality looks strong."
            f"{Style.RESET_ALL}"
        )

    else:

        for issue in issues:

            print(
                f"\n{Fore.RED}"
                f"⚠ {issue}"
                f"{Style.RESET_ALL}"
            )

# =====================================================
# MAIN
# =====================================================

def main():

    rec_path = latest_file(
        "recommendations_*.json"
    )

    dist_path = latest_file(
        "score_distribution_*.csv"
    )

    overlap_path = latest_file(
        "overlap_matrix_*.csv"
    )

    summary_path = latest_file(
        "summary_*.json"
    )

    if not all([

        rec_path,
        dist_path,
        overlap_path,
        summary_path

    ]):

        print(
            "Run evaluation first."
        )

        sys.exit(1)

    with open(
        rec_path,
        encoding="utf-8"
    ) as f:

        results = json.load(f)

    print_top_results(
        results
    )

    print_score_distribution(
        dist_path
    )

    print_overlap_matrix(
        overlap_path
    )

    print_summary(
        summary_path
    )

    print_diagnosis(
        summary_path
    )


if __name__ == "__main__":

    main()