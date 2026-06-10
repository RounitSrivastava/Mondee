"""
scripts/run_evaluation.py

Evaluation Runner

Pipeline:

1. Run all test personas
2. Generate score metrics
3. Generate persona separation matrix
4. Generate global summary
5. Save results

Outputs:

evaluation/results/
    recommendations_*.json
    score_distribution_*.csv
    overlap_matrix_*.csv
    summary_*.json
"""

import os
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

from personas.test_personas import (
    TEST_PERSONAS
)

from evaluation.evaluator import (
    run_all_personas,
    score_distribution,
    overlap_matrix,
    save_results,
    build_summary
)

# =====================================================
# MAIN
# =====================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)
    print("PIVOT RECOMMENDATION ENGINE — EVALUATION RUN")
    print("=" * 70)

    print(
        f"\nRunning "
        f"{len(TEST_PERSONAS)} personas..."
    )

    # -------------------------------------------------
    # Run Recommender
    # -------------------------------------------------

    all_results = run_all_personas(
        TEST_PERSONAS,
        top_k=50
    )

    print(
        f"\n✓ Generated "
        f"{len(all_results)} result sets"
    )

    # -------------------------------------------------
    # Score Metrics
    # -------------------------------------------------

    print(
        "\nComputing score distributions..."
    )

    dist_rows = [

        score_distribution(
            result
        )

        for result in all_results

    ]

    # -------------------------------------------------
    # Persona Separation
    # -------------------------------------------------

    print(
        "Computing persona separation..."
    )

    overlap_rows = overlap_matrix(

        all_results,

        top_n=10

    )

    # -------------------------------------------------
    # Global Summary
    # -------------------------------------------------

    summary = build_summary(

        dist_rows,

        overlap_rows

    )

    print(
        "\nGlobal Summary"
    )

    print(
        "-" * 40
    )

    for k, v in summary.items():

        print(
            f"{k:<25}: {v}"
        )

    # -------------------------------------------------
    # Save Results
    # -------------------------------------------------

    print(
        "\nSaving results..."
    )

    paths = save_results(

        all_results,

        dist_rows,

        overlap_rows

    )

    # -------------------------------------------------
    # Runtime
    # -------------------------------------------------

    elapsed = round(

        time.time() - start_time,

        2

    )

    print("\n" + "=" * 70)

    print(
        f"✓ Evaluation Complete "
        f"({elapsed}s)"
    )

    print("=" * 70)

    print("\nGenerated Files:\n")

    for path in paths:

        print(
            f"  • {path}"
        )

    print(
        "\nRun:\n"
        "python scripts/report.py"
        "\n\nto view the report."
    )


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":

    main()