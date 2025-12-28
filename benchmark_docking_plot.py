import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path: str) -> None:
    """Generate a bar plot comparing docking affinity scores across methods.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file with columns: `method` and `vina_score` (kcal/mol).

    Output
    ------
    Saves `benchmark_docking_comparison.png` in the project root directory.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"method", "vina_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {required_cols}, missing: {missing}")

    # Ensure consistent ordering for the four expected methods if present
    preferred_order = [
        "baseline_ligand",
        "heuristic_optimization",
        "ml_ligand_generation",
        "manual_docking_control",
    ]

    # If the CSV uses nicer display names, fall back to whatever order is present
    if set(preferred_order).issubset(set(df["method"].unique())):
        cat_type = pd.CategoricalDtype(preferred_order, ordered=True)
        df["method"] = df["method"].astype(cat_type)
        df = df.sort_values("method")

    methods = df["method"].astype(str).tolist()
    scores = df["vina_score"].astype(float).tolist()

    # Matplotlib figure
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(methods, scores, color="#4e79a7")

    # Labeling
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Docking affinity (kcal/mol)", fontsize=12)
    ax.set_title("Docking Affinity Comparison Across Methods", fontsize=14, pad=10)

    # Rotate x-labels for readability
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # Annotate each bar with its score
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f"{score:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Tight layout for clean saving
    fig.tight_layout()

    # Save to project root
    out_path = Path(__file__).resolve().parent / "benchmark_docking_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved docking comparison figure to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark_docking_plot.py path/to/docking_scores.csv", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
