from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path: str) -> None:
    """Generate a runtime vs ligand complexity plot from a CSV file.

    The CSV must contain the columns:
        - ligand_id
        - num_atoms
        - runtime_seconds

    The script plots num_atoms (x-axis) vs runtime_seconds (y-axis)
    using a line plot with markers, and saves the figure as
    `runtime_vs_complexity.png` in the same directory as this script.
    """

    t_start = time.time()

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"ligand_id", "num_atoms", "runtime_seconds"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV must contain columns {required_cols}, missing: {missing}"
        )

    # Ensure numeric types
    df["num_atoms"] = pd.to_numeric(df["num_atoms"], errors="coerce")
    df["runtime_seconds"] = pd.to_numeric(df["runtime_seconds"], errors="coerce")
    df = df.dropna(subset=["num_atoms", "runtime_seconds"])  # drop invalid rows

    # Sort by ligand complexity for a clean line plot
    df = df.sort_values("num_atoms")

    x = df["num_atoms"].tolist()
    y = df["runtime_seconds"].tolist()

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, y, marker="o", linestyle="-", color="#4e79a7")

    ax.set_xlabel("Ligand complexity (number of atoms)", fontsize=12)
    ax.set_ylabel("Total runtime (seconds)", fontsize=12)
    ax.set_title("Docking/Optimization Runtime vs Ligand Complexity", fontsize=14, pad=10)

    # Optionally annotate a few points (e.g., first and last) with ligand_id
    if not df.empty:
        first = df.iloc[0]
        last = df.iloc[-1]
        for row in (first, last):
            ax.annotate(
                str(row["ligand_id"]),
                xy=(row["num_atoms"], row["runtime_seconds"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / "runtime_vs_complexity.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    elapsed = time.time() - t_start
    print(f"Saved runtime vs complexity figure to: {out_path}")
    print(f"Script wall-clock time: {elapsed:.3f} seconds")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python runtime_vs_complexity.py path/to/runtime_data.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    main(sys.argv[1])
