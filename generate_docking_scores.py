from pathlib import Path
import csv


def main(out_path: str | None = None) -> None:
    """Generate an example docking_scores.csv file for benchmarking.

    The CSV will have two columns:
        - method
        - vina_score (kcal/mol)

    Default output path is ./docking_scores.csv in the same directory
    as this script.
    """
    if out_path is None:
        out_file = Path(__file__).resolve().parent / "docking_scores.csv"
    else:
        out_file = Path(out_path)

    rows = [
        {"method": "baseline_ligand", "vina_score": -7.8},
        {"method": "heuristic_optimization", "vina_score": -8.5},
        {"method": "ml_ligand_generation", "vina_score": -9.1},
        {"method": "manual_docking_control", "vina_score": -8.0},
    ]

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "vina_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote example docking scores CSV to: {out_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        print("Usage: python generate_docking_scores.py [optional_output_path]", file=sys.stderr)
        raise SystemExit(1)

    arg_path = sys.argv[1] if len(sys.argv) == 2 else None
    main(arg_path)
