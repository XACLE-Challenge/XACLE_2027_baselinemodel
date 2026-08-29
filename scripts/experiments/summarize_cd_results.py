import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for variant in ("C", "D"):
        for split in ("validation", "test"):
            path = (
                args.experiment_dir
                / variant
                / "evaluation"
                / split
                / "evaluation_result.csv"
            )
            metrics = pd.read_csv(path).set_index("metric")["value"].to_dict()
            rows.append({"variant": variant, "split": split, **metrics})

    summary = pd.DataFrame(rows)[
        ["variant", "split", "SRCC", "LCC", "KTAU", "MSE", "N"]
    ]
    output_path = args.experiment_dir / "summary.csv"
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"Summary written to: {output_path}")


if __name__ == "__main__":
    main()
