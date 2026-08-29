import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"wav_file_name", "text", "average_score"}
SPLITS = ("train", "validation", "test")
VARIANTS = ("C", "D")


def validate(metadata_dir: Path, wav_dir: Path) -> None:
    errors = []
    frames = {}

    for variant in VARIANTS:
        for split in SPLITS:
            csv_path = metadata_dir / f"{split}_{variant}.csv"
            if not csv_path.is_file():
                errors.append(f"Missing CSV: {csv_path}")
                continue

            frame = pd.read_csv(csv_path)
            frames[(variant, split)] = frame
            missing_columns = REQUIRED_COLUMNS - set(frame.columns)
            if missing_columns:
                errors.append(
                    f"{csv_path}: missing columns {sorted(missing_columns)}"
                )
                continue
            if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
                errors.append(f"{csv_path}: required columns contain missing values")
            if frame["wav_file_name"].duplicated().any():
                errors.append(f"{csv_path}: duplicate wav_file_name values")
            scores = pd.to_numeric(frame["average_score"], errors="coerce")
            if scores.isnull().any() or not scores.between(0.0, 10.0).all():
                errors.append(f"{csv_path}: scores must be numeric values in [0, 10]")

            missing_audio = [
                name
                for name in frame["wav_file_name"]
                if not (wav_dir / split / str(name).lstrip("/")).is_file()
            ]
            if missing_audio:
                errors.append(
                    f"{csv_path}: {len(missing_audio)} missing audio files; "
                    f"first={missing_audio[0]}"
                )

            print(
                f"{variant}/{split}: N={len(frame)}, "
                f"score_mean={scores.mean():.4f}, "
                f"score_range=[{scores.min():.4f}, {scores.max():.4f}]"
            )

    for variant in VARIANTS:
        available = [split for split in SPLITS if (variant, split) in frames]
        for index, left in enumerate(available):
            left_names = set(frames[(variant, left)]["wav_file_name"])
            for right in available[index + 1 :]:
                overlap = left_names & set(frames[(variant, right)]["wav_file_name"])
                if overlap:
                    errors.append(
                        f"{variant}: {left}/{right} overlap contains "
                        f"{len(overlap)} files; first={sorted(overlap)[0]}"
                    )

    if errors:
        raise SystemExit("Metadata validation failed:\n- " + "\n- ".join(errors))
    print("C/D metadata validation: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        default="datasets/XACLE_dataset/meta_data/experiments",
        type=Path,
    )
    parser.add_argument(
        "--wav-dir", default="datasets/XACLE_dataset/wav", type=Path
    )
    args = parser.parse_args()
    validate(args.metadata_dir, args.wav_dir)


if __name__ == "__main__":
    main()
