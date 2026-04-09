import argparse
import contextlib
import json
import os
import statistics
import wave
from pathlib import Path

import pandas as pd

ALLOWED_CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]


def parse_filename_metadata(file_name: str) -> dict:
    stem = Path(file_name).stem
    parts = stem.split("-")

    gender = None
    age_code = None
    reason_code = None

    if len(parts) >= 3:
        gender = parts[-3]
        age_code = parts[-2]
        reason_code = parts[-1]

    return {
        "gender": gender,
        "age_code": age_code,
        "reason_code": reason_code,
    }


def wav_info(path: Path) -> dict:
    with contextlib.closing(wave.open(str(path), "rb")) as wav_reader:
        sample_rate = wav_reader.getframerate()
        num_frames = wav_reader.getnframes()
        channels = wav_reader.getnchannels()
        duration_sec = num_frames / float(sample_rate) if sample_rate else 0.0

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": duration_sec,
    }


def collect_rows(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for class_name in sorted(ALLOWED_CLASSES):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue

        for wav_path in class_dir.rglob("*.wav"):
            meta = parse_filename_metadata(wav_path.name)
            audio = wav_info(wav_path)
            rows.append(
                {
                    "path": str(wav_path.resolve()),
                    "label": class_name,
                    **meta,
                    **audio,
                }
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "num_samples": 0,
            "class_counts": {},
            "sample_rates": {},
            "channels": {},
            "duration": {},
        }

    duration_vals = df["duration_sec"].tolist()
    return {
        "num_samples": int(len(df)),
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "sample_rates": {str(k): int(v) for k, v in df["sample_rate"].value_counts().sort_index().items()},
        "channels": {str(k): int(v) for k, v in df["channels"].value_counts().sort_index().items()},
        "duration": {
            "min": round(float(min(duration_vals)), 4),
            "max": round(float(max(duration_vals)), 4),
            "mean": round(float(statistics.mean(duration_vals)), 4),
            "median": round(float(statistics.median(duration_vals)), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cleaned Donate-a-Cry WAV dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data"),
        help="Path to cleaned Donate-a-Cry dataset root",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./artifacts"),
        help="Directory to write audit files",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(dataset_root)
    if df.empty:
        raise RuntimeError(f"No WAV files found under: {dataset_root}")

    df = df.sort_values(["label", "path"]).reset_index(drop=True)

    metadata_csv = out_dir / "metadata.csv"
    summary_json = out_dir / "dataset_summary.json"

    df.to_csv(metadata_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summarize(df), f, indent=2)

    print(f"Saved metadata: {metadata_csv}")
    print(f"Saved summary:  {summary_json}")


if __name__ == "__main__":
    main()
