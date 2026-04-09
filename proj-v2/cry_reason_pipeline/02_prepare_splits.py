import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test stratified splits")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("./artifacts/metadata.csv"),
        help="Metadata CSV from 01_dataset_audit.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./artifacts/splits"),
        help="Output directory for split CSV files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    metadata_csv = args.metadata_csv.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    if "label" not in df.columns:
        raise ValueError("metadata CSV must contain a 'label' column")

    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=args.seed,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1764705882,
        random_state=args.seed,
        stratify=train_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} -> {train_path}")
    print(f"Val:   {len(val_df)} -> {val_path}")
    print(f"Test:  {len(test_df)} -> {test_path}")
    print("Class balance in train split:")
    print(train_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
