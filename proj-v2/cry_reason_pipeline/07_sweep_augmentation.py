import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd):
    print("[RUN]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def macro_f1(per_class_csv: Path) -> float:
    df = pd.read_csv(per_class_csv)
    return float(df["f1"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep augmentation hyperparameters and pick best macro-F1")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--splits-dir", type=Path, default=Path("./artifacts/splits"))
    parser.add_argument("--sweep-root", type=Path, default=Path("./artifacts/sweeps"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--clip-seconds", type=float, default=7.0)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--lr-plateau-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = args.python_exe

    splits_dir = (root / args.splits_dir).resolve()
    sweep_root = (root / args.sweep_root).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    candidates = [
        {
            "name": "aug_a",
            "aug_noise_std": 0.002,
            "aug_time_shift_max": 0.02,
            "aug_gain_min": 0.85,
            "aug_gain_max": 1.15,
            "aug_time_mask_max": 6,
            "aug_freq_mask_max": 6,
        },
        {
            "name": "aug_b",
            "aug_noise_std": 0.003,
            "aug_time_shift_max": 0.03,
            "aug_gain_min": 0.8,
            "aug_gain_max": 1.2,
            "aug_time_mask_max": 8,
            "aug_freq_mask_max": 8,
        },
        {
            "name": "aug_c",
            "aug_noise_std": 0.0015,
            "aug_time_shift_max": 0.015,
            "aug_gain_min": 0.9,
            "aug_gain_max": 1.1,
            "aug_time_mask_max": 5,
            "aug_freq_mask_max": 5,
        },
        {
            "name": "aug_d",
            "aug_noise_std": 0.004,
            "aug_time_shift_max": 0.04,
            "aug_gain_min": 0.75,
            "aug_gain_max": 1.25,
            "aug_time_mask_max": 10,
            "aug_freq_mask_max": 10,
        },
    ]

    results = []

    for cfg in candidates:
        run_dir = sweep_root / cfg["name"]
        model_dir = run_dir / "model"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            py,
            str(root / "03_train_model.py"),
            "--splits-dir",
            str(splits_dir),
            "--out-dir",
            str(model_dir),
            "--sample-rate",
            str(args.sample_rate),
            "--clip-seconds",
            str(args.clip_seconds),
            "--n-mels",
            str(args.n_mels),
            "--n-fft",
            str(args.n_fft),
            "--hop-length",
            str(args.hop_length),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--lr-plateau-patience",
            str(args.lr_plateau_patience),
            "--aug-noise-std",
            str(cfg["aug_noise_std"]),
            "--aug-time-shift-max",
            str(cfg["aug_time_shift_max"]),
            "--aug-gain-min",
            str(cfg["aug_gain_min"]),
            "--aug-gain-max",
            str(cfg["aug_gain_max"]),
            "--aug-time-mask-max",
            str(cfg["aug_time_mask_max"]),
            "--aug-freq-mask-max",
            str(cfg["aug_freq_mask_max"]),
            "--seed",
            str(args.seed),
            "--balanced-sampling",
            "--augment-minority",
        ]
        run_cmd(train_cmd)

        export_cmd = [
            py,
            str(root / "04_export_tflite.py"),
            "--model-dir",
            str(model_dir),
            "--quantize",
        ]
        run_cmd(export_cmd)

        per_class_csv = model_dir / "per_class_metrics.csv"
        info_json = model_dir / "training_info.json"

        mf1 = macro_f1(per_class_csv)
        info = json.loads(info_json.read_text(encoding="utf-8"))
        results.append(
            {
                "name": cfg["name"],
                "macro_f1": mf1,
                "test_accuracy": float(info["test_accuracy"]),
                **cfg,
                "run_dir": str(run_dir),
            }
        )

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary_csv = sweep_root / "sweep_results.csv"
    results_df.to_csv(summary_csv, index=False)

    best = results_df.iloc[0].to_dict()
    best_dir = Path(str(best["run_dir"])) / "model"

    final_model_dir = (root / "artifacts/model").resolve()
    final_model_dir.mkdir(parents=True, exist_ok=True)

    for name in ["cry_reason_model.keras", "cry_reason_model.tflite", "training_info.json", "per_class_metrics.csv", "x_train_features.npy"]:
        src = best_dir / name
        dst = final_model_dir / name
        if src.exists():
            shutil.copy2(src, dst)

    best_json = sweep_root / "best_config.json"
    best_json.write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("Sweep complete")
    print(f"Summary: {summary_csv}")
    print(f"Best config: {best_json}")
    print(f"Best macro-F1: {best['macro_f1']:.4f}")
    print(f"Promoted model dir: {final_model_dir}")


if __name__ == "__main__":
    main()
