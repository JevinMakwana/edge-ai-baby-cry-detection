import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_step(command):
    print("\n[RUN]", " ".join(str(c) for c in command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full baby-cry pipeline in sequence with configurable hyperparameters"
    )

    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable to use for running pipeline steps",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data"),
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("./artifacts"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--clip-seconds", type=float, default=7.0)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--lr-plateau-patience", type=int, default=4)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--augment-minority", action="store_true")
    parser.add_argument("--aug-noise-std", type=float, default=0.002)
    parser.add_argument("--aug-time-shift-max", type=float, default=0.02)
    parser.add_argument("--aug-gain-min", type=float, default=0.85)
    parser.add_argument("--aug-gain-max", type=float, default=1.15)
    parser.add_argument("--aug-time-mask-max", type=int, default=6)
    parser.add_argument("--aug-freq-mask-max", type=int, default=6)
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Optimize per-class thresholds after training and export",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="f1-macro",
        choices=["f1-weighted", "f1-macro", "balanced-accuracy", "f1-micro"],
        help="Metric to optimize when threshold tuning is enabled",
    )

    parser.add_argument("--quantize", action="store_true", help="Export int8 TFLite")
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip dataset audit and metadata generation",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Skip split generation and reuse existing splits",
    )
    parser.add_argument(
        "--skip-smoke-infer",
        action="store_true",
        help="Skip one-sample TFLite smoke inference",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    artifacts_dir = (root / args.artifacts_dir).resolve()
    model_dir = artifacts_dir / "model"
    splits_dir = artifacts_dir / "splits"

    py = args.python_exe

    if not args.skip_audit:
        run_step(
            [
                py,
                str(root / "01_dataset_audit.py"),
                "--dataset-root",
                str((root / args.dataset_root).resolve()),
                "--out-dir",
                str(artifacts_dir),
            ]
        )

    if not args.skip_splits:
        run_step(
            [
                py,
                str(root / "02_prepare_splits.py"),
                "--metadata-csv",
                str(artifacts_dir / "metadata.csv"),
                "--out-dir",
                str(splits_dir),
                "--seed",
                str(args.seed),
            ]
        )

    run_step(
        [
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
            str(args.aug_noise_std),
            "--aug-time-shift-max",
            str(args.aug_time_shift_max),
            "--aug-gain-min",
            str(args.aug_gain_min),
            "--aug-gain-max",
            str(args.aug_gain_max),
            "--aug-time-mask-max",
            str(args.aug_time_mask_max),
            "--aug-freq-mask-max",
            str(args.aug_freq_mask_max),
            "--seed",
            str(args.seed),
        ]
        + (["--balanced-sampling"] if args.balanced_sampling else [])
        + (["--augment-minority"] if args.augment_minority else [])
    )

    export_cmd = [py, str(root / "04_export_tflite.py"), "--model-dir", str(model_dir)]
    if args.quantize:
        export_cmd.append("--quantize")
    run_step(export_cmd)

    if args.optimize_thresholds:
        run_step(
            [
                py,
                str(root / "08_threshold_optimization.py"),
                "--model-dir",
                str(model_dir),
                "--metric",
                str(args.threshold_metric),
                "--output-file",
                str(model_dir / "threshold_config.json"),
            ]
        )

    if not args.skip_smoke_infer:
        test_csv = pd.read_csv(splits_dir / "test.csv")
        if len(test_csv) > 0:
            sample_path = str(test_csv.iloc[0]["path"])
            run_step(
                [
                    py,
                    str(root / "05_infer_clip.py"),
                    "--use-tflite",
                    "--model-dir",
                    str(model_dir),
                    "--clip",
                    sample_path,
                ]
            )

    print("\nPipeline completed successfully.")
    print(f"Best model: {model_dir / 'cry_reason_model.keras'}")
    print(f"Per-class F1: {model_dir / 'per_class_metrics.csv'}")
    print(f"TFLite model: {model_dir / 'cry_reason_model.tflite'}")


if __name__ == "__main__":
    main()
