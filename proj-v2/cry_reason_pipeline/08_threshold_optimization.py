#!/usr/bin/env python3
"""
Threshold Optimization for Imbalanced Classification.

Finds per-class probability thresholds on the validation set and writes them
to a JSON config that can be consumed by inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve


def compute_metrics(y_true, y_pred, metric="f1-weighted"):
    if metric == "f1-weighted":
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if metric == "f1-macro":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric == "balanced-accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1-micro":
        return f1_score(y_true, y_pred, average="micro", zero_division=0)
    raise ValueError(f"Unknown metric: {metric}")


def apply_threshold(y_proba, threshold=0.5, class_thresholds=None):
    if class_thresholds is None:
        return np.argmax(y_proba, axis=1)

    y_pred = np.zeros(len(y_proba), dtype=np.int32)
    for i, proba_sample in enumerate(y_proba):
        ordered_classes = np.argsort(proba_sample)[::-1]
        chosen_class = int(ordered_classes[0])

        for class_idx in ordered_classes:
            class_idx = int(class_idx)
            class_thresh = float(class_thresholds.get(class_idx, threshold))
            if proba_sample[class_idx] >= class_thresh:
                chosen_class = class_idx
                break

        y_pred[i] = chosen_class

    return y_pred


def find_per_class_thresholds(y_true, y_proba, n_classes, labels_sorted, metric="f1-weighted"):
    class_thresholds = {}

    for class_idx in range(n_classes):
        binary_true = (y_true == class_idx).astype(np.int32)
        precision, recall, thresholds = precision_recall_curve(binary_true, y_proba[:, class_idx])

        if len(thresholds) == 0:
            class_thresholds[class_idx] = 0.5
            continue

        precision = precision[:-1]
        recall = recall[:-1]
        f1_values = np.where(
            (precision + recall) > 0,
            (2.0 * precision * recall) / (precision + recall),
            0.0,
        )

        best_idx = int(np.argmax(f1_values))
        class_thresholds[class_idx] = float(thresholds[best_idx])

    y_pred_final = apply_threshold(y_proba, class_thresholds=class_thresholds)
    final_score = compute_metrics(y_true, y_pred_final, metric)
    return class_thresholds, final_score


def load_labels_sorted(model_dir: Path, data_dir: Path):
    candidates = [
        model_dir / "metadata.json",
        data_dir / "metadata.json",
        model_dir / "training_info.json",
        data_dir / "training_info.json",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        with candidate.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if "labels_sorted" in payload:
            return payload["labels_sorted"]

        if "index_to_label" in payload:
            index_to_label = payload["index_to_label"]
            return [index_to_label[str(idx)] for idx in sorted(map(int, index_to_label.keys()))]

    raise FileNotFoundError(
        "Could not find labels metadata in model-dir or data-dir. Expected metadata.json or training_info.json."
    )


def main():
    parser = argparse.ArgumentParser(description="Optimize thresholds for imbalanced classification")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="artifacts/model",
        help="Directory containing trained model artifacts",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="artifacts/data",
        help="Directory containing data metadata if not present in model-dir",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1-macro",
        choices=["f1-weighted", "f1-macro", "balanced-accuracy", "f1-micro"],
        help="Metric to optimize",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="artifacts/model/threshold_config.json",
        help="Output file for threshold configuration",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    labels_sorted = load_labels_sorted(model_dir, data_dir)
    n_classes = len(labels_sorted)

    val_predictions_path = model_dir / "val_predictions.npy"
    val_labels_path = model_dir / "val_labels.npy"
    if not val_predictions_path.exists() or not val_labels_path.exists():
        raise FileNotFoundError(
            "Validation predictions not found. Run training first so val_predictions.npy and val_labels.npy are saved in the model directory."
        )

    val_predictions = np.load(val_predictions_path)
    val_labels = np.load(val_labels_path)

    print(f"Loaded {len(val_labels)} validation samples with {n_classes} classes")
    print(f"Optimizing for: {args.metric}")
    print("\nFinding per-class optimal thresholds...")

    class_thresholds, best_score = find_per_class_thresholds(
        val_labels, val_predictions, n_classes, labels_sorted, args.metric
    )

    print("\nOptimal per-class thresholds:")
    for class_idx, thresh in sorted(class_thresholds.items()):
        print(f"  {labels_sorted[class_idx]}: {thresh:.3f}")

    print(f"\nBest {args.metric}: {best_score:.4f}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "per-class",
        "metric": args.metric,
        "thresholds": {str(k): float(v) for k, v in class_thresholds.items()},
        "score": float(best_score),
        "labels_sorted": labels_sorted,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nThreshold configuration saved to: {output_path}")


if __name__ == "__main__":
    main()
