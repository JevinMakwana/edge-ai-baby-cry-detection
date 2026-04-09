#!/usr/bin/env python3
"""
Inference script with optional threshold adjustment for better minority class handling.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def load_threshold_config(config_file):
    """Load threshold configuration from JSON file."""
    if not Path(config_file).exists():
        return None
    
    with open(config_file) as f:
        config = json.load(f)
    return config


def apply_thresholds(y_proba, config):
    """Apply threshold configuration to predictions.
    
    Args:
        y_proba: Predicted probabilities (n_samples, n_classes)
        config: Threshold configuration dict
    
    Returns:
        y_pred: Predicted class indices
    """
    if config is None:
        return np.argmax(y_proba, axis=1)
    
    if config.get("type") == "global":
        return np.argmax(y_proba, axis=1)
    
    elif config.get("type") == "per-class":
        thresholds = {int(k): v for k, v in config["thresholds"].items()}
        y_pred = np.zeros(len(y_proba), dtype=np.int32)
        
        for i, proba_sample in enumerate(y_proba):
            ordered_classes = np.argsort(proba_sample)[::-1]
            chosen_class = int(ordered_classes[0])

            for class_idx in ordered_classes:
                class_idx = int(class_idx)
                if proba_sample[class_idx] >= thresholds.get(class_idx, 0.5):
                    chosen_class = class_idx
                    break

            y_pred[i] = chosen_class
        
        return y_pred
    
    return np.argmax(y_proba, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Inference with threshold adjustment")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/model/best_model.h5",
        help="Path to trained model",
    )
    parser.add_argument(
        "--threshold-config",
        type=str,
        default="artifacts/model/threshold_config.json",
        help="Path to threshold configuration JSON",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="artifacts/model/metadata.json",
        help="Path to metadata JSON with labels",
    )
    parser.add_argument(
        "--test-predictions",
        type=str,
        default="artifacts/model/test_predictions.npy",
        help="Path to test predictions (for evaluation demo)",
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        default="artifacts/model/test_labels.npy",
        help="Path to test labels (for evaluation demo)",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata_file)
    if not metadata_path.exists():
        fallback_path = Path(args.model_path).resolve().parent / "training_info.json"
        if fallback_path.exists():
            metadata_path = fallback_path

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "labels_sorted" in metadata:
        labels_sorted = metadata["labels_sorted"]
    else:
        index_to_label = metadata.get("index_to_label", {})
        labels_sorted = [index_to_label[str(idx)] for idx in sorted(map(int, index_to_label.keys()))]

    n_classes = len(labels_sorted)

    # Load threshold config
    threshold_config = load_threshold_config(args.threshold_config)
    if threshold_config:
        print(f"Loaded threshold configuration from: {args.threshold_config}")
        print(f"  Type: {threshold_config['type']}")
        print(f"  Metric: {threshold_config['metric']}")
        if threshold_config["type"] == "global":
            print(f"  Global threshold: {threshold_config['threshold']:.3f}")
        else:
            print(f"  Per-class thresholds:")
            for class_idx, thresh in sorted(threshold_config["thresholds"].items()):
                print(f"    {labels_sorted[int(class_idx)]}: {thresh:.3f}")
    else:
        print("No threshold configuration found, using default (argmax)")

    # Demo: Load test predictions and evaluate
    if Path(args.test_predictions).exists():
        print("\n" + "=" * 60)
        print("EVALUATION ON TEST SET")
        print("=" * 60)

        test_predictions = np.load(args.test_predictions)
        test_labels = np.load(args.test_labels)

        # Get predictions with thresholds
        y_pred = apply_thresholds(test_predictions, threshold_config)

        # Accuracy
        accuracy = np.mean(y_pred == test_labels)
        print(f"\nAccuracy: {accuracy:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                test_labels,
                y_pred,
                target_names=labels_sorted,
                zero_division=0,
            )
        )

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, y_pred)
        print(cm)


if __name__ == "__main__":
    main()
