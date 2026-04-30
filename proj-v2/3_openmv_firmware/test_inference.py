"""
Test inference locally before deploying to Nicla Voice board
Verify model, preprocessing, and thresholds work correctly
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

# Import from this package
sys.path.insert(0, str(Path(__file__).parent))
from audio_preprocessor import preprocess_for_board


def load_model_config(model_dir):
    """Load model and configuration files"""
    model_dir = Path(model_dir)
    
    # Load TFLite model
    tflite_path = model_dir / "cry_reason_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Load training info
    with open(model_dir / "training_info.json") as f:
        training_info = json.load(f)
    
    # Load threshold config
    try:
        with open(model_dir / "threshold_config.json") as f:
            threshold_config = json.load(f)
    except FileNotFoundError:
        threshold_config = None
    
    return interpreter, training_info, threshold_config


def apply_thresholds(probs, threshold_config, index_to_label):
    """Apply per-class thresholds"""
    if threshold_config is None or threshold_config.get("type") != "per-class":
        pred_idx = int(np.argmax(probs))
        return pred_idx, float(probs[pred_idx])
    
    thresholds = threshold_config.get("thresholds", {})
    ordered_indices = np.argsort(probs)[::-1]
    
    for class_idx in ordered_indices:
        class_idx = int(class_idx)
        thresh = float(thresholds.get(str(class_idx), 0.5))
        if probs[class_idx] >= thresh:
            return class_idx, float(probs[class_idx])
    
    # Fallback
    pred_idx = int(ordered_indices[0])
    return pred_idx, float(probs[pred_idx])


def run_inference(wav_path, model_dir="./model"):
    """Run inference on a single audio file"""
    
    print("=" * 70)
    print("NICLA VOICE TEST INFERENCE")
    print("=" * 70)
    print()
    
    # Load model and config
    print(f"Loading model from: {model_dir}")
    interpreter, training_info, threshold_config = load_model_config(model_dir)
    
    normalization = training_info["normalization"]
    index_to_label = {int(k): v for k, v in training_info["index_to_label"].items()}
    feature_params = training_info["feature_params"]
    
    print(f"✓ Model loaded")
    print(f"✓ Training info loaded")
    if threshold_config:
        print(f"✓ Threshold config loaded")
    print()
    
    # Get interpreter details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Model Details:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    print()
    
    # Preprocess audio
    print(f"Preprocessing audio: {wav_path}")
    print(f"  Sample rate: {feature_params['sample_rate']} Hz")
    print(f"  Mel bins: {feature_params['n_mels']}")
    print(f"  FFT size: {feature_params['n_fft']}")
    
    mel_spec = preprocess_for_board(wav_path, feature_params, normalization)
    print(f"  Output shape: {mel_spec.shape}")
    print(f"  Value range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    print()
    
    # Run inference
    print("Running inference...")
    interpreter.set_tensor(input_details[0]["index"], mel_spec)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    
    print(f"  Raw output shape: {output.shape}")
    print(f"  Output sum (should be ~1.0): {output[0].sum():.4f}")
    print()
    
    # Apply thresholds
    print("Applying thresholds...")
    pred_idx, pred_prob = apply_thresholds(output[0], threshold_config, index_to_label)
    pred_label = index_to_label[pred_idx]
    
    print()
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"  Predicted Label: {pred_label.upper()}")
    print(f"  Confidence:      {pred_prob:.2%}")
    print()
    
    print("All Class Probabilities:")
    print("-" * 70)
    for idx in range(len(index_to_label)):
        label = index_to_label[idx]
        prob = output[0][idx]
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:12} │ {bar} │ {prob:.4f}")
    print("-" * 70)
    
    if threshold_config:
        print()
        print("Applied Thresholds (per-class):")
        thresholds = threshold_config.get("thresholds", {})
        for idx in range(len(index_to_label)):
            label = index_to_label[idx]
            thresh = float(thresholds.get(str(idx), 0.5))
            print(f"  {label:12}: {thresh:.4f}")
    
    print()
    print("=" * 70)
    
    return {
        "label": pred_label,
        "confidence": float(pred_prob),
        "all_probabilities": {index_to_label[i]: float(output[0][i]) for i in range(len(index_to_label))},
    }


def batch_inference(wav_dir, model_dir="./model", extension="*.wav"):
    """Run inference on all WAV files in a directory"""
    
    wav_dir = Path(wav_dir)
    results = []
    
    wav_files = sorted(wav_dir.glob(extension))
    print(f"Found {len(wav_files)} WAV files")
    print()
    
    for i, wav_path in enumerate(wav_files):
        print(f"\n[{i+1}/{len(wav_files)}] {wav_path.name}")
        
        try:
            result = run_inference(str(wav_path), model_dir)
            results.append({
                "file": str(wav_path),
                **result
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "file": str(wav_path),
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH INFERENCE SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if "error" not in r]
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        # Accuracy against filename label (if present)
        print("\nResults by file:")
        for r in successful:
            filename = Path(r["file"]).stem
            print(f"  {filename}: {r['label']} ({r['confidence']:.2%})")


def main():
    parser = argparse.ArgumentParser(
        description="Test Nicla Voice model inference locally"
    )
    parser.add_argument(
        "--clip",
        type=str,
        help="Path to WAV file for single inference",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory with multiple WAV files for batch inference",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./model",
        help="Directory containing model files",
    )
    
    args = parser.parse_args()
    
    if args.clip:
        run_inference(args.clip, args.model_dir)
    elif args.dir:
        batch_inference(args.dir, args.model_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
