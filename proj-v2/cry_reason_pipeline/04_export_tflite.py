import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def representative_dataset(features: np.ndarray, max_samples: int = 100):
    count = min(len(features), max_samples)
    for i in range(count):
        sample = features[i : i + 1].astype(np.float32)
        yield [sample]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained Keras model to TFLite")
    parser.add_argument("--model-dir", type=Path, default=Path("./artifacts/model"))
    parser.add_argument("--quantize", action="store_true", help="Enable full int8 quantization")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    keras_model_path = model_dir / "cry_reason_model.keras"
    tflite_path = model_dir / "cry_reason_model.tflite"

    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantize:
        features_path = model_dir / "x_train_features.npy"
        if not features_path.exists():
            raise FileNotFoundError(
                "x_train_features.npy is required for int8 quantization. Run training first."
            )

        features = np.load(features_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(features)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024.0
    print(f"Saved TFLite model: {tflite_path}")
    print(f"Model size: {size_kb:.2f} KB")


if __name__ == "__main__":
    main()
