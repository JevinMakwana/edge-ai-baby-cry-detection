import argparse
import json
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import tensorflow as tf


def preprocess_clip(path: Path, feature_params: dict, norm: dict) -> np.ndarray:
    sr = int(feature_params["sample_rate"])
    clip_seconds = float(feature_params["clip_seconds"])
    n_mels = int(feature_params["n_mels"])
    n_fft = int(feature_params["n_fft"])
    hop_length = int(feature_params["hop_length"])

    clip_len = int(sr * clip_seconds)
    wav, _ = librosa.load(str(path), sr=sr, mono=True)

    if len(wav) < clip_len:
        wav = np.pad(wav, (0, clip_len - len(wav)), mode="constant")
    else:
        wav = wav[:clip_len]

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    feat = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    feat = feat[..., np.newaxis]

    mean = float(norm["mean"])
    std = float(norm["std"]) + 1e-6
    feat = (feat - mean) / std

    return feat[np.newaxis, ...].astype(np.float32)


def predict_tflite(interpreter: tf.lite.Interpreter, x: np.ndarray) -> np.ndarray:
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    x_input = x
    if in_detail["dtype"] == np.int8:
        scale, zero_point = in_detail["quantization"]
        x_input = (x / scale + zero_point).round().astype(np.int8)

    interpreter.set_tensor(in_detail["index"], x_input)
    interpreter.invoke()
    out = interpreter.get_tensor(out_detail["index"])

    if out_detail["dtype"] == np.int8:
        scale, zero_point = out_detail["quantization"]
        out = scale * (out.astype(np.float32) - zero_point)

    return out.astype(np.float32)


def load_threshold_config(config_path: Path):
    if not config_path.exists():
        return None

    return json.loads(config_path.read_text(encoding="utf-8"))


def apply_thresholds(probs: np.ndarray, threshold_config: Optional[dict]) -> int:
    if threshold_config is None:
        return int(np.argmax(probs))

    if threshold_config.get("type") == "per-class":
        thresholds = {int(k): float(v) for k, v in threshold_config.get("thresholds", {}).items()}
        ordered_classes = np.argsort(probs)[::-1]

        for class_idx in ordered_classes:
            class_idx = int(class_idx)
            if probs[class_idx] >= thresholds.get(class_idx, 0.5):
                return class_idx

    return int(np.argmax(probs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-clip inference")
    parser.add_argument("--clip", type=Path, required=True, help="Path to WAV clip")
    parser.add_argument("--model-dir", type=Path, default=Path("./artifacts/model"))
    parser.add_argument("--use-tflite", action="store_true", help="Use TFLite model")
    parser.add_argument(
        "--threshold-config",
        type=Path,
        default=Path("./artifacts/model/threshold_config.json"),
        help="Optional threshold configuration JSON",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    info_path = model_dir / "training_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    threshold_config = load_threshold_config(args.threshold_config.resolve())

    x = preprocess_clip(args.clip.resolve(), info["feature_params"], info["normalization"])

    index_to_label = {int(k): v for k, v in info["index_to_label"].items()}

    if args.use_tflite:
        tflite_path = model_dir / "cry_reason_model.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        probs = predict_tflite(interpreter, x)[0]
    else:
        keras_path = model_dir / "cry_reason_model.keras"
        model = tf.keras.models.load_model(keras_path)
        probs = model.predict(x, verbose=0)[0]

    top_idx = apply_thresholds(np.asarray(probs, dtype=np.float32), threshold_config)

    print("Prediction:")
    print(f"label={index_to_label[top_idx]} prob={probs[top_idx]:.4f}")
    print("All class probabilities:")
    for idx, prob in enumerate(probs):
        print(f"{index_to_label[idx]}: {prob:.4f}")


if __name__ == "__main__":
    main()
