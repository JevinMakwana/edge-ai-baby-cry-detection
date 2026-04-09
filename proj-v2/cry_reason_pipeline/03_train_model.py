import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def load_audio_fixed(path: str, sr: int, clip_seconds: float) -> np.ndarray:
    clip_len = int(sr * clip_seconds)
    wav, _ = librosa.load(path, sr=sr, mono=True)
    if len(wav) < clip_len:
        wav = np.pad(wav, (0, clip_len - len(wav)), mode="constant")
    else:
        wav = wav[:clip_len]
    return wav


def to_log_mel(wav: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def augment_waveform(
    wav: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    max_shift_samples: int,
    gain_min: float,
    gain_max: float,
) -> np.ndarray:
    out = wav.copy()

    if max_shift_samples > 0 and rng.random() < 0.5:
        shift = int(rng.integers(-max_shift_samples, max_shift_samples + 1))
        out = np.roll(out, shift)

    if noise_std > 0 and rng.random() < 0.6:
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    if gain_max > gain_min and rng.random() < 0.5:
        gain = float(rng.uniform(gain_min, gain_max))
        out = out * gain

    out = np.clip(out, -1.0, 1.0)
    return out


def augment_log_mel(
    feat: np.ndarray,
    rng: np.random.Generator,
    max_time_mask: int,
    max_freq_mask: int,
) -> np.ndarray:
    out = feat.copy()
    n_mels, n_frames = out.shape

    if max_time_mask > 0 and rng.random() < 0.6:
        width = int(rng.integers(0, min(max_time_mask, n_frames) + 1))
        if width > 0:
            start = int(rng.integers(0, n_frames - width + 1))
            out[:, start : start + width] = out.min()

    if max_freq_mask > 0 and rng.random() < 0.6:
        width = int(rng.integers(0, min(max_freq_mask, n_mels) + 1))
        if width > 0:
            start = int(rng.integers(0, n_mels - width + 1))
            out[start : start + width, :] = out.min()

    return out


def build_dataset(
    df: pd.DataFrame,
    label_to_index: dict,
    sr: int,
    clip_seconds: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    augment: bool = False,
    minority_ratio: float = 1.0,
    rng: np.random.Generator = None,
    noise_std: float = 0.002,
    max_shift_samples: int = 160,
    gain_min: float = 0.85,
    gain_max: float = 1.15,
    max_time_mask: int = 6,
    max_freq_mask: int = 6,
):
    if rng is None:
        rng = np.random.default_rng(42)

    features = []
    labels = []

    for _, row in df.iterrows():
        row_ratio = float(row.get("_minority_ratio", minority_ratio))
        wav = load_audio_fixed(row["path"], sr=sr, clip_seconds=clip_seconds)
        if augment and rng.random() < row_ratio:
            wav = augment_waveform(
                wav,
                rng=rng,
                noise_std=noise_std,
                max_shift_samples=max_shift_samples,
                gain_min=gain_min,
                gain_max=gain_max,
            )

        feat = to_log_mel(wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

        if augment and rng.random() < row_ratio:
            feat = augment_log_mel(
                feat,
                rng=rng,
                max_time_mask=max_time_mask,
                max_freq_mask=max_freq_mask,
            )

        features.append(feat)
        labels.append(label_to_index[row["label"]])

    x = np.stack(features, axis=0)
    x = x[..., np.newaxis]

    mean = x.mean()
    std = x.std() + 1e-6
    x = (x - mean) / std

    y = np.array(labels, dtype=np.int32)
    return x.astype(np.float32), y, float(mean), float(std)


def make_balanced_train_df(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = train_df["label"].value_counts()
    target = int(counts.max())
    rng = np.random.default_rng(seed)

    frames = []
    for label, group in train_df.groupby("label"):
        idx = rng.choice(group.index.to_numpy(), size=target, replace=True)
        sampled = train_df.loc[idx]
        sampled = sampled.assign(_source_count=int(counts[label]))
        frames.append(sampled)

    balanced = pd.concat(frames, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


def create_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1], dtype=y_pred.dtype)

        pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        ce_loss = -tf.math.log(pt)
        focal_weight = tf.pow(1.0 - pt, gamma)
        return tf.reduce_mean(alpha * focal_weight * ce_loss)
    
    return focal_loss


def create_model(input_shape, num_classes: int, use_focal_loss: bool = False) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    loss_fn = create_focal_loss() if use_focal_loss else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baby cry reason classifier")
    parser.add_argument("--splits-dir", type=Path, default=Path("./artifacts/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("./artifacts/model"))
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
    parser.add_argument("--use-focal-loss", action="store_true", help="Use focal loss for better minority handling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv((args.splits_dir / "train.csv").resolve())
    val_df = pd.read_csv((args.splits_dir / "val.csv").resolve())
    test_df = pd.read_csv((args.splits_dir / "test.csv").resolve())

    train_counts = train_df["label"].value_counts()

    labels_sorted = sorted(train_df["label"].unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(labels_sorted)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    train_df_used = train_df
    if args.balanced_sampling:
        train_df_used = make_balanced_train_df(train_df, seed=args.seed)

    max_count = int(train_counts.max())
    train_df_used = train_df_used.copy()
    train_df_used["_minority_ratio"] = train_df_used["label"].map(
        lambda x: min(1.0, max_count / float(train_counts[x] * 4.0))
    )

    rng = np.random.default_rng(args.seed)
    max_shift_samples = int(max(0.0, args.aug_time_shift_max) * args.sample_rate)

    x_train, y_train, mean, std = build_dataset(
        train_df_used,
        label_to_index,
        sr=args.sample_rate,
        clip_seconds=args.clip_seconds,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        augment=args.augment_minority,
        minority_ratio=1.0,
        rng=rng,
        noise_std=args.aug_noise_std,
        max_shift_samples=max_shift_samples,
        gain_min=args.aug_gain_min,
        gain_max=args.aug_gain_max,
        max_time_mask=args.aug_time_mask_max,
        max_freq_mask=args.aug_freq_mask_max,
    )
    x_val, y_val, _, _ = build_dataset(
        val_df,
        label_to_index,
        sr=args.sample_rate,
        clip_seconds=args.clip_seconds,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    x_test, y_test, _, _ = build_dataset(
        test_df,
        label_to_index,
        sr=args.sample_rate,
        clip_seconds=args.clip_seconds,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    classes = np.array(sorted(np.unique(y_train)))
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = {int(cls): float(wt) for cls, wt in zip(classes, class_weights_arr)}
    
    if args.use_focal_loss:
        class_weights = {int(cls): 1.0 for cls in classes}

    model = create_model(input_shape=x_train.shape[1:], num_classes=len(labels_sorted), use_focal_loss=args.use_focal_loss)

    tmp_best_model_path = out_dir / "_tmp_best_model.keras"
    if tmp_best_model_path.exists():
        tmp_best_model_path.unlink()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(tmp_best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.early_stop_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.lr_plateau_patience,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    if tmp_best_model_path.exists():
        model = tf.keras.models.load_model(tmp_best_model_path)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred_proba_test = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba_test, axis=1)
    
    # Also get validation predictions for threshold optimization
    y_pred_proba_val = model.predict(x_val, verbose=0)
    
    # Save predictions and probabilities for threshold optimization
    np.save(out_dir / "val_predictions.npy", y_pred_proba_val)
    np.save(out_dir / "val_labels.npy", y_val)
    np.save(out_dir / "test_predictions.npy", y_pred_proba_test)
    np.save(out_dir / "test_labels.npy", y_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        target_names=labels_sorted,
        zero_division=0,
    )
    conf = confusion_matrix(y_test, y_pred).tolist()

    keras_model_path = out_dir / "cry_reason_model.keras"
    if keras_model_path.exists():
        keras_model_path.unlink()
    model.save(keras_model_path)

    if tmp_best_model_path.exists():
        tmp_best_model_path.unlink()

    np.save(out_dir / "x_train_features.npy", x_train.astype(np.float32))

    training_info = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "label_to_index": label_to_index,
        "index_to_label": {str(k): v for k, v in index_to_label.items()},
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "normalization": {"mean": mean, "std": std},
        "feature_params": {
            "sample_rate": args.sample_rate,
            "clip_seconds": args.clip_seconds,
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
        },
        "classification_report": report,
        "confusion_matrix": conf,
        "history": {k: [float(vv) for vv in vals] for k, vals in history.history.items()},
    }

    with (out_dir / "training_info.json").open("w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)
    
    # Save metadata for threshold optimization
    metadata = {
        "labels_sorted": labels_sorted,
        "label_to_index": label_to_index,
        "index_to_label": {str(k): v for k, v in index_to_label.items()},
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    per_class_rows = []
    for label in labels_sorted:
        item = report.get(label, {})
        per_class_rows.append(
            {
                "label": label,
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1-score", 0.0)),
                "support": int(item.get("support", 0)),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class_metrics.csv", index=False)

    print(f"Saved model: {keras_model_path}")
    print(f"Saved per-class metrics: {out_dir / 'per_class_metrics.csv'}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
