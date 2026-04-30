"""Nicla Voice firmware entry point scaffold.

Replace the placeholder logic with the board-specific capture and inference loop
once the exported TFLite model is copied into `model/`.
"""

from pathlib import Path


MODEL_DIR = Path("model")
MODEL_FILE = MODEL_DIR / "cry_reason_model.tflite"


def check_model_present() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Missing Nicla Voice model file: model/cry_reason_model.tflite. "
            "Copy the exported TFLite model here before flashing the board."
        )


def main() -> None:
    check_model_present()
    print("Nicla Voice firmware scaffold is ready.")
    print(f"Model file found at: {MODEL_FILE}")
    print("Add the board capture / preprocessing / inference loop here.")


if __name__ == "__main__":
    main()