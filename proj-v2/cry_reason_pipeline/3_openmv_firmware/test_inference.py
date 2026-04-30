"""Smoke test for the Nicla Voice firmware scaffold."""

from pathlib import Path

from main import MODEL_FILE, check_model_present


def main() -> None:
    print("Testing Nicla Voice firmware scaffold")
    print(f"Firmware folder: {Path(__file__).resolve().parent}")
    check_model_present()
    print(f"Model present: {MODEL_FILE}")
    print("Board-specific inference code can be added next.")


if __name__ == "__main__":
    main()