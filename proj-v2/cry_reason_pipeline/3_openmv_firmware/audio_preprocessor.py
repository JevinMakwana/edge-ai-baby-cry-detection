"""Reusable audio preprocessing helpers for Nicla Voice.

The final board implementation can adapt these constants and helpers to the
actual microphone and TFLite runtime available on the device.
"""

SAMPLE_RATE = 8000
CLIP_SECONDS = 7.0
N_MELS = 40


def expected_clip_samples() -> int:
    return int(SAMPLE_RATE * CLIP_SECONDS)


def describe_preprocessor() -> str:
    return (
        f"sample_rate={SAMPLE_RATE}, clip_seconds={CLIP_SECONDS}, "
        f"n_mels={N_MELS}, clip_samples={expected_clip_samples()}"
    )