# Nicla Voice Firmware Scaffold

This folder is the board-side deployment area for Nicla Voice.

Use it after training/exporting the desktop model in `artifacts/model/`.

## Expected flow

1. Train and export the model from the desktop pipeline.
2. Copy `artifacts/model/cry_reason_model.tflite` into `3_openmv_firmware/model/`.
3. Add the board-specific inference logic in `main.py`.
4. Use `audio_preprocessor.py` for any reusable audio feature helpers.
5. Run `test_inference.py` as a local smoke test for the firmware-side logic.

## Files

- `main.py` - entry point for the Nicla Voice firmware.
- `audio_preprocessor.py` - audio preprocessing helpers.
- `test_inference.py` - simple smoke-test script.
- `model/` - place the exported `.tflite` model here.

## Notes

- This is a scaffold only.
- The actual Nicla Voice capture and inference code can be added next without changing the desktop training pipeline.