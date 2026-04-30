# Nicla Voice Deployment Guide

Complete step-by-step guide to deploy the cry reason classifier to Nicla Voice.

## Quick Start (5 minutes)

```bash
# 1. Copy model files to firmware folder (already done)
cd proj-v2/cry_reason_pipeline
copy artifacts\model\cry_reason_model.tflite ..\3_openmv_firmware\model\
copy artifacts\model\training_info.json ..\3_openmv_firmware\model\
copy artifacts\model\threshold_config.json ..\3_openmv_firmware\model\

# 2. Test locally (optional but recommended)
cd ../3_openmv_firmware
python test_inference.py --clip path/to/test_cry.wav

# 3. Connect Nicla Voice and deploy via OpenMV IDE
# (See "Deploy to Board" section below)
```

---

## Detailed Deployment Steps

### Prerequisites

1. **OpenMV IDE** (free)
   - Download: https://openmv.io/ide/
   - Install and launch

2. **Nicla Voice Board**
   - https://www.arduino.cc/pro/hardware/product/nicla-voice
   - Comes with pre-loaded bootloader

3. **Micro-USB Cable**
   - For connection and power

### Step 1: Connect Board

1. Plug Nicla Voice into computer via USB
2. LED should light up (red or green)
3. Wait 3 seconds for device detection

### Step 2: Verify Board Detection in OpenMV IDE

1. In OpenMV IDE, click **Tools** → **Board Info**
2. Should show: `Board: Nicla Voice` and `MCU: STMXXXXX`
3. If not detected: Check USB cable, try **Tools** → **Reset OpenMV Cam**

### Step 3: Upload Model Files

**Method A: Via File System (Recommended)**

1. Click **Tools** → **File System** → **New File System Image**
2. This creates a new file system on the board
3. Drag-and-drop files into the window:
   ```
   3_openmv_firmware/
   ├── main.py
   ├── audio_preprocessor.py
   ├── led_output.py
   └── model/
       ├── cry_reason_model.tflite
       ├── training_info.json
       └── threshold_config.json
   ```
4. Click **Eject** to save (wait for completion)

**Method B: Via Shell**

1. Click **Tools** → **OpenMV Shell**
2. Execute:
   ```python
   import os
   os.mkdir('/sd')  # Create SD card root
   os.mkdir('/sd/model')
   
   # Then transfer files using your file manager
   ```

### Step 4: Verify Files on Board

1. Click **Tools** → **File System** → **Browse**
2. Verify these files exist:
   - `/sd/main.py`
   - `/sd/audio_preprocessor.py`
   - `/sd/led_output.py`
   - `/sd/model/cry_reason_model.tflite` (15.7 KB)
   - `/sd/model/training_info.json`
   - `/sd/model/threshold_config.json`

### Step 5: Run Firmware

1. In OpenMV IDE, open `/sd/main.py`
2. Click **Run** (▶ button) or press **Ctrl+R**
3. Script should start executing

### Step 6: Monitor Output

1. Click **Tools** → **Serial Monitor** (or **Ctrl+Shift+C**)
2. Set baud rate to **9600**
3. You should see:
   ```
   ============================================================
   Baby Cry Reason Classifier - Nicla Voice
   ============================================================
   Labels: ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
   Normalization: mean=-45.34, std=15.92
   
   Starting audio capture... (Listening for cries)
   ============================================================
   ```

---

## Testing Before Deployment (Recommended)

### Test 1: Local Inference

Run on your PC to verify model + preprocessing:

```bash
cd proj-v2/3_openmv_firmware

# Test with one audio file
python test_inference.py --clip ../cry_reason_pipeline/test_data/hungry_sample.wav
```

Expected output:
```
==============================================================================
NICLA VOICE TEST INFERENCE
==============================================================================

Loading model from: ./model
✓ Model loaded
✓ Training info loaded
✓ Threshold config loaded

Model Details:
  Input shape: (1, 40, 67, 1)
  Output shape: (1, 5)
  ...

Preprocessing audio: ../cry_reason_pipeline/test_data/hungry_sample.wav
  ...

PREDICTION RESULT
==============================================================================
  Predicted Label: HUNGRY
  Confidence:      92.15%

All Class Probabilities:
────────────────────────────────────────────────────────────────────────────
  belly_pain   │ ██████░░░░░░░░░░░░░░░░░░░░░░ │ 0.1234
  burping      │ ███░░░░░░░░░░░░░░░░░░░░░░░░░ │ 0.0567
  discomfort   │ ████░░░░░░░░░░░░░░░░░░░░░░░░ │ 0.0712
  hungry       │ ██████████████████████████░░░ │ 0.9215
  tired        │ ██░░░░░░░░░░░░░░░░░░░░░░░░░░ │ 0.0272
```

### Test 2: Batch Testing

Test on multiple audio files:

```bash
# Create a test folder with labeled audio files
# (e.g., test_data/hungry_*.wav, test_data/burping_*.wav, etc.)

python test_inference.py --dir ../cry_reason_pipeline/test_data
```

### Test 3: Board Serial Output

Once running on board, open serial monitor and observe:

```
[Frame 0] Predicted: hungry (0.9215)
  belly_pain: 0.1234
  burping: 0.0567
  discomfort: 0.0712
  hungry: 0.9215
  tired: 0.0272

[Frame 1] Predicted: burping (0.8123)
  belly_pain: 0.0567
  burping: 0.8123
  ...
```

---

## Troubleshooting

### Issue: Board Not Detected

**Solution:**
1. Check USB cable (try different port)
2. In OpenMV IDE: **Tools** → **Reset OpenMV Cam**
3. Unplug and replug board
4. Check if board appears in Device Manager (Windows):
   - Should show as "STM32xxx USB Device"
   - If showing as "Unknown Device", download STM drivers

### Issue: FileNotFoundError for Model Files

**Solution:**
1. Verify files are in `/sd/model/` (check via File System browser)
2. Update paths in `main.py` if needed:
   ```python
   MODEL_PATH = "/sd/model/cry_reason_model.tflite"
   ```
3. Copy files again if missing

### Issue: Serial Monitor Shows Gibberish

**Solution:**
1. Check baud rate is **9600**
2. In OpenMV IDE: **Tools** → **Reset OpenMV Cam**
3. Restart serial monitor

### Issue: Model Inference Errors

**Solution:**
1. Verify model file is not corrupted (should be 15.7 KB)
2. Check TFLite version compatibility
3. Run `test_inference.py` locally first to isolate issue
4. Try reinstalling board firmware via OpenMV IDE

### Issue: LED Not Lighting Up

**Solution:**
1. Verify LED is physically connected to pin D10 (or update pin in `led_output.py`)
2. Add debug print to check color commands:
   ```python
   # In main.py
   print(f"Setting LED to {color}")
   ```
3. Try turning off LED manually:
   ```python
   led_controller.turn_off()
   ```

### Issue: Poor Prediction Accuracy

**Solution:**
1. Test with sample cries from original dataset
2. Ensure microphone is not blocked
3. Verify audio is 8 kHz (check `training_info.json`)
4. Check normalization parameters are applied:
   ```
   mean=-45.34, std=15.92
   ```

---

## File Structure Reference

```
proj-v2/
├── cry_reason_pipeline/
│   └── artifacts/
│       └── model/
│           ├── cry_reason_model.keras     (Training artifact)
│           ├── cry_reason_model.tflite    ✓ Copy to board
│           ├── training_info.json         ✓ Copy to board
│           └── threshold_config.json      ✓ Copy to board
│
└── 3_openmv_firmware/
    ├── main.py                      ✓ Upload to board
    ├── audio_preprocessor.py        ✓ Upload to board
    ├── led_output.py                ✓ Upload to board
    ├── test_inference.py            (Use on PC for testing)
    ├── README.md                    (Firmware documentation)
    ├── DEPLOYMENT.md                (This file)
    └── model/
        ├── cry_reason_model.tflite  ✓ Board-side copy
        ├── training_info.json       ✓ Board-side copy
        └── threshold_config.json    ✓ Board-side copy
```

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| Inference Time | 50-100 ms per clip |
| Model Size | 15.7 KB |
| Memory Usage | ~5 MB (peak) |
| Power Draw | 50-100 mA active |
| Test Accuracy | 75.36% |
| Classes | 5 (belly_pain, burping, discomfort, hungry, tired) |
| Sample Rate | 8 kHz |
| Clip Duration | 7 seconds |

---

## Next Steps

Once deployed and working:

1. **Test with real baby cries** - Record samples and test locally
2. **Iterate on model** - If accuracy is poor, retrain with more balanced data
3. **Optimize preprocessing** - Implement faster mel spectrogram on board
4. **Add cloud logging** - Send results to remote server
5. **Edge fine-tuning** - Allow model to adapt to new users

---

## Support

**Documentation:**
- OpenMV IDE: https://docs.openmv.io/
- Nicla Voice: https://docs.arduino.cc/hardware/nicla-voice
- TensorFlow Lite: https://www.tensorflow.org/lite/guide

**Debugging Commands in OpenMV Shell:**
```python
# Check free memory
import gc
print(f"Free memory: {gc.mem_free()} bytes")

# List files
import os
print(os.listdir('/sd'))

# Check model file
import os
print(f"Model size: {os.stat('/sd/model/cry_reason_model.tflite')[6]} bytes")
```

---

**Last Updated**: April 30, 2026
