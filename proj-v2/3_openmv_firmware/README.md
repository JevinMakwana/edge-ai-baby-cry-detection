# Nicla Voice: Baby Cry Reason Classifier

Real-time cry reason classification on Nicla Voice board with LED + serial output.

## System Overview

```
Nicla Voice Microphone
    ↓
Audio Capture (7-second clips at 8kHz)
    ↓
Mel Spectrogram Preprocessing (Normalized)
    ↓
TFLite Model Inference
    ↓
Per-class Threshold Adjustment
    ↓
├─→ LED Color Indicator
└─→ Serial Monitor Output
```

## Supported Cry Reasons & LED Colors

| Cry Reason   | LED Color | RGB        |
|--------------|-----------|------------|
| Belly Pain   | Red       | (255, 0, 0)       |
| Burping      | Green     | (0, 255, 0)       |
| Discomfort   | Orange    | (255, 165, 0)     |
| Hungry       | Blue      | (0, 0, 255)       |
| Tired        | Magenta   | (255, 0, 255)     |

## Hardware Requirements

- **Nicla Voice** board
- **Micro-USB cable** for connection and power
- **Computer with OpenMV IDE** (free download: https://openmv.io/)

## File Structure

```
3_openmv_firmware/
├── main.py                    # Entry point - continuous audio capture loop
├── audio_preprocessor.py      # Mel spectrogram preprocessing
├── led_output.py              # LED + serial output control
├── test_inference.py          # Testing script (run on PC first)
├── model/
│   ├── cry_reason_model.tflite     # Compiled model
│   ├── training_info.json          # Normalization + labels
│   └── threshold_config.json       # Per-class thresholds
└── README.md                  # This file
```

## Installation & Deployment

### Step 1: Set Up OpenMV IDE

1. Download OpenMV IDE from https://openmv.io/
2. Install and launch
3. Connect Nicla Voice via USB (red LED should light up)
4. OpenMV IDE should auto-detect the board

### Step 2: Copy Files to Board

**Via OpenMV IDE (Recommended):**

1. Click **Tools** → **File System** → **New File System Image**
2. Drag-and-drop these files into the file system:
   - `main.py`
   - `audio_preprocessor.py`
   - `led_output.py`
   - `model/cry_reason_model.tflite`
   - `model/training_info.json`
   - `model/threshold_config.json`

3. Click **Eject** to save the file system to the board

**Or via command line:**
```bash
# If using OpenMV CLI tools
openmv-sync-files 3_openmv_firmware/ /sd/
```

### Step 3: Run on Board

1. In OpenMV IDE, open `main.py`
2. Click the **Run** button (▶) or press **Ctrl+R**
3. Open **Tools** → **Serial Monitor** (or **Ctrl+Shift+C**)
4. Set baud rate to **9600**
5. You should see:
   ```
   ============================================================
   Baby Cry Reason Classifier - Nicla Voice
   ============================================================
   Labels: ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
   Normalization: mean=-45.34, std=15.92
   
   Starting audio capture... (Listening for cries)
   ============================================================
   [Frame 0] Predicted: hungry (0.8523)
     belly_pain: 0.0234
     burping: 0.0123
     discomfort: 0.0120
     hungry: 0.8523
     tired: 0.1000
   ```

## Real-Time Operation

Once running:

1. **Audio Input**: Continuously captures 7-second audio clips from the onboard microphone
2. **Processing**: Each clip is:
   - Converted to mel spectrogram (40 bins, 7s window)
   - Normalized using training statistics
   - Sent to TFLite interpreter
3. **Output**:
   - **LED** changes color to match predicted cry reason
   - **Serial** prints prediction confidence and all class probabilities
4. **Loop**: Repeats every ~7 seconds

## Testing Locally (PC)

Before deploying to the board, test the model on your PC:

```bash
cd proj-v2/3_openmv_firmware

# Test with a sample audio clip
python test_inference.py --clip path/to/cry_sample.wav
```

This will show:
- Mel spectrogram visualization
- Raw model output
- Final prediction with thresholds applied
- Timing metrics

## Troubleshooting

### Serial Monitor Shows Nothing
- Check USB connection (red LED on board should be lit)
- In OpenMV IDE: **Tools** → **Reset OpenMV Cam** 
- Verify baud rate is 9600

### LED Not Lighting Up
- Verify LED is physically connected
- Check `led_output.py` pin assignment matches your board
- Add debug print to confirm color commands are being sent

### Model Inference Errors
- Verify all files in `model/` folder:
  - `cry_reason_model.tflite` (should be ~16 KB)
  - `training_info.json` (should be readable JSON)
  - `threshold_config.json` (should be readable JSON)
- Check paths in `main.py` match `/sd/model/` structure

### Poor Prediction Accuracy
- Ensure microphone is not blocked
- Test with clear, distinct baby cry sounds
- Verify audio is similar to training dataset (8 kHz, mono)
- Check normalization is applied correctly

## Model Details

**Training Configuration:**
- Audio: 8 kHz sample rate, 7-second clips, mono
- Preprocessing: 40 mel bins, FFT size 512, hop 160
- Model: 3-layer CNN with BatchNorm, GlobalAveragePooling
- Export: TFLite with int8 quantization
- Test Accuracy: 75.36%

**Files in `model/`:**
- `cry_reason_model.tflite`: Quantized model for fast inference
- `training_info.json`: Normalization parameters + class labels
- `threshold_config.json`: Per-class probability thresholds for each cry reason

## Performance Metrics

- **Inference time**: ~50-100ms per 7-second clip (on-device)
- **Model size**: 15.7 KB (fits easily on board)
- **Memory usage**: ~5 MB during inference
- **Power draw**: ~50-100mA during audio capture + inference

## Future Enhancements

- [ ] Real-time streaming without waiting for full 7-second clip
- [ ] Sliding window detection (detect cries mid-clip)
- [ ] Edge case handling (noise, silence)
- [ ] Confidence threshold filtering
- [ ] Remote logging to cloud
- [ ] Model on-device fine-tuning with new data

## Support & Debugging

**Serial output debug levels:**

Add to `main.py`:
```python
DEBUG = True  # Enable verbose output

if DEBUG:
    print(f"Raw model output: {output}")
    print(f"Thresholds applied: {threshold_config}")
```

**Capture audio for analysis:**

```python
# Add to main.py to save raw audio
import pickle
pickle.dump(audio_buffer, open('/sd/debug_audio.pkl', 'wb'))
```

Then analyze offline with Python:
```python
import pickle
import numpy as np
audio = pickle.load(open('debug_audio.pkl', 'rb'))
print(f"Audio shape: {audio.shape}, min: {audio.min()}, max: {audio.max()}")
```

---

**Last Updated**: April 30, 2026
**Model**: cry_reason_model.tflite (75.36% test accuracy)
