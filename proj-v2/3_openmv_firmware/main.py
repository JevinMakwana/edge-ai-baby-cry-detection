"""
Nicla Voice: Baby Cry Reason Classification Firmware
Continuous audio capture -> TFLite inference -> LED + Serial output
"""

import json
import numpy as np
import tf_interpreter
from audio_preprocessor import AudioPreprocessor
from led_output import LEDController

# ==================== CONFIG ====================
MODEL_PATH = "/sd/model/cry_reason_model.tflite"
TRAINING_INFO_PATH = "/sd/model/training_info.json"
THRESHOLD_CONFIG_PATH = "/sd/model/threshold_config.json"

SAMPLE_RATE = 8000
CLIP_SECONDS = 7.0
BUFFER_SIZE = int(SAMPLE_RATE * CLIP_SECONDS)  # 56000 samples

# Class to LED color mapping (RGB)
CLASS_COLORS = {
    "belly_pain": (255, 0, 0),      # Red
    "burping": (0, 255, 0),         # Green
    "discomfort": (255, 165, 0),    # Orange
    "hungry": (0, 0, 255),          # Blue
    "tired": (255, 0, 255),         # Magenta
}

# ==================== SETUP ====================

def load_config():
    """Load training info and threshold config"""
    with open(TRAINING_INFO_PATH, 'r') as f:
        training_info = json.load(f)
    
    try:
        with open(THRESHOLD_CONFIG_PATH, 'r') as f:
            threshold_config = json.load(f)
    except:
        threshold_config = None
    
    return training_info, threshold_config


def apply_thresholds(probs, threshold_config, index_to_label):
    """Apply per-class thresholds to get final prediction"""
    if threshold_config is None or threshold_config.get("type") != "per-class":
        # Default: argmax
        pred_idx = int(np.argmax(probs))
        return pred_idx, float(probs[pred_idx])
    
    thresholds = threshold_config.get("thresholds", {})
    ordered_indices = np.argsort(probs)[::-1]
    
    for class_idx in ordered_indices:
        class_idx = int(class_idx)
        thresh = float(thresholds.get(str(class_idx), 0.5))
        if probs[class_idx] >= thresh:
            return class_idx, float(probs[class_idx])
    
    # Fallback: highest probability
    pred_idx = int(ordered_indices[0])
    return pred_idx, float(probs[pred_idx])


def main():
    print("=" * 60)
    print("Baby Cry Reason Classifier - Nicla Voice")
    print("=" * 60)
    
    # Load configs
    training_info, threshold_config = load_config()
    
    normalization = training_info["normalization"]
    index_to_label = {int(k): v for k, v in training_info["index_to_label"].items()}
    feature_params = training_info["feature_params"]
    
    print(f"Labels: {list(index_to_label.values())}")
    print(f"Normalization: mean={normalization['mean']:.2f}, std={normalization['std']:.2f}")
    print()
    
    # Initialize components
    preprocessor = AudioPreprocessor(feature_params, normalization)
    led_controller = LEDController()
    
    # Load TFLite interpreter
    print("Loading TFLite model...")
    interpreter = tf_interpreter.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print()
    print("Starting audio capture... (Listening for cries)")
    print("=" * 60)
    
    # Continuous audio capture loop
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.int16)
    buffer_idx = 0
    frame_count = 0
    
    try:
        while True:
            # Read audio frame (e.g., 256 samples at 8kHz ≈ 32ms)
            frame = tf_micro.get_audio_sample()
            
            if frame is not None:
                # Add frame to buffer
                frame_len = len(frame)
                remaining = BUFFER_SIZE - buffer_idx
                
                if frame_len <= remaining:
                    audio_buffer[buffer_idx:buffer_idx + frame_len] = frame
                    buffer_idx += frame_len
                else:
                    # Buffer full, process it
                    audio_buffer[buffer_idx:] = frame[:remaining]
                    
                    # Preprocess
                    mel_spec = preprocessor.preprocess(audio_buffer)
                    
                    # Inference
                    interpreter.set_tensor(input_details[0]["index"], mel_spec)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details[0]["index"])
                    
                    # Apply thresholds
                    pred_idx, pred_prob = apply_thresholds(output[0], threshold_config, index_to_label)
                    pred_label = index_to_label[pred_idx]
                    
                    # Output results
                    print(f"[Frame {frame_count}] Predicted: {pred_label} ({pred_prob:.4f})")
                    
                    # Set LED color
                    color = CLASS_COLORS.get(pred_label, (255, 255, 255))
                    led_controller.set_color(color)
                    
                    # Print all class probabilities
                    for idx in range(len(index_to_label)):
                        label = index_to_label[idx]
                        prob = output[0][idx]
                        print(f"  {label}: {prob:.4f}")
                    print()
                    
                    # Reset buffer
                    overflow_len = frame_len - remaining
                    if overflow_len > 0:
                        audio_buffer[:overflow_len] = frame[remaining:]
                        buffer_idx = overflow_len
                    else:
                        buffer_idx = 0
                    
                    frame_count += 1
    
    except KeyboardInterrupt:
        print("Stopped by user")
        led_controller.set_color((0, 0, 0))  # Turn off LED


if __name__ == "__main__":
    main()
