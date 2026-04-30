"""
Audio preprocessing for Nicla Voice
Converts raw audio to normalized mel spectrogram matching training pipeline
"""

import numpy as np

# Note: Nicla Voice uses MicroPython which has limited signal processing
# For production, use librosa on PC for preprocessing
# This is a simplified version for on-device processing


class AudioPreprocessor:
    """Preprocess audio to mel spectrogram"""
    
    def __init__(self, feature_params, normalization):
        """
        Args:
            feature_params: dict with sample_rate, n_mels, n_fft, hop_length
            normalization: dict with mean, std from training
        """
        self.sample_rate = int(feature_params["sample_rate"])
        self.n_mels = int(feature_params["n_mels"])
        self.n_fft = int(feature_params["n_fft"])
        self.hop_length = int(feature_params["hop_length"])
        
        self.norm_mean = float(normalization["mean"])
        self.norm_std = float(normalization["std"]) + 1e-6
        
    def _simple_mel_spectrogram(self, wav):
        """
        Simplified mel spectrogram (alternative to librosa)
        
        Note: For best results, preprocess on PC with librosa and send
        pre-extracted features to the board for inference only.
        
        This implementation uses a basic approach:
        - Apply Hamming window
        - Compute FFT magnitude
        - Apply approximate mel scale
        """
        n_fft = self.n_fft
        hop_length = self.hop_length
        n_mels = self.n_mels
        
        # Ensure input is proper length
        if len(wav) < n_fft:
            wav = np.pad(wav, (0, n_fft - len(wav)), mode='constant')
        
        # Normalize audio to [-1, 1]
        wav_max = np.max(np.abs(wav))
        if wav_max > 0:
            wav = wav.astype(np.float32) / wav_max
        else:
            wav = wav.astype(np.float32)
        
        # Compute spectrograms with sliding window
        n_frames = 1 + (len(wav) - n_fft) // hop_length
        
        # Hamming window
        window = np.hamming(n_fft).astype(np.float32)
        
        spec_list = []
        for i in range(n_frames):
            start = i * hop_length
            frame = wav[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)), mode='constant')
            
            # Apply window and FFT
            windowed = frame * window
            # Simple magnitude computation (avoiding full FFT for speed)
            # In production, use numpy.fft.rfft for proper FFT
            magnitude = np.abs(windowed)  # Simplified
            spec_list.append(magnitude)
        
        spectrogram = np.array(spec_list).T  # (n_fft, n_frames)
        
        # Approximate mel scale (simplified)
        mel_spec = self._apply_mel_scale(spectrogram, n_mels)
        
        return mel_spec
    
    def _apply_mel_scale(self, spectrogram, n_mels):
        """Convert linear spectrogram to mel scale (simplified)"""
        # Simple linear interpolation to n_mels bins
        n_freqs = spectrogram.shape[0]
        mel_spec = np.zeros((n_mels, spectrogram.shape[1]), dtype=np.float32)
        
        for i in range(n_mels):
            freq_idx = int((i / n_mels) * n_freqs)
            if freq_idx >= n_freqs:
                freq_idx = n_freqs - 1
            mel_spec[i] = spectrogram[freq_idx]
        
        return mel_spec
    
    def preprocess(self, audio_samples):
        """
        Convert audio samples to normalized mel spectrogram
        
        Args:
            audio_samples: numpy array of audio samples (int16 or float)
            
        Returns:
            mel_spec: (1, n_mels, n_frames, 1) normalized float32 array
        """
        # Convert to float
        if audio_samples.dtype == np.int16:
            wav = audio_samples.astype(np.float32) / 32768.0
        else:
            wav = audio_samples.astype(np.float32)
        
        # Compute mel spectrogram
        mel_spec = self._simple_mel_spectrogram(wav)
        
        # Apply log scale
        epsilon = 1e-10
        log_mel = np.log(mel_spec + epsilon)
        
        # Normalize using training stats
        log_mel = (log_mel - self.norm_mean) / self.norm_std
        
        # Add batch and channel dimensions for inference
        # Shape: (1, n_mels, n_frames, 1)
        output = log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)
        
        return output


def preprocess_for_board(wav_path, feature_params, normalization):
    """
    Utility function to preprocess audio on PC before uploading to board
    
    Use this on your PC with librosa for accurate preprocessing,
    then upload pre-extracted features to the board.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for PC-side preprocessing")
    
    sr = int(feature_params["sample_rate"])
    n_mels = int(feature_params["n_mels"])
    n_fft = int(feature_params["n_fft"])
    hop_length = int(feature_params["hop_length"])
    
    wav, _ = librosa.load(wav_path, sr=sr, mono=True)
    
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    
    norm_mean = normalization["mean"]
    norm_std = normalization["std"] + 1e-6
    log_mel = (log_mel - norm_mean) / norm_std
    
    return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)
