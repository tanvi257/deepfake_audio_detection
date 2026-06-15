import os
import librosa
import numpy as np
import soundfile as sf
import warnings

def load_and_preprocess_audio(file_path, target_sr=16000, duration=3.0):
    """
    Loads an audio file, resamples to target_sr, trims silence, 
    normalizes amplitude, and pads/crops it to a fixed duration.
    """
    try:
        # Load audio (suppress warnings from librosa/soundfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=target_sr)
            
        # Trim leading and trailing silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) == 0:
            return None
            
        # Normalize amplitude to [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
            
        # Standardize length (crop or pad to target duration)
        target_len = int(target_sr * duration)
        if len(y) > target_len:
            # Crop center
            start = (len(y) - target_len) // 2
            y = y[start:start + target_len]
        else:
            # Pad with zeros
            pad_len = target_len - len(y)
            y = np.pad(y, (0, pad_len), 'constant')
            
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features(y, sr=16000, n_mfcc=20):
    """
    Extracts statistical features from the processed audio:
    MFCCs (mean and std), Spectral Centroid, Spectral Rolloff, 
    Zero Crossing Rate, and Chroma.
    """
    if y is None or len(y) == 0:
        return None
        
    try:
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sb_mean = np.mean(spectral_bandwidth)
        sb_std = np.std(spectral_bandwidth)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        # Concatenate all features into a 1D vector
        feature_vector = np.concatenate([
            mfcc_mean,        # 20
            mfcc_std,         # 20
            mfcc_delta_mean,  # 20
            mfcc_delta_std,   # 20
            [sc_mean, sc_std, sb_mean, sb_std, rolloff_mean, zcr_mean, rms_mean] # 7
        ])
        
        return feature_vector # total 87 features
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None
