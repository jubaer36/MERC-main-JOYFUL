#!/usr/bin/env python3
"""
Test librosa with different normalization strategies
"""
import os
import pickle
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BASE_DIR = "/mnt/Work/ML/Github/MERC-main/JOYFUL/data"
RAW_AUDIO_DIR = f"{BASE_DIR}/raw-data/IEMOCAP_full_release/Session2/sentences/wav/Ses02M_impro05"
PKL_FILE = f"{BASE_DIR}/iemocap/data_iemocap.pkl"

def extract_100d_features(audio_path):
    """Extract 100-D audio features using librosa"""
    y, sr = librosa.load(audio_path, sr=None)
    features = []
    
    # MFCCs (40-D)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfccs, axis=1))
    
    # Spectral (11-D)
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_flatness(y=y)))
    features.extend(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1))
    
    # Chroma (12-D)
    features.extend(np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1))
    
    # ZCR + RMS (2-D)
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    
    # Tonnetz (6-D)
    features.extend(np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1))
    
    # Mel Spectrogram (29-D)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=29)
    features.extend(np.mean(mel_spec, axis=1))
    
    return np.array(features)

# Load Sample 0
with open(PKL_FILE, 'rb') as f:
    data = pickle.load(f)

sample_0 = data['train'][0]
pkl_features = sample_0.audio[0]

# Get audio files
audio_files = sorted([f for f in os.listdir(RAW_AUDIO_DIR) 
                      if f.endswith('.wav') and not f.startswith('._')])
first_audio = os.path.join(RAW_AUDIO_DIR, audio_files[0])

# Extract features
raw_features = extract_100d_features(first_audio)

print(f"Sample 0: {sample_0.vid}")
print(f"Testing: {audio_files[0]}\n")

# Test different normalizations
normalizations = [
    ("Raw", raw_features),
    ("Min-Max [0,1]", MinMaxScaler().fit_transform(raw_features.reshape(-1, 1)).flatten()),
    ("Z-score", StandardScaler().fit_transform(raw_features.reshape(-1, 1)).flatten()),
    ("L2-norm", raw_features / np.linalg.norm(raw_features)),
]

print(f"PKL values:  {pkl_features[:10]}")
print(f"PKL stats: min={pkl_features.min():.3f}, max={pkl_features.max():.3f}, mean={pkl_features.mean():.3f}\n")

for name, features in normalizations:
    corr = np.corrcoef(features, pkl_features)[0, 1]
    print(f"{name:15s}: {features[:10]}")
    print(f"                 Stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")
    print(f"                 Correlation: {corr:.4f}\n")
