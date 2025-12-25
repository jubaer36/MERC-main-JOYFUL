#!/usr/bin/env python3
"""
Test OpenSmile LLD averaging based on paper description:
"Audio and visual modalities are utterance-level features by averaging all the token features"
"""
import os
import pickle
import numpy as np
import opensmile

BASE_DIR = "/mnt/Work/ML/Github/MERC-main/JOYFUL/data"
RAW_AUDIO_DIR = f"{BASE_DIR}/raw-data/IEMOCAP_full_release/Session2/sentences/wav/Ses02M_impro05"
PKL_FILE = f"{BASE_DIR}/iemocap/data_iemocap.pkl"

# Load Sample 0
with open(PKL_FILE, 'rb') as f:
    data = pickle.load(f)

sample_0 = data['train'][0]
print(f"Sample 0: {sample_0.vid}")
print(f"Target: {sample_0.audio[0].shape[0]}-D")
print(f"PKL values: {sample_0.audio[0][:10]}\n")

# Get first audio file
audio_files = sorted([f for f in os.listdir(RAW_AUDIO_DIR) 
                      if f.endswith('.wav') and not f.startswith('._')])
first_audio = os.path.join(RAW_AUDIO_DIR, audio_files[0])

# Test all available LLD feature sets
feature_sets = [
    ('eGeMAPSv02', opensmile.FeatureSet.eGeMAPSv02),
    ('GeMAPSv01a', opensmile.FeatureSet.GeMAPSv01a),
    ('GeMAPSv01b', opensmile.FeatureSet.GeMAPSv01b),
    ('eGeMAPSv01a', opensmile.FeatureSet.eGeMAPSv01a),
    ('eGeMAPSv01b', opensmile.FeatureSet.eGeMAPSv01b),
    ('emobase', opensmile.FeatureSet.emobase),
    ('ComParE_2016', opensmile.FeatureSet.ComParE_2016),
]

print(f"Testing: {audio_files[0]}\n")
print("Averaging frame-level LLDs to utterance-level:\n")

for name, feature_set in feature_sets:
    try:
        smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        features = smile.process_file(first_audio)
        averaged = features.values.mean(axis=0)
        
        match = "✓ MATCH!" if averaged.shape[0] == 100 else ""
        print(f"{name:20s} {averaged.shape[0]:4d}-D  {match}")
        
        if averaged.shape[0] == 100:
            print(f"  Values: {averaged[:10]}")
            print(f"  PKL:    {sample_0.audio[0][:10]}")
            print(f"  Correlation: {np.corrcoef(averaged, sample_0.audio[0])[0,1]:.3f}")
        
    except Exception as e:
        print(f"{name:20s} Error: {str(e)[:50]}")
