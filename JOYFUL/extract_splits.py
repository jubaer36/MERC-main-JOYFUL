
import pickle
import json
import sys
import os

# Add project root to sys.path
sys.path.append('/mnt/Work/ML/Github/MERC-main/JOYFUL')

try:
    import joyful
    from joyful.Sample import Sample
except ImportError:
    print("Could not import joyful.Sample")

def extract():
    pkl_path = '/mnt/Work/ML/Github/MERC-main/JOYFUL/data/iemocap/data_iemocap.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    splits = {}
    for key in ['train', 'dev', 'test']:
        # Extract .vid attribute from each Sample object
        splits[key] = [s.vid for s in data[key]]
        
    print(json.dumps(splits, indent=2))

if __name__ == "__main__":
    extract()
