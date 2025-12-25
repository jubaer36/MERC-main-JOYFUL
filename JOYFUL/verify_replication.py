
import pickle
import numpy as np
import sys
import os

# Ensure joyful module can be imported for Sample class
sys.path.append('/mnt/Work/ML/Github/MERC-main/JOYFUL')
try:
    import joyful
    from joyful.Sample import Sample
except ImportError:
    pass # Might fail but pickle load might still work if class matches

def inspect(path, name):
    print(f"\n--- Inspecting {name}: {path} ---")
    if not os.path.exists(path):
        print("File not found.")
        return None
        
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    stats = {}
    for split in ['train', 'dev', 'test']:
        samples = data.get(split, [])
        print(f"Split {split}: {len(samples)} samples")
        stats[split] = len(samples)
        
        if len(samples) > 0:
            s0 = samples[0]
            print(f"  Sample 0 VID: {s0.vid}")
            print(f"  Attributes: {list(s0.__dict__.keys())}")
            
            # Check shapes/types
            for attr in ['text', 'audio', 'visual', 'sbert_sentence_embeddings']:
                if hasattr(s0, attr):
                    val = getattr(s0, attr)
                    if isinstance(val, np.ndarray):
                        print(f"  {attr} type: ndarray, shape: {val.shape}")
                        print(f"  {attr} non-zero: {np.any(val)}")
                    elif isinstance(val, list):
                        print(f"  {attr} type: list, length: {len(val)}")
                        if len(val) > 0 and hasattr(val[0], 'shape'):
                             print(f"  {attr}[0] shape: {val[0].shape}")
                        # Check zeros for list of arrays
                        is_nonzero = any(np.any(x) for x in val) if len(val)>0 else False
                        print(f"  {attr} non-zero: {is_nonzero}")

    return stats

def main():
    orig_path = '/mnt/Work/ML/Github/MERC-main/JOYFUL/data/iemocap/data_iemocap.pkl'
    new_path = '/mnt/Work/ML/Github/MERC-main/JOYFUL/data/iemocap/data_iemocap_replicated.pkl'
    
    s1 = inspect(orig_path, "ORIGINAL")
    s2 = inspect(new_path, "REPLICATED")
    
    if s1 and s2:
        print("\n--- Comparison ---")
        match = True
        for split in ['train', 'dev', 'test']:
            if s1[split] != s2[split]:
                print(f"MISMATCH in {split}: {s1[split]} vs {s2[split]}")
                match = False
        
        if match:
            print("SUCCESS: Sample counts match across all splits.")
        else:
            print("FAILURE: Sample counts do not match.")

if __name__ == "__main__":
    main()
