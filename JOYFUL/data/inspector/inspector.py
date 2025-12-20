import pickle
import os
import sys
import numpy as np
import traceback

# Add project root to sys.path to ensure 'joyful' module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try importing joyful to verify fix
try:
    import joyful
    from joyful.Sample import Sample
except ImportError as e:
    print(f"Warning: Could not import 'joyful' even after path adjustment: {e}")


def describe_value(value, indent=4, max_items=3):
    """Recursively describe a value's type, shape, and contents."""
    prefix = " " * indent
    desc = []
    
    if hasattr(value, 'shape'):
        desc.append(f"{prefix}Type: {type(value).__name__}, Shape: {value.shape}, Dtype: {getattr(value, 'dtype', 'N/A')}")
    elif isinstance(value, dict):
        desc.append(f"{prefix}Type: dict, Keys: {list(value.keys())}")
        for i, (k, v) in enumerate(value.items()):
            if i >= max_items:
                desc.append(f"{prefix}  ... and {len(value) - max_items} more keys")
                break
            desc.append(f"{prefix}  Key '{k}':")
            desc.extend(describe_value(v, indent + 4, max_items))
    elif isinstance(value, (list, tuple)):
        desc.append(f"{prefix}Type: {type(value).__name__}, Length: {len(value)}")
        if len(value) > 0:
            desc.append(f"{prefix}  First element:")
            desc.extend(describe_value(value[0], indent + 4, max_items))
    elif isinstance(value, str):
        desc.append(f"{prefix}Type: str, Value: '{value[:100]}{'...' if len(value) > 100 else ''}'")
    elif isinstance(value, (int, float, bool)):
        desc.append(f"{prefix}Type: {type(value).__name__}, Value: {value}")
    else:
        desc.append(f"{prefix}Type: {type(value).__name__}, Value (repr): {repr(value)[:100]}")
    return desc


def inspect_sample(sample, sample_idx=0):
    """Inspect a joyful.Sample.Sample object and print its attributes."""
    print(f"\n  --- Sample {sample_idx} Details ---")
    
    attrs = ['vid', 'speaker', 'label', 'text', 'audio', 'visual', 'sentence', 'sbert_sentence_embeddings']
    
    for attr in attrs:
        if hasattr(sample, attr):
            value = getattr(sample, attr)
            print(f"\n  Attribute: {attr}")
            
            if hasattr(value, 'shape'):
                print(f"    Type: {type(value).__name__}")
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {getattr(value, 'dtype', 'N/A')}")
            elif isinstance(value, (list, tuple)):
                print(f"    Type: {type(value).__name__}")
                print(f"    Length: {len(value)}")
                if len(value) > 0:
                    first_elem = value[0]
                    if hasattr(first_elem, 'shape'):
                        print(f"    First element shape: {first_elem.shape}")
                    else:
                        print(f"    First element type: {type(first_elem).__name__}")
                        if isinstance(first_elem, str):
                            print(f"    First element value: '{first_elem[:50]}...'")
            elif isinstance(value, str):
                preview = value[:100] + ('...' if len(value) > 100 else '')
                print(f"    Type: str")
                print(f"    Length: {len(value)}")
                print(f"    Preview: '{preview}'")
            elif isinstance(value, (int, float)):
                print(f"    Type: {type(value).__name__}")
                print(f"    Value: {value}")
            else:
                print(f"    Type: {type(value).__name__}")
                print(f"    Value: {repr(value)[:100]}")


def inspect_pickle(file_path, detailed_samples=1):
    print(f"\n{'='*60}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*60}")
    
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nTop-level data type: {type(data).__name__}")
        
        # Handle dict-based pickles (common for train/val/test splits)
        if isinstance(data, dict):
            print(f"Number of top-level keys: {len(data)}")
            print(f"Keys: {list(data.keys())}")
            
            total_samples = sum(len(v) for v in data.values() if isinstance(v, (list, tuple)))
            print(f"\nTotal samples across all splits: {total_samples}")
            
            for key in data.keys():
                print(f"\n{'='*40}")
                print(f"Split: '{key}'")
                print(f"{'='*40}")
                value = data[key]
                if isinstance(value, (list, tuple)):
                    print(f"Number of samples: {len(value)}")
                    
                    # Inspect first N samples in detail
                    for i in range(min(detailed_samples, len(value))):
                        sample = value[i]
                        if hasattr(sample, 'vid'):  # Check if it's a Sample object
                            inspect_sample(sample, i)
                        else:
                            print(f"\n  Sample {i}:")
                            for line in describe_value(sample, indent=4):
                                print(line)
                    
                    # Summary statistics for this split
                    if len(value) > 0 and hasattr(value[0], 'label'):
                        # Labels are per-utterance, so flatten them
                        all_labels = []
                        total_utterances = 0
                        for s in value:
                            if hasattr(s, 'label') and isinstance(s.label, list):
                                all_labels.extend(s.label)
                                total_utterances += len(s.label)
                        
                        if all_labels:
                            unique_labels = set(all_labels)
                            print(f"\n  Total utterances in '{key}': {total_utterances}")
                            print(f"  Label distribution (per utterance):")
                            for label in sorted(unique_labels):
                                count = all_labels.count(label)
                                print(f"    Label {label}: {count} utterances ({100*count/len(all_labels):.1f}%)")
        
        # Handle list/tuple-based pickles
        elif isinstance(data, (list, tuple)):
            print(f"Number of samples: {len(data)}")
            if len(data) > 0:
                for i in range(min(detailed_samples, len(data))):
                    sample = data[i]
                    if hasattr(sample, 'vid'):
                        inspect_sample(sample, i)
                    else:
                        print(f"\n  Sample {i}:")
                        for line in describe_value(sample, indent=4):
                            print(line)
        
        else:
            for line in describe_value(data, indent=2):
                print(line)

    except Exception as e:
        print(f"Error inspecting file: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    file_1 = os.path.join(base_dir, "iemocap", "data_iemocap.pkl")
    file_2 = os.path.join(base_dir, "iemocap_4", "data_iemocap_4.pkl")
    
    print("=" * 60)
    print("JOYFUL DATA INSPECTOR")
    print("=" * 60)
    
    inspect_pickle(file_1, detailed_samples=1)
    inspect_pickle(file_2, detailed_samples=1)
