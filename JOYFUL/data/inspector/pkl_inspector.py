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
    
    # Dynamically get all attributes
    attributes = sorted([attr for attr in dir(sample) if not attr.startswith('__') and not callable(getattr(sample, attr))])
    
    # Filter out standard methods if any remain (though callable check helps)
    # Also prioritize specific interesting attributes
    priority_attrs = ['vid', 'speaker', 'label', 'text', 'audio', 'visual', 'sentence', 'sbert_sentence_embeddings']
    other_attrs = [attr for attr in attributes if attr not in priority_attrs]
    
    all_attrs = priority_attrs + other_attrs
    
    # Dedup and keep order
    seen = set()
    final_attrs = []
    for attr in all_attrs:
        if attr in attributes and attr not in seen:
            final_attrs.append(attr)
            seen.add(attr)
            
    for attr in final_attrs:
        value = getattr(sample, attr)
        print(f"\n  Attribute: {attr}")
        
        # Enhanced inspection logic
        if hasattr(value, 'shape'):
            print(f"    Type: {type(value).__name__}")
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {getattr(value, 'dtype', 'N/A')}")
            # Statistics for arrays
            if hasattr(value, 'min') and hasattr(value, 'max') and hasattr(value, 'mean'):
                try:
                    print(f"    Stats: min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}, std={value.std():.4f}")
                except:
                    pass
            
            # Print content for small arrays or first row for large ones
            if value.size < 20:
                print(f"    Value: {value}")
            else:
                print(f"    First row/sample:\n{value[0]}")
                
        elif isinstance(value, (list, tuple)):
            print(f"    Type: {type(value).__name__}")
            print(f"    Length: {len(value)}")
            if len(value) > 0:
                # Inspect content type uniformity
                types = set(type(x).__name__ for x in value)
                if len(types) == 1:
                    print(f"    Content Type: list of {list(types)[0]}")
                else:
                    print(f"    Content Types: {types}")
                
                # If list of strings or ints, print all or most
                first_elem = value[0]
                if isinstance(first_elem, (str, int, float, bool)):
                    print(f"    Values: {value}")
                elif hasattr(first_elem, 'shape'):
                    print(f"    Element Shape (from index 0): {first_elem.shape}")
                    print(f"    Element Dtype: {getattr(first_elem, 'dtype', 'N/A')}")
                    # Stats for the list of arrays (aggregate maybe? or just first element details)
                    print(f"    First Element content:\n{first_elem}")
                else:
                    print(f"    First 5 elements: {value[:5]}")

        elif isinstance(value, str):
            print(f"    Type: str")
            print(f"    Length: {len(value)}")
            print(f"    Value: '{value}'")
            
        elif isinstance(value, (int, float)):
            print(f"    Type: {type(value).__name__}")
            print(f"    Value: {value}")
            
        else:
            print(f"    Type: {type(value).__name__}")
            print(f"    Value: {repr(value)}")


def inspect_pickle(file_path, detailed_samples=1, target_split=None):
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
            
            keys_to_inspect = [target_split] if target_split and target_split in data else data.keys()
            
            for key in keys_to_inspect:
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
    # file_2 = os.path.join(base_dir, "iemocap_4", "data_iemocap_4.pkl")
    
    print("=" * 60)
    print("JOYFUL DATA INSPECTOR")
    print("=" * 60)
    
    inspect_pickle(file_1, detailed_samples=1, target_split='train')
