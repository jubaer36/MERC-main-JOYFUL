import os
import re
import pickle
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys

# Ensure joyful module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from joyful.Sample import Sample, sbert_model
except ImportError:
    print("Could not import joyful.Sample. Make sure you are in the project root.")
    # Dummy class if import fails
    class Sample:
        def __init__(self, vid, speaker, label, text, audio, visual, sentence): 
            pass
    sbert_model = None

# Optional imports for features
try:
    import opensmile
except ImportError:
    opensmile = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

# =========================
# Configuration
# =========================
EMOTION_MAP = {
    'hap': 0,  # Happiness
    'sad': 1,  # Sadness
    'neu': 2,  # Neutral
    'ang': 3,  # Anger
    'exc': 4,  # Excited
    'fru': 5,  # Frustrated
}

# JOYFUL uses 6 classes. 
# "oth", "fea", "sur", "dis" are usually discarded or mapped if needed.
# Strict replication of JOYFUL means we likely only keep these 6.

# =========================
# Helpers
# =========================
def parse_emotions(data_dir):
    """
    Parse EmoEvaluation files to get {utterance_id: (label_idx, start, end, speaker)}
    """
    ann = {}
    for sess in sorted(os.listdir(data_dir)):
        if not sess.startswith("Session"): continue
        
        emo_dir = os.path.join(data_dir, sess, "dialog", "EmoEvaluation")
        if not os.path.exists(emo_dir): continue
        
        for f in sorted(os.listdir(emo_dir)):
            if not f.endswith(".txt") or f.startswith("._"): continue
            
            with open(os.path.join(emo_dir, f), "r", errors="ignore") as fh:
                for line in fh:
                    # Format: [START - END] TURN_NAME EMOTION CONFIDENCE ...
                    # Example: [6.2900 - 8.2350] Ses01F_impro01_F000 neu [2.5000, 2.5000, 2.5000, 2.5000] ...
                    m = re.match(r"\[(\d+\.\d+)\s-\s(\d+\.\d+)\]\s(\S+)\s(\w+)", line)
                    if m:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        utt_id = m.group(3)
                        emo = m.group(4)
                        
                        if emo in EMOTION_MAP:
                            ann[utt_id] = {
                                "label": EMOTION_MAP[emo],
                                "start": start,
                                "end": end,
                                "speaker": utt_id.split("_")[-1][0] # 'F' or 'M' typically
                            }
    return ann

def parse_transcripts(data_dir):
    """
    Parse transcriptions to get {utterance_id: text}
    """
    tr = {}
    for sess in sorted(os.listdir(data_dir)):
        if not sess.startswith("Session"): continue
        
        tr_dir = os.path.join(data_dir, sess, "dialog", "transcriptions")
        if not os.path.exists(tr_dir): continue
        
        for f in sorted(os.listdir(tr_dir)):
            if not f.endswith(".txt") or f.startswith("._"): continue
            
            with open(os.path.join(tr_dir, f), "r", errors="ignore") as fh:
                for line in fh:
                    # Format: Ses01F_impro01_F000 [START-END]: Text
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        utt_info = parts[0].strip().split(" ")
                        if len(utt_info) >= 1:
                            utt_id = utt_info[0]
                            text = parts[1].strip()
                            tr[utt_id] = text
    return tr

def load_splits_from_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Split file not found: {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)

# =========================
# Feature Extractors
# =========================
class FeaturePipeline:
    def __init__(self, use_opensmile=False, use_openface=False):
        # Text
        print("Using SBERT model from joyful.Sample...")
        if sbert_model:
            self.sbert = sbert_model
        else:
            # Fallback (should not happen if import works)
            print("Fallback: loading local SBERT model")
            self.sbert = SentenceTransformer("paraphrase-distilroberta-base-v1")
        
        self.pca = None
        
        # Audio
        self.smile = None
        if use_opensmile:
            if opensmile:
                # IS13-ComParE yields ~6k features. eGeMAPS yields ~88. 
                # JOYFUL paper says 100-d audio. 
                # We will use eGeMAPSv02 -> 88 dims. Padding to 100.
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
            else:
                print("Warning: 'opensmile' lib not installed. Skipping audio extraction.")
        
        # Visual
        self.use_openface = use_openface

    def extract_text_sbert(self, texts):
        return self.sbert.encode(texts, convert_to_numpy=True)
    
    def train_text_pca(self, embeddings, n_components=100):
        if PCA is None:
            print("Warning: sklearn not installed, cannot compute PCA.")
            return
        self.pca = PCA(n_components=n_components)
        self.pca.fit(embeddings)
        print(f"PCA fitted. Explained variance: {np.sum(self.pca.explained_variance_ratio_):.2f}")

    def get_text_100d(self, embeddings):
        if self.pca is None:
            return np.zeros((len(embeddings), 100), dtype=np.float32)
        return self.pca.transform(embeddings).astype(np.float32)

    def extract_audio(self, wav_path):
        if self.smile and os.path.exists(wav_path):
            try:
                # Returns DataFrame
                y = self.smile.process_file(wav_path)
                vec = y.values[0] # 88 dims for eGeMAPS
                # Pad to 100
                if len(vec) < 100:
                    vec = np.pad(vec, (0, 100 - len(vec)), 'constant')
                elif len(vec) > 100:
                    vec = vec[:100]
                return vec.astype(np.float32)
            except Exception as e:
                print(f"Error processing audio {wav_path}: {e}")
                return np.zeros(100, dtype=np.float32)
        return np.zeros(100, dtype=np.float32)

    def extract_visual(self, vid_id, feat_dir):
        # Path: feat_dir/openface/vid_id.csv
        # Note: OpenFace docker script outputs with same basename
        csv_path = os.path.join(feat_dir, "openface", f"{vid_id}.csv")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Filter valid frames
                if 'success' in df.columns:
                    df = df[df['success'] == 1]
                
                # Drop non-feature columns
                drops = ['frame', 'face_id', 'timestamp', 'confidence', 'success']
                cols = [c for c in df.columns if c.strip() not in drops and not c.startswith('p_')] # p_ are 3D points usually, keep them?
                # OpenFace CSV has lots of columns. 
                # To get 512 dimensions?
                # Let's just take all numeric columns except metadata.
                numeric_df = df.select_dtypes(include=[np.number])
                feature_cols = [c for c in numeric_df.columns if c.strip() not in drops]
                
                if len(feature_cols) == 0:
                     return np.zeros(512, dtype=np.float32)

                x = numeric_df[feature_cols].values
                
                # Functional: Mean + Std
                vec = np.concatenate([x.mean(0), x.std(0)])
                
                # Check NaNs
                vec = np.nan_to_num(vec)
                
                # Force to 512-d
                target_dim = 512
                if len(vec) < target_dim:
                    vec = np.pad(vec, (0, target_dim - len(vec)), 'constant')
                elif len(vec) > target_dim:
                    # PCA would be better but we can't fit it easily in this loop structure without 2 passes
                    # Just truncate for now to strictly match dimension
                    vec = vec[:target_dim]
                    
                return vec.astype(np.float32)
            except Exception as e:
                print(f"Error reading visual {csv_path}: {e}")
                return np.zeros(512, dtype=np.float32)
        
        return np.zeros(512, dtype=np.float32)


# =========================
# Main Logic
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to IEMOCAP_full_release")
    parser.add_argument("--feat_dir", default="features", help="Path to store/read raw features")
    parser.add_argument("--out", default="data/iemocap/data_iemocap.pkl", help="Output pickle path")
    parser.add_argument("--splits_file", default="joyful/utilities/iemocap_splits.json", help="Path to splits definition")
    args = parser.parse_args()

    # 1. Parse Annotations & Transcripts
    print(f"Parsing data from {args.data_dir}...")
    annotations = parse_emotions(args.data_dir)
    transcripts = parse_transcripts(args.data_dir)
    
    # Filter: Intersect texts and labels
    valid_utts = set(annotations.keys()) & set(transcripts.keys())
    print(f"Found {len(valid_utts)} valid utterances with both label and text.")

    # 2. Group by Dialogue
    # vid -> list of valid utterances
    dialogs = defaultdict(list)
    for u in valid_utts:
        vid = u.rsplit("_", 1)[0]
        dialogs[vid].append(u)
    
    # Sort utterances in each dialog by time
    for vid in dialogs:
        dialogs[vid] = sorted(dialogs[vid], key=lambda x: annotations[x]['start'])

    # 3. Load Splits
    splits_def = load_splits_from_json(args.splits_file)
    
    # 4. Prepare Features
    pipeline = FeaturePipeline(use_opensmile=True, use_openface=True)
    
    # We need to collect ALL text embeddings first to fit PCA
    # But strictly, we should fit PCA only on TRAIN set.
    print("Collecting training texts for PCA...")
    train_texts = []
    train_vids = set(splits_def['train'])
    
    for vid in train_vids:
        if vid in dialogs:
            for u in dialogs[vid]:
                train_texts.append(transcripts[u])
    
    print(f"Encoding {len(train_texts)} training sentences for PCA fit...")
    if len(train_texts) > 0:
        train_sbert = pipeline.extract_text_sbert(train_texts)
        pipeline.train_text_pca(train_sbert, n_components=100)
    else:
        print("Warning: No training texts found. PCA will not be fitted.")

    # 5. Build Samples
    final_data = {'train': [], 'dev': [], 'test': []}
    
    # Process all splits
    for split_name in ['train', 'dev', 'test']:
        print(f"Processing split: {split_name}")
        target_vids = splits_def.get(split_name, [])
        
        for vid in tqdm(target_vids):
            if vid not in dialogs:
                continue
                
            utts = dialogs[vid]
            if not utts: continue
            
            # Prepare Lists
            utt_ids = []
            labels = []
            speakers = []
            sentences = []
            
            raw_sbert = [] 
            audio_feats = []
            visual_feats = []
            
            for u in utts:
                # Meta
                utt_ids.append(u)
                labels.append(annotations[u]['label'])
                speakers.append(annotations[u]['speaker'])
                txt = transcripts[u]
                sentences.append(txt)
                
                # Features
                # 1. SBERT (768)
                sb = pipeline.extract_text_sbert([txt])[0]
                raw_sbert.append(sb)
                
                # 2. Audio (100)
                sess_name = "Session" + vid[4] # Ses01 -> Session1
                wav_path = os.path.join(args.data_dir, sess_name, "sentences", "wav", vid, f"{u}.wav")
                aud = pipeline.extract_audio(wav_path)
                audio_feats.append(aud)
                
                # 3. Visual (512)
                vis = pipeline.extract_visual(vid, args.feat_dir)
                visual_feats.append(vis)

            # Convert to Arrays
            raw_sbert = np.array(raw_sbert, dtype=np.float32)
            text_100d = pipeline.get_text_100d(raw_sbert) # (N, 100)
            audio_feats = np.array(audio_feats, dtype=np.float32)
            visual_feats = np.array(visual_feats, dtype=np.float32)
            
            # Create Sample Object
            s = Sample(
                vid=vid,
                speaker=speakers,
                label=labels,
                text=text_100d,
                audio=audio_feats,
                visual=visual_feats,
                sentence=sentences
            )
            # Sample.__init__ computes sbert_sentence_embeddings automatically
            
            final_data[split_name].append(s)

    # 6. Save
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print(f"Saving to {args.out}...")
    with open(args.out, "wb") as f:
        pickle.dump(final_data, f)
    
    print("Done!")
    for split in final_data:
        print(f"  {split}: {len(final_data[split])} samples")

if __name__ == "__main__":
    main()
