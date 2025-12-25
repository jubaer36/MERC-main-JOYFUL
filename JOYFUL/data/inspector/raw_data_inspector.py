import os
import argparse
from pathlib import Path

def inspect_iemocap(root_dir):
    """
    Traverses the IEMOCAP dataset folder structure and prints a comprehensive summary.
    This script identifies the key components of the dataset such as emotion labels,
    transcriptions, and audio files, explaining what each represents.
    """
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: Directory {root} does not exist.")
        return

    print(f"\n============================================================")
    print(f"             IEMOCAP RAW DATA INSPECTOR")
    print(f"============================================================")
    print(f"Root Directory: {root}\n")

    print(f"STRUCTURAL OVERVIEW")
    print(f"-------------------")
    print(f"The IEMOCAP dataset is typically organized into 5 Sessions (Session1 - Session5).")
    print(f"Each Session represents a set of dyadic interactions between two actors.")
    print(f"Inside each Session folder, data is split into:")
    print(f"  1. dialog/    -> Contains continuous data (full session recordings, labels, transcripts).")
    print(f"  2. sentences/ -> Contains segmented data (cut per utterance/sentence).")
    print(f"============================================================\n")
    
    sessions = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("Session")])
    
    total_sessions = len(sessions)
    print(f"Scanning {total_sessions} Sessions: {[s.name for s in sessions]}\n")
    
    global_total_wavs = 0
    global_total_transcripts = 0

    for session in sessions:
        print(f"------------------------------------------------------------")
        print(f"ANALYZING: {session.name}")
        print(f"------------------------------------------------------------")
        
        # --- PATHS ---
        dialog_dir = session / "dialog"
        sentences_dir = session / "sentences"
        
        # --- 1. EMOTION LABELS (EmoEvaluation) ---
        # Description: These files contain the categorical emotion labels (e.g., happiness, anger)
        # and dimensional labels (valence, activation, dominance) for each utterance.
        emo_dir = dialog_dir / "EmoEvaluation"
        emo_files = []
        if emo_dir.exists():
            emo_files = [f for f in emo_dir.glob("*.txt") if not f.name.startswith("._")]
        
        print(f"1. [LABELS] Folder: {emo_dir.relative_to(root)}")
        print(f"   - Contains: Categorical and attribute-based emotion evaluations.")
        print(f"   - File Count: {len(emo_files)} annotation files (usually 1 per conversation).")
        print(f"   - Example Content: '[6.2900 - 8.2350] Ses01F_impro01_F000 neu ...'")

        # --- 2. TRANSCRIPTIONS ---
        # Description: Manual text transcriptions of what was said.
        trans_dir = dialog_dir / "transcriptions"
        trans_files = []
        if trans_dir.exists():
            trans_files = [f for f in trans_dir.glob("*.txt") if not f.name.startswith("._")]
            global_total_transcripts += len(trans_files)
            
        print(f"\n2. [TRANSCRIPTS] Folder: {trans_dir.relative_to(root)}")
        print(f"   - Contains: Verbatim text transcripts of the dialogs.")
        print(f"   - File Count: {len(trans_files)} transcription files.")
        
        # --- 3. SEGMENTED AUDIO (sentences/wav) ---
        # Description: Audio files cut individually for each turn/utterance.
        # This is what is typically used for training "utterance-level" emotion recognition.
        wav_dir = sentences_dir / "wav"
        wav_subfolders = []
        session_wav_count = 0
        if wav_dir.exists():
            # Usually stricture is sentences/wav/Ses01F_impro01/*.wav
            wav_subfolders = [d for d in wav_dir.iterdir() if d.is_dir() and not d.name.startswith("._")]
            for imp_folder in wav_subfolders:
                wavs = list(imp_folder.glob("*.wav"))
                session_wav_count += len(wavs)
        
        global_total_wavs += session_wav_count
        
        print(f"\n3. [SEGMENTED AUDIO] Folder: {wav_dir.relative_to(root)}")
        print(f"   - Contains: Individual .wav files for each utterance.")
        print(f"   - Structure: Organized into {len(wav_subfolders)} subfolders (one per conversation).")
        print(f"   - Total Segments: {session_wav_count} audio clips in this session.")

        # --- 4. FULL AUDIO (dialog/wav) ---
        # Description: The full continuous recording of the session.
        full_wav_dir = dialog_dir / "wav"
        full_wavs = []
        if full_wav_dir.exists():
            full_wavs = [f for f in full_wav_dir.glob("*.wav") if not f.name.startswith("._")]
            
        print(f"\n4. [FULL SESSION AUDIO] Folder: {full_wav_dir.relative_to(root)}")
        print(f"   - Contains: Uninterrupted full-length recordings of the dialogs.")
        print(f"   - File Count: {len(full_wavs)} files.")

        print("\n")

    print(f"============================================================")
    print(f"FINAL SUMMARY")
    print(f"============================================================")
    print(f"Total Sessions Processed: {total_sessions}")
    print(f"Total Segmented Audio Clips (Utterances): {global_total_wavs}")
    print(f"Total Transcription Files: {global_total_transcripts}")
    print(f"Data location seems valid: {total_sessions == 5 and global_total_wavs > 0}")
    print(f"============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect IEMOCAP Raw Data Structure")
    parser.add_argument("--data_dir", type=str, 
                        default="/mnt/Work/ML/Github/MERC-main/JOYFUL/data/raw-data/IEMOCAP_full_release",
                        help="Path to the IEMOCAP_full_release directory")
    
    args = parser.parse_args()
    inspect_iemocap(args.data_dir)
