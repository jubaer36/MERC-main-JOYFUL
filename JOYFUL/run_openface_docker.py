
import os
import subprocess
import glob
from tqdm import tqdm

def run_openface():
    base_dir = "/mnt/Work/ML/Github/MERC-main/JOYFUL"
    data_dir = os.path.join(base_dir, "data/raw-data/IEMOCAP_full_release")
    out_dir = os.path.join(base_dir, "data/features/openface")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Find all AVI files
    # Pattern: SessionX/dialog/avi/DivX/*.avi
    pattern = os.path.join(data_dir, "Session*", "dialog", "avi", "DivX", "*.avi")
    videos = glob.glob(pattern)
    videos = [v for v in videos if not os.path.basename(v).startswith("._")]
    # videos = videos[:1] # Test with 1 video
    
    print(f"Found {len(videos)} videos to process.")
    
    # We mount the base_dir to /ws in the container
    # So a file at /mnt/.../JOYFUL/data/... becomes /ws/data/...
    
    for vid_path in tqdm(videos):
        rel_path = os.path.relpath(vid_path, base_dir)
        container_in_path = f"/ws/{rel_path}"
        container_out_path = "/ws/data/features/openface"
        
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        expected_csv = os.path.join(out_dir, f"{vid_name}.csv")
        
        if os.path.exists(expected_csv):
            print(f"Skipping {vid_name}, already exists.")
            continue

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{base_dir}:/ws",
            "--entrypoint", "/home/openface-build/build/bin/FeatureExtraction",
            "algebr/openface:latest",
            "-f", container_in_path,
            "-out_dir", container_out_path,
            "-of", vid_name,
            "-2Dfp", "-3Dfp", "-pdnu", "-pose", "-gaze", "-aus"
        ]
        
        print(f"\nProcessing {vid_name}...")
        print("Command:", " ".join(cmd))
        
        # Stream output directly to console
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {vid_name}: {e}")

if __name__ == "__main__":
    run_openface()
