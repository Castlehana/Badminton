import glob, os, subprocess
vids = glob.glob("dataset/raw_videos/*.mp4")
os.makedirs("dataset/landmarks", exist_ok=True)
for v in vids:
    base = os.path.splitext(os.path.basename(v))[0]
    out = f"dataset/landmarks/{base}.parquet"
    if not os.path.exists(out):
        subprocess.run(["python","scripts/02_extract_landmarks.py", v, out], check=True)
