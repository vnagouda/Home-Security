import os, glob, shutil, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PEOPLE_DIR   = PROJECT_ROOT / "data" / "people" / "Mom"
OUT_DIR      = PROJECT_ROOT / "data" / "face_yolo"
IM_DIR       = OUT_DIR / "images"

PER_FOLDER_SAMPLES = 15   # tweak

def main():
    os.makedirs(IM_DIR, exist_ok=True)
    picked = []
    for fldr in sorted([d for d in os.listdir(PEOPLE_DIR) if d.startswith("frames_dist")]):
        fpath = PEOPLE_DIR / fldr
        frames = sorted(glob.glob(str(fpath / "*.jpg")))
        if not frames: continue
        k = min(PER_FOLDER_SAMPLES, len(frames))
        chosen = random.sample(frames, k)
        picked += chosen

    # copy into images/
    for src in picked:
        dst = IM_DIR / os.path.basename(src)
        shutil.copy2(src, dst)

    print(f"[OK] Copied {len(picked)} frames into {IM_DIR}")

if __name__ == "__main__":
    main()
