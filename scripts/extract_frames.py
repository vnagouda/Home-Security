import os
import cv2

# -------- settings --------
PERSON = "Mom"
VIDEO_NAMES = [f"dist_{i}.mp4" for i in range(1, 7)]
TARGET_FPS = 5  # how many frames per second to save
MAX_FRAMES_PER_VIDEO = None  # or set an int to cap saved frames
# --------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PEOPLE_DIR = os.path.join(DATA_DIR, "people")
PERSON_DIR = os.path.join(PEOPLE_DIR, PERSON)

PREFERRED_VID_DIR = os.path.join(PERSON_DIR, "videos")        # data\people\Mom\videos
FALLBACK_VID_DIR  = os.path.join(PROJECT_ROOT, "bodies", PERSON)  # bodies\Mom

def ensure(d): os.makedirs(d, exist_ok=True)

def extract_frames(video_path: str, out_dir: str, target_fps: float, max_frames=None):
    ensure(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path}")
        return 0

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 1e-3:
        native_fps = 30.0  # fallback guess

    # save every Nth frame to approximate target_fps
    frame_interval = max(1, int(round(native_fps / target_fps)))

    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # save every Nth frame
        if idx % frame_interval == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1

    cap.release()
    print(f"[OK] {os.path.basename(video_path)} → {saved} frames to {out_dir}")
    return saved

def main():
    # choose video directory
    if os.path.isdir(PREFERRED_VID_DIR):
        vid_dir = PREFERRED_VID_DIR
    elif os.path.isdir(FALLBACK_VID_DIR):
        vid_dir = FALLBACK_VID_DIR
    else:
        print(f"[ERROR] No video directory found.\n"
              f"Put your videos in:\n  {PREFERRED_VID_DIR}\n  or\n  {FALLBACK_VID_DIR}")
        return

    for i, vname in enumerate(VIDEO_NAMES, start=1):
        vpath = os.path.join(vid_dir, vname)
        out_dir = os.path.join(PERSON_DIR, f"frames_dist{i}")
        extract_frames(vpath, out_dir, TARGET_FPS, MAX_FRAMES_PER_VIDEO)

if __name__ == "__main__":
    main()
