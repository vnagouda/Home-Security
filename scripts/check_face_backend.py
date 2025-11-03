# scripts/check_face_backend.py
import os, glob, cv2
from _face_runtime import make_face_app

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR   = os.path.join(PROJECT_ROOT, "data", "people", "Mom", "frames_dist1")  # pick the closest set

def main():
    app = make_face_app(det_thresh=0.20, nms_iou=0.50)  # extra sensitive for debug
    imgs = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))[:20]
    if not imgs:
        print("No frames found in:", SAMPLE_DIR); return

    for fp in imgs:
        img = cv2.imread(fp)
        faces = app.get(img)
        print(f"{os.path.basename(fp)} -> faces: {len(faces)}")
        for f in faces:
            x1,y1,x2,y2 = map(int, f.bbox)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        # fit-to-screen
        H,W = img.shape[:2]
        scale = min(1280/W, 720/H, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(W*scale), int(H*scale)))
        cv2.imshow("face backend debug", img)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
