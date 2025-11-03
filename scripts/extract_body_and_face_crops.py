# scripts/extract_face_body_pairs.py
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
PERSON_MODEL = Path(r"C:\Viresh\Projects\Web-Apps\HomeSecurity\models\yolov8n.pt")
FACE_MODEL   = Path(r"C:\Viresh\Projects\Web-Apps\HomeSecurity\models\yolo_face.pt")

SOURCE_DIR   = Path(r"C:\Viresh\Projects\Web-Apps\HomeSecurity\data\people\Mom")
OUT_FACE_DIR = SOURCE_DIR / "final_faces"
OUT_BODY_DIR = SOURCE_DIR / "final_bodies"

CONF_PERSON = 0.35
CONF_FACE   = 0.25
IOU_THRESH  = 0.5
MIN_SIZE    = 32  # skip tiny detections
# ----------------------------------------

os.makedirs(OUT_FACE_DIR, exist_ok=True)
os.makedirs(OUT_BODY_DIR, exist_ok=True)

yolo_person = YOLO(PERSON_MODEL)
yolo_face   = YOLO(FACE_MODEL)

def intersect(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return (x2 - x1) > 1 and (y2 - y1) > 1, [x1, y1, x2, y2]

def clip_box(x1, y1, x2, y2, w, h):
    return [max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2))]

# Process all frame folders
frame_folders = sorted([f for f in SOURCE_DIR.glob("frames_dist*") if f.is_dir()])

for fldr in frame_folders:
    frames = sorted(list(fldr.glob("*.jpg")))
    print(f"[INFO] Processing {fldr.name} ({len(frames)} frames)")

    for i, img_path in enumerate(frames):
        img = cv2.imread(str(img_path))
        if img is None: continue
        H, W = img.shape[:2]

        # Detect persons
        res_p = yolo_person.predict(img, conf=CONF_PERSON, iou=IOU_THRESH, classes=[0], verbose=False)[0]
        person_boxes = []
        if res_p.boxes is not None:
            for box in res_p.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) > MIN_SIZE and (y2 - y1) > MIN_SIZE:
                    person_boxes.append([x1, y1, x2, y2])

        # Detect faces
        res_f = yolo_face.predict(img, conf=CONF_FACE, iou=IOU_THRESH, classes=[0], verbose=False)[0]
        face_boxes = []
        if res_f.boxes is not None:
            for box in res_f.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) > 12 and (y2 - y1) > 12:
                    face_boxes.append([x1, y1, x2, y2])

        # Pair face + body (only if both exist)
        for (px1, py1, px2, py2) in person_boxes:
            best_face = None
            max_overlap = 0

            for (fx1, fy1, fx2, fy2) in face_boxes:
                inter, inter_box = intersect([px1, py1, px2, py2], [fx1, fy1, fx2, fy2])
                if inter:
                    overlap_area = (inter_box[2] - inter_box[0]) * (inter_box[3] - inter_box[1])
                    if overlap_area > max_overlap:
                        max_overlap = overlap_area
                        best_face = [fx1, fy1, fx2, fy2]

            # if matching face found
            if best_face is not None:
                fx1, fy1, fx2, fy2 = best_face

                # FACE crop
                face_crop = img[fy1:fy2, fx1:fx2]
                if face_crop.size == 0:
                    continue

                # BODY crop = person - face overlap
                by1 = fy2  # start just below the face
                body_crop = img[by1:py2, px1:px2]
                if body_crop.size == 0:
                    continue

                # save both
                basename = f"{fldr.name}_{i:05d}"
                face_out = OUT_FACE_DIR / f"{basename}_face.jpg"
                body_out = OUT_BODY_DIR / f"{basename}_body.jpg"

                cv2.imwrite(str(face_out), face_crop)
                cv2.imwrite(str(body_out), body_crop)

    print(f"[DONE] {fldr.name}: saved to final_faces & final_bodies")

print("\n✅ All done!")
print(f"Faces → {OUT_FACE_DIR}")
print(f"Bodies → {OUT_BODY_DIR}")
