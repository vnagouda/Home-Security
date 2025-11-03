# scripts/preview_face_detections.py
import os
from pathlib import Path
import cv2
from ultralytics import YOLO

# ---- CONFIG ----
MODEL_PATH = Path(r"C:\Viresh\Projects\Web-Apps\HomeSecurity\models\yolo_face.pt")
INPUT_DIR  = Path(r"C:\Viresh\Projects\Web-Apps\HomeSecurity\data\people\Mom\frames_dist6")  # <-- test any dist folder
OUTPUT_DIR = Path(r"C:\Viresh\Projects\Web-Apps\HomeSecurity\runs\face_preview")
CONF_THRES = 0.25
IOU_THRES  = 0.5
# -----------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

img_files = sorted([f for f in INPUT_DIR.glob("*.jpg")] + [f for f in INPUT_DIR.glob("*.png")])
if not img_files:
    print(f"[ERROR] No images found in {INPUT_DIR}")
    exit()

print(f"[INFO] Running detection on {len(img_files)} frames...")

for i, img_path in enumerate(img_files, 1):
    results = model.predict(
        source=str(img_path),
        conf=CONF_THRES,
        iou=IOU_THRES,
        classes=[0],
        verbose=False
    )[0]

    img = cv2.imread(str(img_path))
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"face {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        cv2.putText(img, "NO FACE DETECTED", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    out_path = OUTPUT_DIR / f"{img_path.stem}_preview.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"[{i}/{len(img_files)}] Saved {out_path}")

print(f"\n✅ All done! Check results at: {OUTPUT_DIR}")
