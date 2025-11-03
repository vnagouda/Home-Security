# scripts/preview_extractor_once.py
import os, glob, cv2
from ultralytics import YOLO
from _face_runtime import make_face_app

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR   = os.path.join(PROJECT_ROOT, "data", "people", "Mom", "frames_dist1")

CONF_PERSON = 0.35
IOU_PERSON  = 0.45
HEAD_TOP_FRAC = 0.42

def main():
    yolo = YOLO("yolov8n.pt")
    app  = make_face_app(det_thresh=0.20, nms_iou=0.50)

    imgs = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))[:30]
    for fp in imgs:
        img = cv2.imread(fp); H,W = img.shape[:2]
        # persons
        ppl = []
        res = yolo.predict(img, conf=CONF_PERSON, iou=IOU_PERSON, classes=[0], verbose=False)
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                ppl.append((x1,y1,x2,y2))
        # faces
        faces = app.get(img)
        # draw
        vis = img.copy()
        for (x1,y1,x2,y2) in ppl:
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 2)
            hy2 = int(y1 + HEAD_TOP_FRAC*(y2-y1))
            cv2.rectangle(vis, (x1,y1), (x2,hy2), (0,255,255), 2)  # head_any region
        for f in faces:
            x1,y1,x2,y2 = map(int, f.bbox)
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        # fit-to-screen
        scale = min(1280/W, 720/H, 1.0)
        if scale < 1.0:
            vis = cv2.resize(vis, (int(W*scale), int(H*scale)))
        cv2.imshow("preview (white=person, yellow=head_any, green=face)", vis)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
