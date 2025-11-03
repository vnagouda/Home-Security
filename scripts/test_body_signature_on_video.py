# scripts/test_body_signature_on_video.py
import os, sys, time, asyncio
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from telegram import Bot
import joblib
import torch
from torchvision import models, transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# ==============================================================================
# CONFIG
# ==============================================================================
DEBUG_MODE = 1
VIDEO_PATH = r"C:\Viresh\Projects\Web-Apps\HomeSecurity\data\people\Mom\videos\dist_6.mp4"
CONF_THRESHOLD = 0.5
FACE_CONF = 0.5
COOLDOWN_SECONDS = 3
SLEEP_BETWEEN_LOOPS = 0.05
PERSON_CLASS_ID = 0

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ROI_PATH = os.path.join(PROJECT_ROOT, "data", "roi_polygon.npy")
FACE_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "person_recognizer_face_Mom.pkl")
BODY_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "person_body_sig_Mom.pkl")
YOLO_FACE_MODEL = os.path.join(PROJECT_ROOT, "models", "yolo_face.pt")  # your trained YOLO-face

# Load .env for Telegram alerts
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

FACE_OK = 0.65
BODY_OK = 0.55
ALPHA = 0.6
SMOOTH_WINDOW = 5

# ==============================================================================
# UTILS
# ==============================================================================
def load_roi_polygon_raw():
    if os.path.exists(ROI_PATH):
        poly = np.load(ROI_PATH)
        return np.array(poly, dtype=np.int32)
    print("[WARN] No roi_polygon.npy found, using default ROI box.")
    return np.array([[200,100],[440,100],[440,380],[200,380]], dtype=np.int32)

def scale_polygon_to_frame(raw_polygon, w, h):
    max_x, max_y = np.max(raw_polygon[:,0]), np.max(raw_polygon[:,1])
    sx, sy = w/float(max_x), h/float(max_y)
    scaled = np.array([(int(px*sx), int(py*sy)) for px,py in raw_polygon], dtype=np.int32)
    print(f"[INFO] ROI scaled to {w}x{h}  scale=({sx:.3f},{sy:.3f})")
    return scaled

def point_in_polygon(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

# ==============================================================================
# TELEGRAM
# ==============================================================================
class TelegramAlerter:
    def __init__(self, token, chat_id):
        self.chat_id = chat_id
        self.bot = Bot(token=token)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    async def _send_async(self, text):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
            print("[TELEGRAM SENT]", text)
        except Exception as e:
            print("[TELEGRAM ERROR]", e)
    def send(self, text):
        try:
            self.loop.run_until_complete(self._send_async(text))
        except Exception as e:
            print("[TELEGRAM LOOP ERROR]", e)

# ==============================================================================
# LOAD MODELS
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

face_bundle = joblib.load(FACE_MODEL_PATH)
body_bundle = joblib.load(BODY_MODEL_PATH)
face_centroid = np.array(face_bundle["centroid"], dtype=np.float32).reshape(-1)
body_centroid = np.array(body_bundle["body_centroid"], dtype=np.float32).reshape(-1)
print(f"[INFO] Face centroid {face_centroid.shape}, Body centroid {body_centroid.shape}")

# Load models
face_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
weights = models.ResNet50_Weights.IMAGENET1K_V2
resnet = models.resnet50(weights=weights)
resnet.fc = torch.nn.Identity()
resnet.eval().to(device)
tfm_body = weights.transforms()

# YOLOs
yolo_person = YOLO("yolov8n.pt")
yolo_face = YOLO(YOLO_FACE_MODEL)

# ==============================================================================
# FEATURE HELPERS
# ==============================================================================
def cosim(a,b,eps=1e-9):
    a,b = a.reshape(-1),b.reshape(-1)
    return float(np.dot(a,b)/((np.linalg.norm(a)+eps)*(np.linalg.norm(b)+eps)))

def body_feat_from_pil(pil):
    x = tfm_body(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        f = resnet(x).cpu().numpy()[0]
    f /= (np.linalg.norm(f)+1e-9)
    return f.astype(np.float32)

def face_feat_from_frame(frame, box):
    """Detect face inside given person box using YOLO-face"""
    x1,y1,x2,y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    results = yolo_face.predict(crop, conf=FACE_CONF, verbose=False)
    if not results or not results[0].boxes: return None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    fx1,fy1,fx2,fy2 = boxes[np.argmax(areas)].astype(int)
    face_crop = crop[fy1:fy2, fx1:fx2]
    if face_crop.size == 0: return None
    pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        face_tensor = tfm_body.transforms()(pil_face) if hasattr(tfm_body, "transforms") else None
        # fallback simple conversion
        face_tensor = transforms.ToTensor()(pil_face).unsqueeze(0).to(device)
        emb = face_embedder(face_tensor).cpu().numpy()[0]
    emb /= (np.linalg.norm(emb)+1e-9)
    return emb.astype(np.float32)

def identify_person(face_emb, body_emb):
    s_face = cosim(face_emb, face_centroid) if face_emb is not None else None
    s_body = cosim(body_emb, body_centroid) if body_emb is not None else None
    if s_face is not None and s_body is not None:
        fused = ALPHA*s_face + (1-ALPHA)*s_body
        label = "Mom ✅" if fused >= (ALPHA*FACE_OK + (1-ALPHA)*BODY_OK) else "Unknown ❌"
        return label, dict(face=s_face, body=s_body, fused=fused)
    if s_face is not None:
        return ("Mom ✅" if s_face>=FACE_OK else "Unknown ❌"), dict(face=s_face)
    if s_body is not None:
        return ("Mom ✅" if s_body>=BODY_OK else "Unknown ❌"), dict(body=s_body)
    return "Unknown ❌", {}

# ==============================================================================
# STABILIZATION
# ==============================================================================
recent_labels = deque(maxlen=SMOOTH_WINDOW)
def stabilized_label(new_label):
    recent_labels.append(new_label)
    if recent_labels.count("Mom ✅") >= 3: return "Mom ✅"
    if recent_labels.count("Unknown ❌") >= 3: return "Unknown ❌"
    return recent_labels[-1]

# ==============================================================================
# DRAW + MAIN
# ==============================================================================
def annotate_frame(frame, detections, roi_polygon, fps, count):
    vis = frame.copy()
    cv2.polylines(vis, [roi_polygon], True, (0,255,0), 2)
    for x1,y1,x2,y2,label in detections:
        color = (0,255,0) if "Mom" in label else (0,0,255)
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
        cv2.putText(vis,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    hud = [f"Inside: {count}",f"FPS: {fps:.1f}","Press 'q' to quit"]
    for i,txt in enumerate(hud):
        cv2.putText(vis,txt,(10,25+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
    return vis

def main():
    alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) if TELEGRAM_BOT_TOKEN else None
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: cannot open video"); return
    w,h = int(cap.get(3)), int(cap.get(4))
    roi_scaled = scale_polygon_to_frame(load_roi_polygon_raw(), w, h)
    prev_time = time.time(); last_alert = 0
    print("[INFO] Running recognition loop...")

    while True:
        ret,frame = cap.read()
        if not ret: break
        results = yolo_person.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        detections=[]; inside=0
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                if int(box.cls[0])!=PERSON_CLASS_ID: continue
                x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                cx,cy=(x1+x2)//2,(y1+y2)//2
                if not point_in_polygon((cx,cy),roi_scaled): continue
                inside+=1
                body_crop = Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
                body_emb = body_feat_from_pil(body_crop)
                face_emb = face_feat_from_frame(frame,(x1,y1,x2,y2))
                label,_ = identify_person(face_emb,body_emb)
                label = stabilized_label(label)
                detections.append((x1,y1,x2,y2,label))

                if time.time()-last_alert>COOLDOWN_SECONDS:
                    if "Mom" in label:
                        msg=f"👩 Mom detected at {datetime.now().strftime('%H:%M:%S')}"
                    else:
                        msg=f"⚠️ Unknown person detected at {datetime.now().strftime('%H:%M:%S')}"
                    if alerter: alerter.send(msg)
                    last_alert=time.time()

        now=time.time(); fps=1.0/(now-prev_time+1e-9); prev_time=now
        if DEBUG_MODE:
            vis=annotate_frame(frame,detections,roi_scaled,fps,inside)
            cv2.imshow("Mom Recognition (YOLO-Face)",vis)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        else:
            time.sleep(SLEEP_BETWEEN_LOOPS)

    cap.release(); cv2.destroyAllWindows()
    print("[INFO] Completed video test.")

if __name__=="__main__":
    main()
