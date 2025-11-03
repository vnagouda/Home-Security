import os
import sys
import time
import asyncio
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from telegram import Bot


# ==============================================================================
# CONFIG
# ==============================================================================

# 0 = Raspberry Pi / headless (no GUI, silent loop, for production)
# 1 = Windows debugging (shows live window with boxes, ROI, count, FPS)
DEBUG_MODE = 1

WEBCAM_INDEX = 1            # which camera to open (you set this to 1)
CONF_THRESHOLD = 0.4        # YOLO confidence threshold for 'person'
COOLDOWN_SECONDS = 3        # min seconds between alerts when count changes
SLEEP_BETWEEN_LOOPS = 0.05  # pause each loop to control CPU
PERSON_CLASS_ID = 0         # 'person' class in COCO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ROI_PATH = os.path.join(PROJECT_ROOT, "data", "roi_polygon.npy")

# load .env for secrets
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ==============================================================================
# ROI LOAD + SCALE
# ==============================================================================

def load_roi_polygon_raw():
    """
    Load the raw polygon (in original reference image coordinates)
    from roi_polygon.npy.
    If missing, fall back to a default rect.
    Returns np.ndarray shape (N,2) int32.
    """
    if os.path.exists(ROI_PATH):
        poly = np.load(ROI_PATH)
        poly = np.array(poly, dtype=np.int32)
        print("[INFO] Loaded ROI polygon from file:", ROI_PATH)
        print("       Raw Points:", poly.tolist())
        return poly
    else:
        print("[WARN] No roi_polygon.npy found, using default box.")
        return np.array([
            [200, 100],
            [440, 100],
            [440, 380],
            [200, 380]
        ], dtype=np.int32)


def scale_polygon_to_frame(raw_polygon, frame_w, frame_h):
    """
    Scales the polygon (which was defined on the original reference frame)
    to match the CURRENT live frame size.

    How do we know scale?
    - We guess the reference "design resolution" from the polygon extents:
      max_x, max_y ~ how big the original frame was when polygon was saved.
    - Then we scale all polygon points from that original size to current size.

    This works because your roi_helper saved huge coords like x~2683,y~1512,
    which means your reference frame was around that scale.
    """
    # Get polygon bounds
    max_x = np.max(raw_polygon[:, 0])
    max_y = np.max(raw_polygon[:, 1])

    # Avoid divide-by-zero
    if max_x == 0 or max_y == 0:
        return raw_polygon.copy()

    # Compute scale factors
    sx = frame_w / float(max_x)
    sy = frame_h / float(max_y)

    # We'll keep aspect separately on x/y (not force same scale),
    # because camera aspect might differ slightly.
    scaled_pts = []
    for (px, py) in raw_polygon:
        new_x = int(px * sx)
        new_y = int(py * sy)
        scaled_pts.append((new_x, new_y))

    scaled_pts = np.array(scaled_pts, dtype=np.int32)

    print("[INFO] Scaling ROI polygon:")
    print(f"       frame_w={frame_w}, frame_h={frame_h}, max_x={max_x}, max_y={max_y}")
    print(f"       sx={sx:.3f}, sy={sy:.3f}")
    print("       Scaled Points:", scaled_pts.tolist())

    return scaled_pts


def point_in_polygon(point_xy, polygon_np):
    """
    point_xy: (x, y)
    polygon_np: Nx2 np.array polygon vertices
    Returns True if point is inside polygon.
    """
    return cv2.pointPolygonTest(polygon_np, point_xy, False) >= 0


class TelegramAlerter:
    """
    Persistent Bot + event loop for stable async send.
    """
    def __init__(self, token: str, chat_id: str):
        self.chat_id = chat_id
        self.bot = Bot(token=token)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def _send_async(self, text: str):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
            print("[TELEGRAM SENT]", text)
        except Exception as e:
            print("[TELEGRAM ERROR]", e)

    def send(self, text: str):
        try:
            self.loop.run_until_complete(self._send_async(text))
        except Exception as e:
            print("[TELEGRAM LOOP ERROR]", e)


def annotate_frame(frame, results_list, roi_polygon, person_class_id, fps, people_inside):
    """
    Draw ROI polygon, detections, HUD.
    """
    vis = frame.copy()

    # draw ROI polygon in green
    cv2.polylines(vis, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    # draw each detected 'person'
    for result in results_list:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != person_class_id:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cxi, cyi = int(cx), int(cy)

            inside = point_in_polygon((cx, cy), roi_polygon)

            color = (0, 0, 255) if inside else (255, 255, 255)
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.circle(vis, (cxi, cyi), 4, color, -1)

    # HUD info
    hud_lines = [
        f"Inside count: {people_inside}",
        f"FPS: {fps:.1f}",
        "Press 'q' to quit (debug mode)"
    ]
    y0 = 20
    for i, line in enumerate(hud_lines):
        y = y0 + i * 20
        cv2.putText(vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)

    return vis


def main():
    # --- sanity check env
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in .env")
        return

    # --- telegram alerter
    alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    # --- webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam index {WEBCAM_INDEX}")
        return

    # grab one frame first so we know resolution
    ok, test_frame = cap.read()
    if not ok:
        print("ERROR: couldn't read initial frame from camera")
        return

    frame_h, frame_w = test_frame.shape[:2]
    print(f"[INFO] Camera live resolution: {frame_w}x{frame_h}")

    # load + scale ROI polygon
    raw_roi = load_roi_polygon_raw()
    roi_polygon_scaled = scale_polygon_to_frame(raw_roi, frame_w, frame_h)

    # --- YOLO model
    print("[INFO] Loading YOLO model (yolov8n)...")
    model = YOLO("yolov8n.pt")

    last_reported_count = 0
    last_alert_time = 0

    prev_time = time.time()
    print("[INFO] Starting security loop. Ctrl+C or 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # run inference
        results_list = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        # count how many people are inside scaled ROI polygon
        people_inside = 0
        for result in results_list:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if point_in_polygon((cx, cy), roi_polygon_scaled):
                    people_inside += 1

        # alert logic
        now = time.time()
        if people_inside != last_reported_count:
            if now - last_alert_time >= COOLDOWN_SECONDS:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = (
                    f"Entry detected: {people_inside} person(s) "
                    f"inside boundary at {timestamp}"
                )
                alerter.send(msg)

                last_reported_count = people_inside
                last_alert_time = now

        # debug mode drawing
        if DEBUG_MODE == 1:
            now_time = time.time()
            dt = now_time - prev_time
            fps = (1.0 / dt) if dt > 0 else 0.0
            prev_time = now_time

            vis_frame = annotate_frame(
                frame,
                results_list,
                roi_polygon_scaled,
                PERSON_CLASS_ID,
                fps,
                people_inside
            )

            cv2.imshow("HomeSecurity Live", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(SLEEP_BETWEEN_LOOPS)
        else:
            time.sleep(SLEEP_BETWEEN_LOOPS)

    cap.release()
    if DEBUG_MODE == 1:
        cv2.destroyAllWindows()
    print("[INFO] Stopped cleanly.")


if __name__ == "__main__":
    main()
