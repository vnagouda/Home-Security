# scripts/face_label_gui.py (safe autosave)
import os, cv2, glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "face_yolo"
IM_DIR   = DATA_DIR / "images"
LBL_DIR  = DATA_DIR / "labels"

os.makedirs(LBL_DIR, exist_ok=True)

images = sorted(glob.glob(str(IM_DIR / "*.jpg")))
idx = 0
drawing = False
x0=y0=x1=y1=0
box = None          # (x0,y0,x1,y1) in pixels
dirty = False       # needs save?

HELP = "[Enter/Space]=save+next  [n]=next  [p]=prev  [r]=remove  [q]=save+quit"

def clamp_box(b, w, h):
    if b is None: return None
    x0,y0,x1,y1 = b
    x0 = max(0, min(w-1, x0))
    y0 = max(0, min(h-1, y0))
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    if x1-x0 < 2 or y1-y0 < 2:
        return None
    return (x0,y0,x1,y1)

def write_yolo_label(img_path, b):
    """Save YOLO txt (class 0) or blank file if no box."""
    p = Path(img_path)
    lbl_path = LBL_DIR / (p.stem + ".txt")
    if b is None:
        open(lbl_path, "w").close()
        return
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    b = clamp_box(b, w, h)
    if b is None:
        open(lbl_path, "w").close()
        return
    x0,y0,x1,y1 = b
    cx = (x0+x1)/2 / w
    cy = (y0+y1)/2 / h
    bw = (x1-x0) / w
    bh = (y1-y0) / h
    with open(lbl_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def read_yolo_label(img_path):
    p = Path(img_path)
    lbl_path = LBL_DIR / (p.stem + ".txt")
    if not lbl_path.exists(): return None
    s = open(lbl_path).read().strip()
    if not s: return None
    parts = s.split()
    if len(parts) != 5: return None
    _, cx, cy, bw, bh = map(float, parts)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    x0 = int((cx - bw/2) * w)
    y0 = int((cy - bh/2) * h)
    x1 = int((cx + bw/2) * w)
    y1 = int((cy + bh/2) * h)
    return clamp_box((x0,y0,x1,y1), w, h)

def save_current():
    global dirty
    if not images: return
    if dirty:
        write_yolo_label(images[idx], box)
        dirty = False

def load_current():
    global box, dirty
    box = read_yolo_label(images[idx])
    dirty = False

def on_mouse(event, x, y, flags, param):
    global drawing, x0,y0,x1,y1,box,dirty
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; x0,y0 = x,y; x1,y1 = x,y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x1,y1 = x,y; dirty = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x1-x0) > 5 and abs(y1-y0) > 5:
            box = (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
            dirty = True

def draw_window():
    img = cv2.imread(images[idx])
    disp = img.copy()
    if box:
        cv2.rectangle(disp, (box[0],box[1]), (box[2],box[3]), (0,255,0), 2)
    cv2.putText(disp, f"{idx+1}/{len(images)}  {HELP}",
                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    if dirty:
        cv2.putText(disp, "unsaved*", (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.imshow("Face Label (autosaves)", disp)

def main():
    global idx, box, dirty

    import sys
    if len(sys.argv) > 1:
        IM_DIR = Path(sys.argv[1])

    if not images:
        print(f"No images found in {IM_DIR}")
        return

    cv2.namedWindow("Face Label (autosaves)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Face Label (autosaves)", on_mouse)
    load_current()

    while True:
        draw_window()
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):               # q or ESC: save and quit
            save_current()
            break

        elif key in (ord('n'), ord(' '), 13):   # n / Space / Enter: save+next
            save_current()
            if idx < len(images)-1:
                idx += 1
                load_current()
            else:
                # stay on last; brief toast by blinking title bar
                cv2.setWindowTitle("Face Label (autosaves)", "Face Label (end of list)")
                cv2.waitKey(300)
                cv2.setWindowTitle("Face Label (autosaves)", "Face Label (autosaves)")

        elif key == ord('p'):                   # prev
            save_current()
            if idx > 0:
                idx -= 1
                load_current()

        elif key == ord('r'):                   # remove box
            box = None; dirty = True

    cv2.destroyAllWindows()
    print(f"[OK] Labels saved to: {LBL_DIR}")

if __name__ == "__main__":
    main()
