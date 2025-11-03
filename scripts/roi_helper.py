import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

REFERENCE_IMAGE_PATH = r"C:\Viresh\Projects\Web-Apps\HomeSecurity\data\roi\WIN_20251101_11_45_48_Pro.jpg"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROI_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "roi_polygon.npy")

# ---------------------------------------------------------
# Globals for mouse callback
# ---------------------------------------------------------
points = []
window_name = "ROI Setup"

# These will store scaling ratio for later back-conversion
scale_x, scale_y = 1.0, 1.0


def draw_preview(img, pts):
    preview = img.copy()

    # Draw existing points
    for p in pts:
        cv2.circle(preview, p, 5, (0, 255, 0), -1)

    # Draw polygon lines
    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.line(preview, pts[i], pts[i + 1], (0, 255, 0), 2)
        if len(pts) > 2:
            cv2.line(preview, pts[-1], pts[0], (0, 255, 0), 2)

    # HUD
    y0 = 20
    msg_lines = [
        "Left click = add point",
        "u = undo last point",
        "s = save polygon",
        "q = quit (no save)"
    ]
    for i, line in enumerate(msg_lines):
        y = y0 + i * 20
        cv2.putText(preview, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return preview


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"[ADD] ({x}, {y})")


def main():
    global points, scale_x, scale_y

    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"ERROR: reference image not found at {REFERENCE_IMAGE_PATH}")
        return

    img_orig = cv2.imread(REFERENCE_IMAGE_PATH)
    if img_orig is None:
        print("ERROR: failed to load image")
        return

    h, w = img_orig.shape[:2]

    # --- compute resize factor so image fits screen ---
    max_width = 1080
    max_height = 720
    scale = min(max_width / w, max_height / h, 1.0)  # never upscale

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img_orig, (new_w, new_h))
        scale_x = w / new_w
        scale_y = h / new_h
        print(f"[INFO] Image resized for display: {new_w}x{new_h} (scale_x={scale_x:.2f}, scale_y={scale_y:.2f})")
    else:
        img = img_orig.copy()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("[INFO] ROI helper running.")
    print("Left click to add polygon points in order around YOUR PRIVATE ZONE.")
    print("Press 'u' to undo last point, 's' to save, 'q' to quit without saving.")

    while True:
        preview = draw_preview(img, points)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('u'):
            if points:
                removed = points.pop()
                print(f"[UNDO] removed {removed}")

        elif key == ord('s'):
            if len(points) < 3:
                print("[WARN] Need at least 3 points to form a polygon.")
                continue

            # rescale back to original coordinate system
            poly_scaled = np.array(
                [(int(x * scale_x), int(y * scale_y)) for (x, y) in points],
                dtype=np.int32
            )

            os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
            np.save(ROI_SAVE_PATH, poly_scaled)
            print(f"[SAVE] ROI polygon saved to {ROI_SAVE_PATH}")
            print("Points (original image coords):", poly_scaled.tolist())
            break

        elif key == ord('q'):
            print("[EXIT] quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
