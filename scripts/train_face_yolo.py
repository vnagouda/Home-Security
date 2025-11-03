from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "face_yolo"
YAML = BASE / "dataset.yaml"
OUT  = PROJECT_ROOT / "models" / "yolo_face.pt"

def main():
    model = YOLO("yolov8n.pt")          # start from COCO-pretrained
    results = model.train(
        data=str(YAML),
        epochs=30,                       # small dataset → 20–40 is fine
        imgsz=640,
        batch=16,
        patience=10,
        project=str(PROJECT_ROOT / "runs"),
        name="face_yolo",
        verbose=True
    )
    # best weights:
    best = list((PROJECT_ROOT / "runs" / "face_yolo").glob("weights/best.pt"))[-1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(best.read_bytes())
    print(f"[OK] Saved: {OUT}")

if __name__ == "__main__":
    main()
