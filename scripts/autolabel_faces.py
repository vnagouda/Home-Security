# scripts/autolabel_from_frames.py
import os, sys, glob, shutil, argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

PROJECT = Path(__file__).resolve().parents[1]
DATA    = PROJECT / "data"
PEOPLE  = DATA / "people"
OUTROOT = DATA / "face_yolo"

def copy_image(src: Path, dst: Path, overwrite: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not dst.exists():
        shutil.copy2(src, dst)

def write_label(txt_path: Path, boxes):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        for (cx, cy, bw, bh) in boxes:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    # if boxes == [], file is empty (negative sample), still created → keeps parity

def autolabel_folder(model, src_folder: Path, dst_img_root: Path, dst_lbl_root: Path,
                     conf=0.25, iou=0.5, overwrite=False):
    imgs = sorted(glob.glob(str(src_folder / "*.jpg"))) + \
           sorted(glob.glob(str(src_folder / "*.png")))
    saved_imgs = saved_lbls = 0

    # mirror path fragment after person dir (e.g., frames_dist3)
    rel = src_folder.name

    for sp in imgs:
        sp = Path(sp)
        dst_img = dst_img_root / rel / sp.name
        dst_lbl = dst_lbl_root / rel / (sp.stem + ".txt")

        # 1) copy image (or skip)
        copy_image(sp, dst_img, overwrite)
        saved_imgs += 1

        # 2) run detector and write label (always create a file; empty=negative)
        r = model.predict(source=str(sp), conf=conf, iou=iou, classes=[0], verbose=False)[0]
        h, w = r.orig_img.shape[:2]
        boxes = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                boxes.append((cx, cy, bw, bh))
        write_label(dst_lbl, boxes)
        saved_lbls += 1

    return saved_imgs, saved_lbls

def main():
    ap = argparse.ArgumentParser(description="Auto-label faces from frames_distX folders with mirrored structure.")
    ap.add_argument("--person", default="Mom", help="Name under data/people/<PERSON>")
    ap.add_argument("--model", default=str(PROJECT / "models" / "yolo_face.pt"),
                    help="Path to trained face YOLO model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite copied images/labels")
    ap.add_argument("--review", action="store_true", help="Launch label GUI after autolabel")
    args = ap.parse_args()

    person_dir = PEOPLE / args.person
    if not person_dir.exists():
        print(f"ERROR: {person_dir} not found.")
        sys.exit(1)

    # find frames folders
    frame_folders = sorted([p for p in person_dir.iterdir()
                            if p.is_dir() and p.name.startswith("frames_dist")],
                           key=lambda p: int(''.join(c for c in p.name if c.isdigit()) or "0"))
    if not frame_folders:
        print(f"ERROR: No frames_distX folders under {person_dir}")
        sys.exit(1)

    # output mirrored roots
    dst_img_root = OUTROOT / "images" / args.person
    dst_lbl_root = OUTROOT / "labels" / args.person
    dst_img_root.mkdir(parents=True, exist_ok=True)
    dst_lbl_root.mkdir(parents=True, exist_ok=True)

    # load model
    model = YOLO(args.model)

    total_i = total_l = 0
    print(f"[INFO] Source person: {args.person}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Images root: {dst_img_root}")
    print(f"[INFO] Labels root: {dst_lbl_root}\n")

    for fldr in frame_folders:
        i_count, l_count = autolabel_folder(
            model, fldr, dst_img_root, dst_lbl_root,
            conf=args.conf, iou=args.iou, overwrite=args.overwrite
        )
        total_i += i_count; total_l += l_count
        print(f"[OK] {fldr.name}: images={i_count} labels={l_count}")

    print("\n=== SUMMARY ===")
    print(f"Total images copied: {total_i}")
    print(f"Total label files  : {total_l}")
    if total_i != total_l:
        print("⚠️ Mismatch! (should not happen; each image writes a label file)")
    else:
        print("✅ 1:1 parity ensured (every image has a label file).")

    if args.review:
        # Launch your existing GUI (it will read from OUTROOT/images by default),
        # or call with an argument if your GUI supports it.
        gui = PROJECT / "scripts" / "face_label_gui.py"
        if gui.exists():
            # pass a starting folder to keep context inside this person's tree
            try:
                import subprocess, sys as _sys
                subprocess.run([_sys.executable, str(gui), str(dst_img_root)], check=False)
            except Exception as e:
                print(f"[WARN] Could not launch GUI automatically: {e}")
        else:
            print("[NOTE] face_label_gui.py not found; open it manually if needed.")

if __name__ == "__main__":
    main()
