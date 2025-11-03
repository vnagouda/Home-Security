import os, glob, shutil, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "face_yolo"
IM = BASE / "images"
LBL = BASE / "labels"

TRAIN_IM = BASE / "train" / "images"
TRAIN_LB = BASE / "train" / "labels"
VAL_IM   = BASE / "val" / "images"
VAL_LB   = BASE / "val" / "labels"

SPLIT = 0.85

def main():
    ims = sorted(glob.glob(str(IM / "*.jpg")))
    random.shuffle(ims)
    n_train = int(len(ims) * SPLIT)
    train, val = ims[:n_train], ims[n_train:]

    for d in [TRAIN_IM, TRAIN_LB, VAL_IM, VAL_LB]:
        os.makedirs(d, exist_ok=True)

    def move(set_list, dst_im, dst_lb):
        for p in set_list:
            name = os.path.basename(p)
            lbl  = LBL / (Path(p).stem + ".txt")
            shutil.copy2(p, dst_im / name)
            if lbl.exists(): shutil.copy2(lbl, dst_lb / (Path(p).stem + ".txt"))
            else: open(dst_lb / (Path(p).stem + ".txt"), "w").close()

    move(train, TRAIN_IM, TRAIN_LB)
    move(val,   VAL_IM,   VAL_LB)

    yaml_path = BASE / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(
f"""path: {BASE.as_posix()}
train: {('train/images')}
val:   {('val/images')}
nc: 1
names: [face]
""")
    print(f"[OK] Train:{len(train)}  Val:{len(val)}")
    print(f"[OK] Wrote YAML: {yaml_path}")

if __name__ == "__main__":
    main()
