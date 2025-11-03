# scripts/train_body_signature.py  (you can also name it build_body_signature.py)
import os, glob, joblib, torch, numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchvision import models, transforms
from contextlib import nullcontext

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON      = "Mom"
BODIES_DIR  = os.path.join(PROJECT_ROOT, "data", "people", PERSON, "final_bodies")
OUT_DIR     = os.path.join(PROJECT_ROOT, "data", "models")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device = {device}")

# ---- backbone ----
weights = models.ResNet50_Weights.IMAGENET1K_V2
resnet  = models.resnet50(weights=weights)
resnet.fc = torch.nn.Identity()
resnet.eval().to(device)

# ---- preprocessing (robust across torchvision versions) ----
# Prefer the official transform recipe when available
try:
    tfm = weights.transforms()  # includes resize -> toTensor -> normalize
except Exception:
    # Fallback to standard ImageNet normalization
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

paths = sorted(glob.glob(os.path.join(BODIES_DIR, "*.jpg")))
if not paths:
    raise SystemExit(f"[ERR] no images in {BODIES_DIR}")

BATCH = 32
feats = []
good  = 0

def l2norm(v, eps=1e-9):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)

with torch.no_grad():
    for i in tqdm(range(0, len(paths), BATCH), desc="Body feats"):
        batch = []
        for p in paths[i:i+BATCH]:
            try:
                img = Image.open(p).convert("RGB")
            except (UnidentifiedImageError, OSError):
                continue
            batch.append(tfm(img))
        if not batch:
            continue

        x = torch.stack(batch).to(device)

        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with ctx:
            f = resnet(x)   # [N, 2048]

        f = f.float().cpu().numpy()
        f = l2norm(f)
        feats.append(f)
        good += f.shape[0]

if good == 0:
    raise SystemExit("[ERR] no valid crops produced features")

feats     = np.concatenate(feats, axis=0)      # [M, 2048]
centroid  = feats.mean(axis=0)
centroid  = centroid / (np.linalg.norm(centroid) + 1e-9)

bundle = dict(
    person=PERSON,
    body_centroid=centroid.astype(np.float32),
    dim=int(centroid.shape[0]),
    count=int(good),
    backbone="resnet50_imagenet_v2"
)

out_path = os.path.join(OUT_DIR, f"person_body_sig_{PERSON}.pkl")
joblib.dump(bundle, out_path)
print(f"[OK] saved -> {out_path} | dim={bundle['dim']} count={bundle['count']}")
