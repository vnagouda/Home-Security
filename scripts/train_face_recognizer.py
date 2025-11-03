import os, glob, joblib, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from facenet_pytorch import InceptionResnetV1, MTCNN

# ---------------- CONFIG ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON = "Mom"
FACES_DIR = os.path.join(PROJECT_ROOT, "data", "people", PERSON, "final_faces")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "models")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------- MODELS ----------------
mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    keep_all=False, post_process=True, device=device
)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---------------- LOAD IMAGES ----------------
img_paths = sorted(glob.glob(os.path.join(FACES_DIR, "*.jpg")))
if not img_paths:
    raise SystemExit(f"[ERR] No images found in {FACES_DIR}")

print(f"[INFO] Found {len(img_paths)} face images")

# ---------------- EMBEDDING EXTRACTION ----------------
emb_list, labels = [], []
for path in tqdm(img_paths, desc="Embedding faces"):
    try:
        img = Image.open(path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            print(f"[WARN] No face detected in: {os.path.basename(path)}")
            continue
        with torch.no_grad():
            emb = embedder(face.unsqueeze(0).to(device)).cpu().numpy()[0]
        emb /= (np.linalg.norm(emb) + 1e-9)  # Normalize (L2)
        emb_list.append(emb)
        labels.append(PERSON)
    except Exception as e:
        print(f"[ERR] Failed on {path}: {e}")

if not emb_list:
    raise SystemExit("[ERR] No valid embeddings created.")

X = np.array(emb_list, dtype=np.float32)
y_text = np.array(labels)
print(f"[INFO] Embeddings shape: {X.shape}, samples kept: {len(X)}")

# ---------------- TRAIN CLASSIFIER ----------------
if len(np.unique(y_text)) < 1:
    raise SystemExit("[ERR] Need at least 1 unique class label for training.")

le = LabelEncoder()
y = le.fit_transform(y_text)

# For a single person, SVM can't classify → skip fit, store centroid only
if len(np.unique(y)) == 1:
    print("[WARN] Only one class found, skipping SVM fit. Using embedding centroid for recognition.")
    clf = None
else:
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y)

# ---------------- SAVE MODEL ----------------
centroid = X.mean(axis=0, keepdims=True)
bundle = dict(model=clf, le=le, centroid=centroid, person=PERSON)
out_path = os.path.join(OUT_DIR, f"person_recognizer_face_{PERSON}.pkl")
joblib.dump(bundle, out_path)
print(f"[OK] Saved recognizer → {out_path}")
