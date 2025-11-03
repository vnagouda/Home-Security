import os, joblib, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "data", "models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Face runtime ----
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20, keep_all=False, device=device)
face_emb = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_bundle = joblib.load(os.path.join(MODELS_DIR, "person_recognizer_face_Mom.pkl"))
face_clf = face_bundle["model"]
face_le  = face_bundle["le"]
face_centroid = face_bundle["centroid"]  # (1,512)

def face_predict(bgr):
    """Return ('Mom', prob) or ('Unknown', prob)"""
    rgb = Image.fromarray(bgr[:,:,::-1].copy())
    f = mtcnn(rgb)
    if f is None:
        return ("NoFace", 0.0)
    with torch.no_grad():
        e = face_emb(f.unsqueeze(0).to(device)).cpu().numpy()[0]
    e = e / (np.linalg.norm(e)+1e-9)
    # SVM prob
    prob = face_clf.predict_proba([e])[0].max()
    pred = face_le.inverse_transform([face_clf.predict([e])[0]])[0]
    # Optional extra: cosine to Mom centroid to guard against weird probs
    cos = float(np.dot(e, face_centroid[0]))
    if pred == "Mom" and cos > 0.45:  # tweakable safety margin
        return ("Mom", max(prob, cos))
    return ("Unknown", max(prob, cos))

# ---- Body runtime ----
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = torch.nn.Identity()
resnet.eval().to(device)
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=models.ResNet50_Weights.IMAGENET1K_V2.meta["mean"],
                         std =models.ResNet50_Weights.IMAGENET1K_V2.meta["std"]),
])

body_bundle = joblib.load(os.path.join(MODELS_DIR, "person_body_sig_Mom.pkl"))
body_centroid = body_bundle["body_centroid"]

def body_sim(bgr_crop):
    """Return cosine similarity to Mom body signature."""
    with torch.no_grad():
        rgb = Image.fromarray(bgr_crop[:,:,::-1].copy())
        x = tfm(rgb).unsqueeze(0).to(device)
        f = resnet(x).cpu().numpy()[0]
    f = f / (np.linalg.norm(f)+1e-9)
    return float(np.dot(f, body_centroid))
