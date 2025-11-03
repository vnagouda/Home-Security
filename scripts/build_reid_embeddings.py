import os, glob, json, torch, timm, numpy as np, cv2
from config import PEOPLE_DIR, MODELS_DIR

device="cuda" if torch.cuda.is_available() else "cpu"
model=timm.create_model("resnet50", pretrained=True, num_classes=0).to(device).eval()

def get_vec(img):
    img=cv2.resize(img,(224,224))
    t=torch.tensor(img/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad(): v=model(t)
    v=v[0].detach().cpu().numpy()
    v=v/np.linalg.norm(v)
    return v

def build_body(person):
    person_root=os.path.join(PEOPLE_DIR, person)
    crops=[]
    for d in glob.glob(os.path.join(person_root,"body_crops_*")):
        crops += glob.glob(os.path.join(d,"*.jpg"))
    vecs=[]
    for p in crops:
        img=cv2.imread(p)
        if img is None: continue
        vecs.append(get_vec(img))
    if not vecs: return None
    mean=np.mean(np.stack(vecs),axis=0)
    mean/=np.linalg.norm(mean)
    return mean

if __name__=="__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    person="Mom"
    v=build_body(person)
    if v is None:
        print("No body images found.")
    else:
        np.save(os.path.join(MODELS_DIR,"embeddings_reid.npy"), np.array([v]))
        with open(os.path.join(MODELS_DIR,"names_reid.json"),"w") as f:
            json.dump([person],f)
        print("[OK] Body embedding saved → models/")
