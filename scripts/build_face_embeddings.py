import os, glob, json, numpy as np, face_recognition
from config import PEOPLE_DIR, MODELS_DIR

os.makedirs(MODELS_DIR, exist_ok=True)
OUT_PATH = os.path.join(MODELS_DIR, "embeddings_face.json")

def build(person):
    final_dir=os.path.join(PEOPLE_DIR, person, "face_final")
    imgs=glob.glob(os.path.join(final_dir,"*.jpg"))
    vecs=[]
    for p in imgs:
        img=face_recognition.load_image_file(p)
        locs=face_recognition.face_locations(img, model="hog")
        if not locs: continue
        encs=face_recognition.face_encodings(img, locs)
        if encs: vecs.append(encs[0])
    if not vecs: return None
    return np.mean(np.stack(vecs),axis=0).tolist()

if __name__=="__main__":
    person="Mom"
    emb=build(person)
    if emb is None:
        print("No usable faces found.")
    else:
        db={person: emb}
        with open(OUT_PATH,"w") as f: json.dump(db,f)
        print(f"[OK] {person} face embedding saved → {OUT_PATH}")
