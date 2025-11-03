import os, glob, cv2, random
from config import PEOPLE_DIR

def ensure(d): os.makedirs(d, exist_ok=True)

def augment(img):
    out=[]
    base=cv2.resize(img,(224,224))
    out.append(base)
    # simulate distance
    scale=random.uniform(0.3,0.6)
    small=cv2.resize(base,(int(224*scale),int(224*scale)))
    up=cv2.resize(small,(224,224))
    out.append(up)
    # blur
    out.append(cv2.GaussianBlur(up,(random.choice([3,5]),)*2,0))
    # brightness/contrast
    alpha=random.uniform(0.6,1.3); beta=random.randint(-30,30)
    out.append(cv2.convertScaleAbs(up,alpha=alpha,beta=beta))
    return out

def process_person(name):
    person_root=os.path.join(PEOPLE_DIR,name)
    raws = []
    for d in glob.glob(os.path.join(person_root,"face_raw_*")):
        raws += glob.glob(os.path.join(d,"*.jpg"))
    outdir=os.path.join(person_root,"face_final"); ensure(outdir)
    i=0
    for p in raws:
        img=cv2.imread(p)
        if img is None: continue
        for aug in augment(img):
            cv2.imwrite(os.path.join(outdir, f"{name}_{i:05d}.jpg"), aug)
            i+=1
    print(f"[DONE] {name}: {i} augmented faces in {outdir}")

if __name__=="__main__":
    process_person("Mom")
