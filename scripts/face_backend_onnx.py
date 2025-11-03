# scripts/face_backend_onnx.py
import os, numpy as np, cv2, onnxruntime as ort

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")

# Accept any of these detectors:
DET_CANDIDATES = [
    os.path.join(MODELS_DIR, "det_10g.onnx"),
    os.path.join(MODELS_DIR, "det_2.5g.onnx"),
    os.path.join(MODELS_DIR, "det_500m.onnx"),
    os.path.join(MODELS_DIR, "scrfd_10g_bnkps.onnx"),
]

# ArcFace embedder (use glintr100 or glintr50)
ARCFACE = None
for name in ["glintr100.onnx", "glintr50.onnx"]:
    p = os.path.join(MODELS_DIR, name)
    if os.path.exists(p):
        ARCFACE = p
        break

def _providers():
    prov = ort.get_available_providers()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in prov else ["CPUExecutionProvider"]

def _exist_any(paths):
    for p in paths:
        if os.path.exists(p): return p
    return None

class OnnxFaceBackend:
    """Detector (RetinaFace/SCRFD) + ArcFace (glintrXX) via ONNX Runtime."""
    def __init__(self, det_thresh=0.45, nms_iou=0.45):
        det_path = _exist_any(DET_CANDIDATES)
        if det_path is None:
            raise FileNotFoundError("No detector found in data/models (det_10g.onnx / det_2.5g.onnx / det_500m.onnx / scrfd_10g_bnkps.onnx).")
        if ARCFACE is None:
            raise FileNotFoundError("No embedder found in data/models (glintr100.onnx or glintr50.onnx).")

        self.det_sess = ort.InferenceSession(det_path, providers=_providers())
        self.rec_sess = ort.InferenceSession(ARCFACE, providers=_providers())
        self.det_in = self.det_sess.get_inputs()[0].name
        self.rec_in = self.rec_sess.get_inputs()[0].name
        self.det_size = (640, 640)
        self.det_thresh = det_thresh
        self.nms_iou = nms_iou

    @staticmethod
    def _prep(img, size):
        h, w = img.shape[:2]
        scale = min(size[1]/h, size[0]/w)
        nh, nw = int(h*scale), int(w*scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        top = (size[1]-nh)//2; left = (size[0]-nw)//2
        canvas[top:top+nh, left:left+nw] = resized
        blob = canvas.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2,0,1)[None,...]
        meta = (scale, left, top, w, h)
        return blob, meta

    @staticmethod
    def _nms(dets, iou_thr):
        if len(dets)==0: return []
        x1,y1,x2,y2,s = dets[:,0],dets[:,1],dets[:,2],dets[:,3],dets[:,4]
        order = s.argsort()[::-1]
        keep=[]
        while order.size>0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2-xx1); h=np.maximum(0.0, yy2-yy1)
            inter = w*h
            area_i = (x2[i]-x1[i])*(y2[i]-y1[i])
            area_j = (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]])
            ovr = inter / (area_i + area_j - inter + 1e-9)
            inds = np.where(ovr<=iou_thr)[0]
            order = order[inds+1]
        return keep

    def _post_det(self, out, meta):
        """Handle common RetinaFace/SCRFD outputs. Expect boxes [N,5], kps [N,5,2]."""
        boxes = None; kps = None
        for a in out:
            if a.ndim==2 and a.shape[1]==5:
                boxes = a
            elif a.ndim==2 and a.shape[1]==10:
                kps = a.reshape(-1,5,2)
            elif a.ndim==3 and a.shape[2]==15:
                arr = a.reshape(-1,15)
                boxes = arr[:,:5]; kps = arr[:,5:].reshape(-1,5,2)
        if boxes is None:
            return np.zeros((0,5),dtype=np.float32), np.zeros((0,5,2),dtype=np.float32)

        scale,left,top,w0,h0 = meta
        b = boxes.copy()
        b[:,0] = (b[:,0]-left)/scale; b[:,1] = (b[:,1]-top)/scale
        b[:,2] = (b[:,2]-left)/scale; b[:,3] = (b[:,3]-top)/scale
        m = b[:,4] >= self.det_thresh
        b = b[m]
        if kps is not None: kps = kps[m]
        if len(b):
            keep = self._nms(b, self.nms_iou)
            b = b[keep]
            if kps is not None: kps = kps[keep]
        if kps is None:
            kps = np.zeros((len(b),5,2), dtype=np.float32)
        return b, kps

    def detect(self, img_bgr):
        blob, meta = self._prep(img_bgr, self.det_size)
        out = self.det_sess.run(None, {self.det_in: blob})
        boxes, kps = self._post_det(out, meta)
        return boxes, kps

    @staticmethod
    def _align_112(img_bgr, kps):
        # 5-point ArcFace template
        tmpl = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        if kps.sum()==0:  # no landmarks -> center crop fallback
            h,w = img_bgr.shape[:2]
            sz = min(h,w)
            y0 = (h-sz)//2; x0 = (w-sz)//2
            crop = img_bgr[y0:y0+sz, x0:x0+sz]
            return cv2.resize(crop, (112,112))
        src = kps.astype(np.float32)
        dst = tmpl
        src_mean = src.mean(axis=0); dst_mean = dst.mean(axis=0)
        src_c = src - src_mean; dst_c = dst - dst_mean
        U,S,Vt = np.linalg.svd(src_c.T @ dst_c)
        R = (U @ Vt).T
        s = (S.sum() / (src_c**2).sum())
        M = np.zeros((2,3), np.float32); M[:2,:2] = s*R; M[:,2] = dst_mean - (M[:2,:2] @ src_mean)
        return cv2.warpAffine(img_bgr, M, (112,112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def embed_faces_from_bgr(self, img_bgr):
        boxes, kps = self.detect(img_bgr)
        if len(boxes)==0:
            return np.zeros((0,512), np.float32), [], []
        embs=[]; rects=[]
        for i in range(len(boxes)):
            face112 = self._align_112(img_bgr, kps[i])
            rgb = cv2.cvtColor(face112, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = (rgb - 127.5)/128.0
            nchw = np.transpose(rgb, (2,0,1))[None,...]
            feat = self.rec_sess.run(None, {self.rec_in: nchw})[0][0].astype(np.float32)
            feat /= (np.linalg.norm(feat)+1e-9)
            embs.append(feat); rects.append(tuple(map(int, boxes[i][:4])))
        return np.vstack(embs), rects, boxes[:,4]
