# scripts/_face_runtime.py
import numpy as np
from collections import namedtuple
from face_backend_onnx import OnnxFaceBackend  # same folder import

FaceDet = namedtuple("FaceDet", ["bbox", "kps", "det_score"])

def make_face_app(det_thresh: float = 0.45, nms_iou: float = 0.45):
    backend = OnnxFaceBackend(det_thresh=det_thresh, nms_iou=nms_iou)

    class _App:
        def get(self, img_bgr):
            boxes, kps = backend.detect(img_bgr)  # boxes [N,5], kps [N,5,2]
            out=[]
            for i in range(len(boxes)):
                x1,y1,x2,y2,score = boxes[i]
                out.append(FaceDet(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    kps=np.asarray(kps[i]).astype(float),
                    det_score=float(score)
                ))
            return out

        def embed_faces_from_bgr(self, img_bgr):
            return backend.embed_faces_from_bgr(img_bgr)

    return _App()
