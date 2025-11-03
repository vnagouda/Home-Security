import os, sys, requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# We need:
#  A) ONE detector file (any of these): det_10g.onnx | det_2.5g.onnx | det_500m.onnx | scrfd_10g_bnkps.onnx
#  B) ONE embedder file: glintr100.onnx (or glintr50.onnx)
CANDIDATES = {
    # --- DETECTORS (prefer GitHub release assets: no login required) ---
    "det_10g.onnx": [
        "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx",   # GitHub releases (direct)
        "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/det_10g.onnx",                  # HF mirror
        "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/insightface/models/buffalo_l/det_10g.onnx",
    ],
    "det_2.5g.onnx": [
        "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx",
    ],
    "det_500m.onnx": [
        "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx",
    ],
    # Optional SCRFD variant with landmarks (if one of the det_* fails, this also works)
    "scrfd_10g_bnkps.onnx": [
        "https://huggingface.co/typhoon01/aux_models/resolve/main/scrfd_10g_bnkps.onnx",
        "https://ai-apps.momodel.cn/repo/YK5L4iaR_NlRptdKNKHxHh0EaJMIGms5bfWvPBCSNg%3D%3D/raw/master/insightface_func/models/scrfd_10g_bnkps.onnx",
    ],

    # --- EMBEDDERS ---
    "glintr100.onnx": [
        "https://huggingface.co/LPDoctor/insightface/resolve/main/models/antelopev2/glintr100.onnx",
        "https://huggingface.co/deepinsight/insightface/resolve/main/models/antelopev2/glintr100.onnx",
    ],
    # smaller optional
    "glintr50.onnx": [
        "https://huggingface.co/garavv/arcface-onnx/resolve/main/glintr50.onnx",
    ],
}

def download(url, dest):
    print(f"[TRY] {url}")
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=262144):
                if not chunk: continue
                f.write(chunk); done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    sys.stdout.write(f"\r  -> {os.path.basename(dest)} {pct}%")
                    sys.stdout.flush()
    print("\r", end="")

def main():
    print(f"[INFO] Saving models to: {MODELS_DIR}\n")
    for fname, urls in CANDIDATES.items():
        dest = os.path.join(MODELS_DIR, fname)
        if os.path.exists(dest):
            print(f"[SKIP] {fname} already exists.")
            continue

        ok = False
        for url in urls:
            tmp = dest + ".part"
            try:
                download(url, tmp)
                os.replace(tmp, dest)
                print(f"[OK] {fname} downloaded.\n")
                ok = True
                break
            except Exception as e:
                print(f"[FAIL] {url} -> {e}")
                try:
                    if os.path.exists(tmp): os.remove(tmp)
                except: pass
        if not ok:
            print(f"[ERROR] Could not download {fname} from any mirror.\n")

    # Check presence: at least one detector + one embedder
    det_found = any(os.path.exists(os.path.join(MODELS_DIR, f))
                    for f in ["det_10g.onnx", "det_2.5g.onnx", "det_500m.onnx", "scrfd_10g_bnkps.onnx"])
    emb_found = any(os.path.exists(os.path.join(MODELS_DIR, f))
                    for f in ["glintr100.onnx", "glintr50.onnx"])

    if det_found and emb_found:
        print("\n✅ All required models are present.")
    else:
        print("\n⚠️ Missing files:")
        if not det_found: print("  - Detector (det_10g.onnx OR det_2.5g.onnx OR det_500m.onnx OR scrfd_10g_bnkps.onnx)")
        if not emb_found: print("  - Embedder (glintr100.onnx OR glintr50.onnx)")
        print("Re-run the script; you can also delete any .part file before retrying.")

if __name__ == "__main__":
    main()
