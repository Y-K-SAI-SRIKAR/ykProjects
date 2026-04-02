"""
Flask API server for Accident Detector
Run: python server.py
"""
import os, json, glob, time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

WATCH_FOLDER = "watch_folder"
MODEL_DIR    = "model"
KERAS_PATH   = os.path.join(MODEL_DIR, "keras_model.h5")
IMAGE_SIZE   = 224
LABELS       = ["Accident Happened", "No Accident Happened"]
ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}

os.makedirs(WATCH_FOLDER, exist_ok=True)

model = None

def load_model():
    global model
    if model is not None:
        return model
    if not os.path.exists(KERAS_PATH):
        return None
    import tensorflow as tf
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)
    return model


def analyse_authenticity(img):
    from PIL import ImageFilter
    arr  = np.array(img.resize((224, 224)), dtype=np.float32)
    gray = arr.mean(axis=2)
    scores = {}

    white_frac    = np.all(arr > 240, axis=2).mean()
    light_frac    = (gray > 235).mean()
    corners_white = all(
        np.all(arr[r:r+10, c:c+10, :] > 245, axis=2).mean() > 0.85
        for r, c in [(0,0),(0,214),(214,0),(214,214)]
    )
    border_mean = np.mean([arr[:4,:,:].mean(), arr[-4:,:,:].mean(),
                           arr[:,:4,:].mean(), arr[:,-4:,:].mean()])
    if white_frac > 0.25 or corners_white or light_frac > 0.35 or border_mean > 248:
        scores['background'] = 0.0
    elif white_frac > 0.10 or light_frac > 0.20:
        scores['background'] = 0.20
    else:
        scores['background'] = 1.0

    bvars = [np.var(gray[i:i+4, j:j+4]) for i in range(0,220,4) for j in range(0,220,4)]
    ultra_flat_frac = (np.array(bvars) < 2.0).mean()
    if   ultra_flat_frac > 0.35: scores['flat_blocks'] = 0.0
    elif ultra_flat_frac > 0.22: scores['flat_blocks'] = max(0.0, 1.0 - ultra_flat_frac*3.5)
    else:                        scores['flat_blocks'] = 1.0

    r_hist, _ = np.histogram(arr[:,:,0].flatten(), bins=128, range=(0,256))
    top3_frac  = np.sort(r_hist)[-3:].sum() / (224*224)
    if   top3_frac > 0.28: scores['colour_banding'] = 0.0
    elif top3_frac > 0.15: scores['colour_banding'] = max(0.0, 1.0 - top3_frac*4.0)
    else:                  scores['colour_banding'] = 1.0

    maxc = np.maximum(np.maximum(arr[:,:,0], arr[:,:,1]), arr[:,:,2]) + 1e-6
    minc = np.minimum(np.minimum(arr[:,:,0], arr[:,:,1]), arr[:,:,2])
    sat  = (maxc - minc) / maxc
    mid_mask = (gray > 30) & (gray < 220)
    if mid_mask.sum() > 200:
        hyper_sat = (sat[mid_mask] > 0.88).mean()
        scores['saturation'] = max(0.0, min(1.0, 1.0 - hyper_sat / 0.25))
    else:
        scores['saturation'] = 0.3

    pil_g  = Image.fromarray(gray.astype(np.uint8))
    edge_a = np.array(pil_g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    strong = (edge_a > 35).sum()
    medium = ((edge_a > 8) & (edge_a <= 35)).sum()
    if strong > 100:
        ratio = medium / max(strong, 1)
        scores['edge_character'] = min(1.0, ratio / 3.0)
    else:
        scores['edge_character'] = 0.5

    weights = {'background':0.28,'flat_blocks':0.25,'colour_banding':0.22,
               'saturation':0.07,'edge_character':0.07,'lum_histogram':0.04,
               'texture':0.04,'gradient':0.03}
    for k in weights:
        if k not in scores: scores[k] = 0.5

    realness = sum(scores[k]*weights[k] for k in weights)
    if scores['background'] == 0.0:     realness = min(realness, 0.25)
    if scores['flat_blocks'] == 0.0:    realness = min(realness, 0.30)
    if scores['colour_banding'] == 0.0: realness = min(realness, 0.30)
    return float(realness), scores


def apply_penalty(preds, realness):
    if   realness >= 0.65: penalty = 1.00
    elif realness >= 0.50: penalty = 0.70
    elif realness >= 0.35: penalty = 0.30
    elif realness >= 0.20: penalty = 0.10
    else:                  penalty = 0.02
    uniform  = np.array([0.5, 0.5], dtype=np.float32)
    adjusted = penalty * preds + (1.0 - penalty) * uniform
    return adjusted / adjusted.sum()


def get_severity(label, confidence):
    if label != "Accident Happened":
        return "none"
    if   confidence >= 85: return "severe"
    elif confidence >= 65: return "moderate"
    else:                  return "mild"


def predict_image(img_path):
    m = load_model()
    if m is None:
        return {"error": "Model not loaded. Run python detect.py --setup first."}
    img = Image.open(img_path).convert("RGB")
    realness, subs = analyse_authenticity(img)
    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    inp = np.expand_dims((arr / 127.5) - 1.0, axis=0)
    raw   = m.predict(inp, verbose=0)[0]
    adj   = apply_penalty(raw, realness)
    best  = int(np.argmax(adj))
    top_label = LABELS[best]
    top_conf  = float(adj[best]) * 100

    if   realness >= 0.65: img_type = "real";      type_label = "Real Photo"
    elif realness >= 0.50: img_type = "maybe";     type_label = "Possibly Real"
    elif realness >= 0.30: img_type = "uncertain"; type_label = "Uncertain"
    else:                  img_type = "fake";      type_label = "Illustration / CGI / Drawing"

    return {
        "filename":      os.path.basename(img_path),
        "result":        top_label,
        "is_accident":   best == 0,
        "confidence":    top_conf,
        "severity":      get_severity(top_label, top_conf),
        "scores": {
            "accident":    float(adj[0]) * 100,
            "no_accident": float(adj[1]) * 100,
        },
        "realness": {
            "score":      float(realness) * 100,
            "type":       img_type,
            "type_label": type_label,
        },
        "rejected": img_type in ("fake", "uncertain"),
    }


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f   = request.files["image"]
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": "Invalid file type"}), 400
    fname   = secure_filename(f.filename)
    path    = os.path.join(WATCH_FOLDER, fname)
    f.save(path)
    result = predict_image(path)
    return jsonify(result)


@app.route("/latest", methods=["GET"])
def latest():
    files = []
    for e in ["*.jpg","*.jpeg","*.png","*.bmp","*.webp"]:
        files.extend(glob.glob(os.path.join(WATCH_FOLDER, e)))
    if not files:
        return jsonify({"error": "No images in watch_folder"}), 404
    latest = max(files, key=os.path.getmtime)
    return jsonify(predict_image(latest))


@app.route("/health", methods=["GET"])
def health():
    m = load_model()
    return jsonify({"status": "ok", "model_loaded": m is not None})


if __name__ == "__main__":
    print("Starting Accident Detector API on http://localhost:5000")
    load_model()
    app.run(host="0.0.0.0", debug=False, port=5000)