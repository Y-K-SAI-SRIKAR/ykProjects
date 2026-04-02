"""
Accident Detector — with Deep Real-Photo Filter
================================================
Accepts only real-world photographs.
Rejects: vector illustrations, CGI renders, drawings, logos, clip art.

Install:
    pip install tensorflow pillow watchdog numpy

Usage:
    python detect.py --setup               # ONE-TIME: builds keras_model.h5
    python detect.py --image photo.jpg
    python detect.py --latest ./folder
    python detect.py --watch  ./folder
"""

import os, sys, json, glob, time, argparse
import numpy as np

LABELS     = ["Accident Happened", "No Accident Happened"]
IMAGE_SIZE = 224
MODEL_DIR  = "model"
KERAS_PATH = os.path.join(MODEL_DIR, "keras_model.h5")

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  X = "\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE AUTHENTICITY ANALYSER  (8 checks)
#
#  The 3 dominant signals that catch illustrations/CGI/vectors:
#    1. Large white/solid background  (illustrations: 40%+ pixels are pure white)
#    2. Ultra-flat pixel blocks       (illustrations: 40%+ 4x4 blocks have var<2)
#    3. Colour banding                (illustrations: top-3 bins cover 40%+ pixels)
#
#  Supporting signals:
#    4. Saturation extremes
#    5. Edge character (cartoon outlines vs natural edges)
#    6. Luminance histogram spikiness
#    7. Texture in non-edge areas
#    8. Gradient smoothness pattern
# ══════════════════════════════════════════════════════════════════════════════
def analyse_authenticity(img):
    """Returns (realness 0.0–1.0, scores_dict)"""
    from PIL import ImageFilter

    arr  = np.array(img.resize((224, 224)), dtype=np.float32)
    gray = arr.mean(axis=2)

    scores = {}

    # ── 1. Background / solid region detection ────────────────────────────────
    white_frac  = np.all(arr > 240, axis=2).mean()
    light_frac  = (gray > 235).mean()
    corners_white = all(
        np.all(arr[r:r+10, c:c+10, :] > 245, axis=2).mean() > 0.85
        for r, c in [(0,0),(0,214),(214,0),(214,214)]
    )
    border_mean = np.mean([
        arr[:4,:,:].mean(), arr[-4:,:,:].mean(),
        arr[:,:4,:].mean(), arr[:,-4:,:].mean()
    ])
    if white_frac > 0.25 or corners_white or light_frac > 0.35 or border_mean > 248:
        scores['background'] = 0.0
    elif white_frac > 0.10 or light_frac > 0.20:
        scores['background'] = 0.20
    else:
        scores['background'] = 1.0

    # ── 2. Flat-block ratio ───────────────────────────────────────────────────
    bvars = []
    for i in range(0, 220, 4):
        for j in range(0, 220, 4):
            bvars.append(np.var(gray[i:i+4, j:j+4]))
    bvars = np.array(bvars)
    ultra_flat_frac = (bvars < 2.0).mean()
    if   ultra_flat_frac > 0.35: scores['flat_blocks'] = 0.0
    elif ultra_flat_frac > 0.22: scores['flat_blocks'] = max(0.0, 1.0 - ultra_flat_frac * 3.5)
    else:                        scores['flat_blocks'] = 1.0

    # ── 3. Colour banding ─────────────────────────────────────────────────────
    r_hist, _ = np.histogram(arr[:,:,0].flatten(), bins=128, range=(0, 256))
    top3_frac  = np.sort(r_hist)[-3:].sum() / (224 * 224)
    if   top3_frac > 0.28: scores['colour_banding'] = 0.0
    elif top3_frac > 0.15: scores['colour_banding'] = max(0.0, 1.0 - top3_frac * 4.0)
    else:                  scores['colour_banding'] = 1.0

    # ── 4. Saturation extremes ────────────────────────────────────────────────
    maxc = np.maximum(np.maximum(arr[:,:,0], arr[:,:,1]), arr[:,:,2]) + 1e-6
    minc = np.minimum(np.minimum(arr[:,:,0], arr[:,:,1]), arr[:,:,2])
    sat  = (maxc - minc) / maxc
    mid_mask = (gray > 30) & (gray < 220)
    if mid_mask.sum() > 200:
        hyper_sat = (sat[mid_mask] > 0.88).mean()
        scores['saturation'] = max(0.0, min(1.0, 1.0 - hyper_sat / 0.25))
    else:
        scores['saturation'] = 0.3

    # ── 5. Edge character ─────────────────────────────────────────────────────
    from PIL import Image as PILImage
    pil_g  = PILImage.fromarray(gray.astype(np.uint8))
    edge_a = np.array(pil_g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    strong = (edge_a > 35).sum()
    medium = ((edge_a > 8) & (edge_a <= 35)).sum()
    if strong > 100:
        ratio = medium / max(strong, 1)
        scores['edge_character'] = min(1.0, ratio / 3.0)
    else:
        scores['edge_character'] = 0.5

    # ── 6. Luminance histogram spikiness ─────────────────────────────────────
    hist64, _ = np.histogram(gray.flatten(), bins=64, range=(0, 256))
    max_bin   = hist64.max() / (224 * 224)
    zero_bins = (hist64 == 0).sum()
    if   max_bin > 0.15 or zero_bins > 30: scores['lum_histogram'] = 0.1
    elif max_bin > 0.08 or zero_bins > 18: scores['lum_histogram'] = 0.4
    else:                                  scores['lum_histogram'] = 1.0

    # ── 7. Texture in non-edge areas ─────────────────────────────────────────
    lap      = np.array(pil_g.filter(ImageFilter.SMOOTH_MORE), dtype=np.float32)
    lap_diff = np.abs(gray - lap)
    non_edge = edge_a < 12
    if non_edge.sum() > 500:
        tex_val = np.mean(lap_diff[non_edge])
        scores['texture'] = min(1.0, tex_val / 2.5)
    else:
        scores['texture'] = 0.4

    # ── 8. Gradient smoothness pattern ───────────────────────────────────────
    dx = np.abs(np.diff(arr, axis=1)).flatten()
    dy = np.abs(np.diff(arr, axis=0)).flatten()
    all_grad   = np.concatenate([dx, dy])
    near_zero  = (all_grad < 3).mean()
    large_jump = (all_grad > 55).mean()
    if near_zero > 0.70 and large_jump > 0.04:
        scores['gradient'] = 0.05
    elif near_zero > 0.60:
        scores['gradient'] = 0.3
    elif 0.35 < near_zero < 0.60:
        scores['gradient'] = 1.0
    else:
        scores['gradient'] = 0.6

    # ── Weighted final score ──────────────────────────────────────────────────
    weights = {
        'background':    0.28,
        'flat_blocks':   0.25,
        'colour_banding':0.22,
        'saturation':    0.07,
        'edge_character':0.07,
        'lum_histogram': 0.04,
        'texture':       0.04,
        'gradient':      0.03,
    }
    realness = sum(scores[k] * weights[k] for k in weights)

    # Hard overrides
    if scores['background'] == 0.0:     realness = min(realness, 0.25)
    if scores['flat_blocks'] == 0.0:    realness = min(realness, 0.30)
    if scores['colour_banding'] == 0.0: realness = min(realness, 0.30)

    return float(realness), scores


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIDENCE PENALTY
# ══════════════════════════════════════════════════════════════════════════════
def apply_penalty(preds, realness):
    if   realness >= 0.65: penalty = 1.00
    elif realness >= 0.50: penalty = 0.70
    elif realness >= 0.35: penalty = 0.30
    elif realness >= 0.20: penalty = 0.10
    else:                  penalty = 0.02

    uniform  = np.array([0.5, 0.5], dtype=np.float32)
    adjusted = penalty * preds + (1.0 - penalty) * uniform
    return adjusted / adjusted.sum(), penalty


def severity_tag(label, confidence):
    """Returns severity string for accident results."""
    if label != "Accident Happened":
        return ""
    if   confidence >= 85: return f" {R}[ SEVERE ]{X}"
    elif confidence >= 65: return f" {Y}[ MODERATE ]{X}"
    else:                  return f" {Y}[ MILD ]{X}"


def realness_tag(s):
    if s >= 0.65: return f"{G}📷 Real Photo{X}",                     "real"
    if s >= 0.50: return f"{Y}📷 Possibly Real (low confidence){X}", "maybe"
    if s >= 0.30: return f"{Y}⚠  Unclear — may not be a photo{X}",  "uncertain"
    return               f"{R}🎨 Illustration / CGI / Drawing{X}",   "fake"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════
def parse_weights_bin():
    with open(os.path.join(MODEL_DIR, "model.json")) as f:
        tfjs = json.load(f)
    manifest     = tfjs["weightsManifest"]
    weight_specs, bin_files = [], []
    for entry in manifest:
        weight_specs.extend(entry["weights"])
        for p in entry["paths"]:
            bin_files.append(os.path.join(MODEL_DIR, p))
    raw = b""
    for bf in bin_files:
        with open(bf, "rb") as f: raw += f.read()
    tensors, offset = [], 0
    for spec in weight_specs:
        name  = spec["name"]
        shape = spec["shape"]
        count = int(np.prod(shape)) if shape else 1
        nb    = count * 4
        arr   = np.frombuffer(raw[offset:offset+nb], dtype=np.float32).copy()
        tensors.append((name, shape, arr.reshape(shape) if shape else arr))
        offset += nb
    print(f"  Parsed {len(tensors)} tensors")
    return tensors


def build_model():
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        alpha=0.35, include_top=False, weights=None)
    inp = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x   = base(inp)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dense(100, activation='relu')(x)
    out = tf.keras.layers.Dense(2,   activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32))
    return model


def assign_weights(model, tensors):
    trainable     = [(n,s,a) for n,s,a in tensors if "moving_mean" not in n and "moving_variance" not in n]
    non_trainable = [(n,s,a) for n,s,a in tensors if "moving_mean" in n or "moving_variance" in n]
    assigned = 0
    for kw, (n,s,a) in zip(model.trainable_weights,     trainable):
        if kw.shape == a.shape: kw.assign(a); assigned += 1
    for kw, (n,s,a) in zip(model.non_trainable_weights, non_trainable):
        if kw.shape == a.shape: kw.assign(a); assigned += 1
    print(f"  Assigned {assigned} weight tensors")
    return assigned


def setup():
    print(f"\n{C}{B}Setting up model (one-time)...{X}")
    model   = build_model()
    tensors = parse_weights_bin()
    assign_weights(model, tensors)
    dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    out   = model(dummy, training=False).numpy()[0]
    print(f"  Sanity check output: {out}")
    ok = abs(out[0] - 0.5) > 0.001
    print(f"  {(G+'✅ Weights loaded correctly!') if ok else (Y+'⚠ Still 50/50 — weight mismatch')}{X}")
    model.save(KERAS_PATH)
    print(f"\n{G}{B}✅  Saved: {KERAS_PATH}{X}\n")


def load_model():
    if not os.path.exists(KERAS_PATH):
        print(f"{R}Run --setup first.{X}"); sys.exit(1)
    import tensorflow as tf
    print(f"{C}Loading model...{X}")
    m = tf.keras.models.load_model(KERAS_PATH, compile=False)
    print(f"{G}✅  Model ready.{X}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(img):
    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    return np.expand_dims((arr / 127.5) - 1.0, axis=0)


def predict(model, img_path):
    from PIL import Image
    if not os.path.isfile(img_path):
        print(f"{R}File not found: {img_path}{X}"); return

    img = Image.open(img_path).convert("RGB")

    # Authenticity analysis
    realness, subs = analyse_authenticity(img)
    tag, rtype     = realness_tag(realness)

    # Model prediction
    raw_preds      = model.predict(preprocess(img), verbose=0)[0]
    adj_preds, pen = apply_penalty(raw_preds, realness)

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"{B}  {os.path.basename(img_path)}{X}")
    print(f"{'─'*62}")

    r_bar = "█" * int(realness * 34) + "░" * (34 - int(realness * 34))
    print(f"  Type        : {tag}")
    print(f"  Photo Score : [{r_bar}] {realness*100:.1f}%")

    if rtype != "real":
        lines = []
        for k, v in subs.items():
            flag = f" {R}✗{X}" if v < 0.3 else (f" {Y}~{X}" if v < 0.65 else "")
            lines.append(f"{k[:4]}:{v:.2f}{flag}")
        print(f"  {Y}  Signals: {'  '.join(lines)}{X}")

    print(f"{'─'*62}")

    for label, conf in zip(LABELS, adj_preds):
        f   = int(conf * 44)
        bar = "█"*f + "░"*(44-f)
        col = G if conf == max(adj_preds) else ""
        sev = severity_tag(label, conf * 100) if conf == max(adj_preds) else ""
        print(f"  {col}{label:<22} | {bar} | {conf*100:6.2f}%{sev}{X}")

    print(f"{'─'*62}")
    best   = int(np.argmax(adj_preds))
    winner = LABELS[best]
    conf   = adj_preds[best] * 100
    emoji  = "🚨" if winner == "Accident Happened" else "✅"
    col    = R if winner == "Accident Happened" else G

    if rtype == "fake":
        print(f"  {R}✗  REJECTED — Not a real photograph  (photo score: {realness*100:.0f}%){X}")
    elif rtype == "uncertain":
        print(f"  {Y}⚠  Low confidence — image does not look like a real photo{X}")
    elif rtype == "maybe":
        print(f"  {Y}⚠  Moderate confidence — image may not be a real photo{X}")

    sev = severity_tag(winner, conf).strip()
    print(f"  {B}{col}{emoji}  {winner}  ({conf:.1f}% confidence){' '+sev if sev else ''}{X}")
    print(f"{'═'*62}\n")
    return winner, adj_preds


# ══════════════════════════════════════════════════════════════════════════════
#  WATCH / LATEST
# ══════════════════════════════════════════════════════════════════════════════
def get_latest_image(folder):
    files = []
    for e in ["*.jpg","*.jpeg","*.png","*.bmp","*.webp",
              "*.JPG","*.JPEG","*.PNG","*.BMP","*.WEBP"]:
        files.extend(glob.glob(os.path.join(folder, e)))
    return max(files, key=os.path.getmtime) if files else None


def watch_folder(model, folder):
    from watchdog.observers import Observer
    from watchdog.events    import FileSystemEventHandler
    EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
    class H(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and \
               os.path.splitext(event.src_path)[1].lower() in EXTS:
                print(f"{Y}📂  {event.src_path}{X}")
                time.sleep(0.5)
                predict(model, event.src_path)
    os.makedirs(folder, exist_ok=True)
    obs = Observer()
    obs.schedule(H(), path=folder, recursive=False)
    obs.start()
    print(f"\n{B}{C}👁  Watching: {os.path.abspath(folder)}{X}")
    print(f"{C}   Drop images in — Ctrl+C to stop.{X}\n")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Accident Detector — Real Photo Filter")
    g  = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--setup",  action="store_true", help="Build keras_model.h5 (run once)")
    g.add_argument("--image",  metavar="FILE",       help="Predict single image")
    g.add_argument("--latest", metavar="FOLDER",     help="Predict newest image in folder")
    g.add_argument("--watch",  metavar="FOLDER",     help="Watch folder, auto-predict")
    args = ap.parse_args()
    if args.setup:
        setup(); return
    model = load_model()
    if args.image:   predict(model, args.image)
    elif args.latest:
        img = get_latest_image(args.latest)
        predict(model, img) if img else print(f"{R}No images found.{X}")
    elif args.watch: watch_folder(model, args.watch)

if __name__ == "__main__":
    main()
