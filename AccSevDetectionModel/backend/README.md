# 🚨 Accident Detector

Detects accidents in images using your **Google Teachable Machine** trained model.
Watches a folder and automatically predicts confidence levels on any new image dropped in.
Includes an **8-signal real photo filter** that rejects drawings, logos, and illustrations.

---

## 📁 Folder Structure

```
accident_detector/
├── detect.py           ← main script (CLI)
├── server.py           ← Flask API server
├── App.jsx             ← React frontend
├── model/
│   ├── model.json      ← from Teachable Machine export (TF.js)
│   ├── weights.bin     ← from Teachable Machine export
│   └── metadata.json   ← from Teachable Machine export
└── watch_folder/       ← drop images here (auto-created)
```

---

## ⚙️ Setup (one time)

```bash
pip install tensorflow pillow watchdog numpy flask flask-cors
```

Then build the Keras model from your TF.js export (only needed once):

```bash
python detect.py --setup
```

This creates `model/keras_model.h5`.

---

## 🚀 Usage — Command Line

### Watch a folder (auto-predict on new images)
```bash
python detect.py --watch ./watch_folder
```
Drop any image into `watch_folder/` — results appear instantly!

### Predict on a single image
```bash
python detect.py --image path/to/photo.jpg
```

### Predict on the latest image in a folder
```bash
python detect.py --latest ./watch_folder
```

---

## 🌐 Usage — Web UI

**Terminal 1** — start the API:
```bash
python server.py
```

**Terminal 2** — serve the React frontend.

If you have Node/Vite:
```bash
npm create vite@latest ui -- --template react
cd ui
cp ../App.jsx src/App.jsx
npm install && npm run dev
```

Or simply open `App.jsx` in any React sandbox (StackBlitz, CodeSandbox).

---

## 📊 Sample Output

```
══════════════════════════════════════════════════════════════
  crash_photo.jpg
──────────────────────────────────────────────────────────────
  Type        : 📷 Real Photo
  Photo Score : [██████████████████████████████████] 91.2%
──────────────────────────────────────────────────────────────
  Accident Happened      | ████████████████████████████░░░░ |  87.43%  [ SEVERE ]
  No Accident Happened   | █████░░░░░░░░░░░░░░░░░░░░░░░░░░░ |  12.57%
──────────────────────────────────────────────────────────────
  🚨  Accident Happened  (87.4% confidence)  SEVERE
══════════════════════════════════════════════════════════════
```

---

## 🏷️ Labels
- **Accident Happened** — image contains a road accident
- **No Accident Happened** — no accident detected

### Severity (when Accident Happened):
| Severity | Confidence |
|---|---|
| 🔴 Severe   | ≥ 85% |
| 🟡 Moderate | 65–84% |
| 🟢 Mild     | < 65% |

---

## 🛡️ Real Photo Filter

The 8-signal authenticity check rejects non-real images:

| Image Type | Photo Score | Effect |
|---|---|---|
| Real accident photo | 65–100% | Full confidence |
| Possibly real | 50–64% | 30% reduction |
| Uncertain | 35–49% | 70% reduction |
| Drawing / Logo / CGI | < 35% | 98% reduction |

Model trained via Google Teachable Machine (image size: 224×224).
