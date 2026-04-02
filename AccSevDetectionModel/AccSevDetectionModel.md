# 🚨 AccSevDetectionModel — Road Accident Severity Detection

> A supervised machine learning model that detects and classifies the severity of road accidents from images using computer vision concepts, deployed as a full-stack web application with React frontend and Python backend, containerized with Docker.

---

## 📌 Overview

**AccSevDetectionModel** is an end-to-end machine learning project that analyzes road accident images and determines whether an accident has occurred and how severe it is. The model is trained on image datasets using supervised learning techniques and leverages image perception concepts from computer vision to classify severity levels. The application is fully containerized using Docker and accessible via a modern React-based web interface.

---

## ✨ Features

- 🧠 **ML-Powered Detection** — Trained on road accident image datasets using supervised machine learning
- 📊 **Severity Classification** — Classifies accidents as `Mild`, `Moderate`, or `Severe` based on image perception
- 🖼️ **Image Authenticity Check** — Detects whether uploaded image is a real photo or illustration/CGI
- ⚡ **Real-time Updates** — Uses `watchdog` to monitor and process new images instantly
- 🌐 **React Frontend** — Clean and responsive web UI built with React + Vite
- 🐍 **Python Backend** — Flask API server handling predictions and image processing
- 🐳 **Dockerized** — Fully containerized as a single Docker image running both frontend and backend
- 📦 **DockerHub Deployment** — Published and available on DockerHub for easy pull and run

---

## 🛠️ Tech Stack

### Machine Learning & Backend
| Technology | Purpose |
|-----------|---------|
| Python | Core backend language |
| TensorFlow / Keras | ML model training and inference |
| NumPy | Numerical computations and array operations |
| Pandas | Data handling and processing |
| Pillow (PIL) | Image preprocessing and manipulation |
| Watchdog | Real-time folder monitoring for new images |
| Flask | REST API server |
| Flask-CORS | Cross-origin request handling |

### Frontend
| Technology | Purpose |
|-----------|---------|
| React | UI framework |
| Vite | Frontend build tool |
| JavaScript | Frontend logic |

### DevOps & Deployment
| Technology | Purpose |
|-----------|---------|
| Docker | Containerization |
| Supervisord | Running multiple processes in single container |
| DockerHub | Image registry and deployment |

---

## 🧠 How It Works

```
User uploads image
        ↓
Image Authenticity Analysis
(Real photo? CGI? Illustration?)
        ↓
Keras Model Inference
(Trained on accident image datasets)
        ↓
Severity Classification
(Based on image perception & computer vision)
        ↓
Result → Accident / No Accident
         Severity → Mild / Moderate / Severe
```

### Severity Classification Logic
| Confidence | Severity |
|-----------|---------|
| ≥ 85% | 🔴 Severe |
| ≥ 65% | 🟠 Moderate |
| < 65% | 🟡 Mild |
| No accident | ✅ None |

---

## 📁 Project Structure

```
AccSevDetectionModel/
├── Dockerfile
├── docker-compose.yml
├── supervisord.conf
├── .dockerignore
├── run.bat                   # Windows quick start script
├── run.sh                    # Mac/Linux quick start script
├── backend/
│   ├── server.py             # Flask API server
│   ├── detect.py             # Detection logic
│   ├── Metrics.py            # Evaluation metrics
│   ├── requirements.txt      # Python dependencies
│   ├── model/
│   │   └── keras_model.h5    # Trained Keras model
│   └── watch_folder/         # Monitored folder for new images
└── frontend/
    ├── src/
    ├── public/
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## 🐳 Run with Docker (Recommended)

### Prerequisites
- Docker Desktop installed and running

### Pull and Run from DockerHub

```bash
# Pull the image
docker pull srikar77/ykprojects:AccSevDetectionModel

# Run the container
docker run -d -p 5173:5173 -p 5000:5000 --name accsev_container srikar77/ykprojects:AccSevDetectionModel
```

### Or use Docker Compose

```bash
# Clone the repository
git clone https://github.com/yourusername/AccSevDetectionModel.git
cd AccSevDetectionModel

# Start the app
docker compose up -d
```

### Access the App
| URL | Service |
|-----|---------|
| `http://localhost:5173` | React Frontend |
| `http://localhost:5000/health` | Backend Health Check |
| `http://localhost:5000/upload` | Upload Image API |
| `http://localhost:5000/latest` | Latest Prediction API |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/health` | Check if server and model are running |
| `POST` | `/upload` | Upload an image for accident detection |
| `GET` | `/latest` | Get prediction for the latest image in watch folder |

### Sample Response
```json
{
  "filename": "accident.jpg",
  "result": "Accident Happened",
  "is_accident": true,
  "confidence": 91.5,
  "severity": "severe",
  "scores": {
    "accident": 91.5,
    "no_accident": 8.5
  },
  "realness": {
    "score": 78.3,
    "type": "real",
    "type_label": "Real Photo"
  },
  "rejected": false
}
```

---

## 🚀 Build from Source

```bash
# Clone the repo
git clone https://github.com/yourusername/AccSevDetectionModel.git
cd AccSevDetectionModel

# Build Docker image
docker build -t srikar77/ykprojects:AccSevDetectionModel .

# Run
docker compose up -d
```

---

## 📦 DockerHub

The image is publicly available on DockerHub:

```
docker pull srikar77/ykprojects:AccSevDetectionModel
```

🔗 [hub.docker.com/r/srikar77/ykprojects](https://hub.docker.com/r/srikar77/ykprojects)

---

## 📊 Model Details

- **Model Type:** Convolutional Neural Network (CNN) via Keras/TensorFlow
- **Input Size:** 224 × 224 RGB images
- **Training:** Supervised learning on labeled road accident image datasets
- **Output:** Binary classification — Accident / No Accident with confidence score
- **Authenticity Filter:** Rejects CGI, illustrations, and synthetic images before inference

---

## 👨‍💻 Author

**Srikar** — [DockerHub: srikar77](https://hub.docker.com/u/srikar77)

---

## 📄 License

This project is licensed under the MIT License.