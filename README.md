# 🧠 Alzheimer MRI Classification System (MLOps Project)

End-to-end machine learning system for **Alzheimer’s disease stage classification** using MRI images.

This project demonstrates how a deep learning model can be transformed from a research experiment into a **production-ready ML system** using modern **MLOps tools such as FastAPI, Docker, Kubernetes, and CI/CD pipelines**.

---

# 🚀 Project Overview

The goal of this project is to classify MRI brain images into **four stages of Alzheimer’s disease** using a Convolutional Neural Network (CNN).

The system includes:

- Custom CNN model for MRI classification
- REST API for inference
- Docker containerization
- Kubernetes deployment
- CI/CD automation using GitHub Actions

---

# 🧠 Alzheimer Classification Classes

| Class | Description |
|------|-------------|
| NonDemented | Healthy brain MRI |
| VeryMildDemented | Early cognitive impairment |
| MildDemented | Noticeable Alzheimer symptoms |
| ModerateDemented | Advanced Alzheimer stage |

---

# 🧰 Tech Stack

- Python
- TensorFlow
- FastAPI
- Docker
- Kubernetes
- GitHub Actions
- NumPy
- Pillow

---

# 📂 Project Structure

```
alzheimers-mlops-project
│
├── api
│   └── main.py                # FastAPI inference API
│
├── model
│   └── alzheimer_cnn_model.h5 # Trained CNN model
│
├── k8s
│   ├── deployment.yaml        # Kubernetes deployment
│   └── service.yaml           # Kubernetes service
│
├── Dockerfile                 # Container configuration
├── requirements.txt           # Python dependencies
│
└── .github
    └── workflows
        └── ci.yml             # GitHub Actions CI pipeline
```

---

# 🧠 Model Architecture

The project uses a **Convolutional Neural Network (CNN)** for MRI image classification.

Typical architecture:

```
Input Image (224x224x3)
        ↓
Conv2D + ReLU
        ↓
MaxPooling
        ↓
Conv2D
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense Layers
        ↓
Softmax Output (4 classes)
```

Output: Probability distribution across Alzheimer stages.

---

# ⚙️ FastAPI Inference Service

The trained CNN model is exposed through a **REST API**.

## Run API Locally

```bash
uvicorn api.main:app --reload
```

Open interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# 📡 API Endpoints

| Endpoint | Method | Purpose |
|--------|--------|--------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/model-info` | GET | Model metadata |
| `/predict` | POST | Upload MRI image and get prediction |

---

# 🧪 Example Prediction Request

Example using Python:

```python
import requests

url = "http://localhost:8000/predict"

files = {"file": open("mri_scan.jpg", "rb")}

response = requests.post(url, files=files)

print(response.json())
```

Example response:

```json
{
 "prediction": "MildDemented",
 "confidence": 0.92
}
```

---

# 🐳 Docker Containerization

Docker is used to package the application and its dependencies into a portable container.

## Build Docker Image

```bash
docker build -t alzheimer-api .
```

Purpose:
- Packages the model, API, and dependencies into a container.

---

## Run Docker Container

```bash
docker run -p 8000:8000 alzheimer-api
```

Purpose:
- Starts the inference API inside a container.

Access API:

```
http://localhost:8000/docs
```

---

# ☸️ Kubernetes Deployment

Kubernetes is used to run the API as scalable containers.

## Deploy Application

```bash
kubectl apply -f k8s/deployment.yaml
```

Purpose:
- Creates pods running the API container.

---

## Expose Service

```bash
kubectl apply -f k8s/service.yaml
```

Purpose:
- Exposes the API externally through a service.

---

## Check Running Pods

```bash
kubectl get pods
```

Purpose:
- Verify the API containers are running.

---

## Check Services

```bash
kubectl get services
```

Purpose:
- Confirm network access to the API.

---

## View Logs

```bash
kubectl logs <pod-name>
```

Purpose:
- Debug errors or monitor application behavior.

---

# 🔁 CI/CD with GitHub Actions

The project uses **GitHub Actions** to automate the build process.

Pipeline workflow:

```
Push Code
   ↓
GitHub Actions
   ↓
Build Docker Image
   ↓
Run Tests
   ↓
Deploy (optional)
```

This ensures the application builds successfully whenever code changes are pushed.

---

# 🏗 System Architecture

```
MRI Image
   │
   ▼
FastAPI Inference API
   │
   ▼
TensorFlow CNN Model
   │
   ▼
Docker Container
   │
   ▼
Kubernetes Pod
   │
   ▼
Prediction Response
```

---

# 🎯 Project Goals

This project demonstrates:

- Medical image classification
- Deep learning model deployment
- Containerization of ML systems
- Kubernetes orchestration
- CI/CD automation
- Scalable inference architecture

---

# 📌 Future Improvements

Potential enhancements:

- Grad-CAM for explainable AI
- MLflow experiment tracking
- Prometheus + Grafana monitoring
- Model versioning
- GPU inference support
- Automated deployment pipeline

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Burhan Ahmed

AI / ML Systems Engineering Project
