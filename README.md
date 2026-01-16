# 🫁 Pneumonia Detection Web Application

A medical-grade deep learning web application for detecting pneumonia from chest X-ray images using ResNet50 transfer learning. Features a modern, responsive frontend and a FastAPI backend with comprehensive model evaluation capabilities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎯 **High Accuracy**: ResNet50-based model with >90% accuracy
- 🏥 **Medical-Grade**: CLAHE preprocessing and clinical evaluation metrics
- 🎨 **Modern UI**: Beautiful, responsive frontend with glassmorphic design
- ⚡ **Fast Inference**: Optimized preprocessing and model serving
- 📊 **Comprehensive Metrics**: AUC-ROC, Precision, Recall, F1-Score
- 🔍 **Interpretability**: Grad-CAM visualization support
- 🚀 **Production-Ready**: FastAPI backend with CORS and error handling

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Documentation](#api-documentation)
- [Frontend Features](#frontend-features)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 📁 Project Structure

```
pneumonia_app/
├── backend/
│   ├── app.py                 # FastAPI service with /predict endpoint
│   ├── requirements.txt       # Python dependencies
│   └── models/
│       └── pneumonia_resnet50_final.h5  # Trained model (not in repo)
├── frontend/
│   ├── index.html             # Main HTML file
│   ├── styles.css             # Modern CSS styling
│   └── app.js                 # Frontend JavaScript
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── TRAINING.md                # Training documentation
```

## 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Node.js (optional, for serving frontend)
- Trained model file (`pneumonia_resnet50_final.h5`)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ArnavPatel-07/PneumoniaDetection.git
cd PneumoniaDetection
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Model File

Place your trained `pneumonia_resnet50_final.h5` model file in the `backend/models/` directory.

**Note**: The model file is not included in the repository due to size. You can:
- Train your own model using the training script (see [Model Training](#model-training))
- Download from a model repository
- Use Git LFS for large files

### 4. Frontend Setup

The frontend is static HTML/CSS/JS. No build process required. You can serve it with:

```bash
# Option 1: Python HTTP server
cd frontend
python -m http.server 5173

# Option 2: Node.js serve
npx serve .

# Option 3: VS Code Live Server extension
# Right-click index.html → "Open with Live Server"
```

## 💻 Usage

### Starting the Backend

```bash
cd backend
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Using the Frontend

1. Open `frontend/index.html` in a browser or serve it using one of the methods above
2. Upload a chest X-ray image (PNG or JPG)
3. Click "Run Analysis"
4. View the prediction results with probability, confidence, and inference time

### API Endpoints

#### Health Check
```bash
GET http://localhost:8000/
```

Response:
```json
{
  "status": "ok",
  "message": "Pneumonia detection API is running."
}
```

#### Predict Pneumonia
```bash
POST http://localhost:8000/predict
Content-Type: multipart/form-data
```

Request: Upload image file as `file` parameter

Response:
```json
{
  "diagnosis": "normal",
  "probability": 0.23,
  "threshold": 0.5,
  "confidence": 0.77,
  "inference_time_ms": 45.23
}
```

## 🎓 Model Training

See [TRAINING.md](TRAINING.md) for detailed training documentation.

### Quick Training Overview

The model uses a two-phase training approach:

1. **Phase 1**: Train classifier head with frozen ResNet50 base
2. **Phase 2**: Fine-tune deeper layers with lower learning rate

Key features:
- CLAHE preprocessing for enhanced X-ray contrast
- Class imbalance handling with weighted loss
- Comprehensive evaluation metrics
- Grad-CAM visualization support

### Training Script

The complete training script is available in the training notebook. Key components:

- Data exploration and class imbalance analysis
- Medical preprocessing with CLAHE
- Two-phase training with callbacks
- Comprehensive evaluation and visualization
- Model saving and versioning

## 📊 API Documentation

### Request Format

- **Method**: POST
- **Endpoint**: `/predict`
- **Content-Type**: `multipart/form-data`
- **Body**: Form data with `file` field containing image

### Response Format

```json
{
  "diagnosis": "normal" | "pneumonia",
  "probability": float,  // Raw model probability (0-1)
  "threshold": float,    // Classification threshold (default: 0.5)
  "confidence": float,  // Confidence in diagnosis (0-1)
  "inference_time_ms": float  // Inference time in milliseconds
}
```

### Error Responses

- **400 Bad Request**: Invalid image or missing file
- **500 Internal Server Error**: Model loading or prediction error

## 🎨 Frontend Features

- **Modern Design**: Glassmorphic UI with animated gradient backdrop
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Real-time Feedback**: Status updates during inference
- **Visual Metrics**: Color-coded diagnosis results
- **Accessibility**: ARIA labels and keyboard navigation support
- **Step-by-step Guide**: User-friendly upload instructions

## 🚢 Deployment

### Backend Deployment

#### Option 1: Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Option 2: Cloud Platforms

- **Heroku**: Use `Procfile` with `web: uvicorn app:app --host 0.0.0.0 --port $PORT`
- **AWS**: Deploy on EC2 or use AWS Lambda with container image
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Use Azure App Service or Container Instances

### Frontend Deployment

- **Netlify**: Drag and drop `frontend/` folder
- **Vercel**: Connect GitHub repository
- **GitHub Pages**: Enable in repository settings
- **AWS S3 + CloudFront**: Static website hosting

### Production Considerations

1. **Security**:
   - Add authentication/authorization
   - Restrict CORS origins
   - Implement rate limiting
   - Add input validation

2. **Performance**:
   - Use model quantization
   - Implement caching
   - Add CDN for frontend
   - Use GPU instances for inference

3. **Monitoring**:
   - Add logging and error tracking
   - Monitor API response times
   - Track prediction accuracy
   - Set up alerts

## ⚠️ Important Notes

### Preprocessing Alignment

⚠️ **Critical**: The training pipeline uses CLAHE preprocessing, but the current FastAPI backend uses standard ResNet50 preprocessing. For accurate predictions, ensure preprocessing matches the training pipeline.

**To fix**: Implement CLAHE preprocessing in `backend/app.py` (see training script for reference).

### Model File

The model file (`pneumonia_resnet50_final.h5`) is not included in the repository. You must:
1. Train your own model using the training script
2. Download from a model repository
3. Use Git LFS for version control

### Medical Disclaimer

⚠️ **This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with licensed healthcare professionals for medical decisions.**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Chest X-Ray dataset providers
- TensorFlow and Keras communities
- FastAPI framework
- Medical imaging research community

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ for medical AI research**
