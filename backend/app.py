"""FastAPI service that exposes the pneumonia detection model for inference."""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


LOGGER = logging.getLogger("pneumonia-api")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pneumonia_resnet50_final.h5"
IMG_SIZE = (384, 384)
THRESHOLD = 0.5
CLASS_LABELS = {0: "normal", 1: "pneumonia"}

app = FastAPI(
    title="Pneumonia Detection API",
    description="Upload a chest X-ray to get the pneumonia probability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None


def load_pneumonia_model() -> Any:
    """Lazy-load the keras model so startup doesn't block imports."""
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Make sure pneumonia_resnet50_final.h5 is in backend/models."
            )
        LOGGER.info("Loading pneumonia detection model...")
        MODEL = load_model(MODEL_PATH, compile=False)
        LOGGER.info("Model loaded successfully.")
    return MODEL


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes into a tensor ready for prediction."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("Unable to read the uploaded image.") from exc

    image = image.resize(IMG_SIZE)
    array = np.asarray(image).astype("float32")
    array = preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    return array


@app.get("/", summary="Health check")
def health_check() -> Dict[str, str]:
    """Simple status endpoint."""
    load_pneumonia_model()
    return {"status": "ok", "message": "Pneumonia detection API is running."}


@app.post("/predict", summary="Predict pneumonia probability")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept an uploaded X-ray image and return the pneumonia probability."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        tensor = preprocess_image(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = load_pneumonia_model()
    start = time.perf_counter()
    prediction = model.predict(tensor)
    elapsed = (time.perf_counter() - start) * 1000

    probability = float(prediction[0][0])
    has_pneumonia = probability >= THRESHOLD
    response = {
        "diagnosis": CLASS_LABELS[int(has_pneumonia)],
        "probability": probability,
        "threshold": THRESHOLD,
        "confidence": probability if has_pneumonia else 1 - probability,
        "inference_time_ms": round(elapsed, 2),
    }
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

