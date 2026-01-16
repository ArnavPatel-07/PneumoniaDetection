# Pneumonia Detection Web App

End-to-end demo that wraps the pneumonia ResNet50 model in a FastAPI backend with a lightweight frontend for uploading chest X-ray images.

## Project structure

```
pneumonia_app/
├── backend/
│   ├── app.py                 # FastAPI service with `/predict`
│   ├── requirements.txt       # Python dependencies
│   └── models/
│       └── pneumonia_resnet50_final.h5
└── frontend/
    ├── index.html             # Single-page UI
    ├── styles.css
    └── app.js
```

## Backend setup

```powershell
cd pneumonia_app/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Place the trained `pneumonia_resnet50_final.h5` file inside `backend/models/`. The API exposes:

- `GET /` – health check
- `POST /predict` – accepts an X-ray image (`multipart/form-data`) and returns the diagnosis, probability, and inference time.

## Frontend usage

Serve the static files with any web server (e.g., VS Code Live Server or `npx serve`):

```powershell
cd pneumonia_app/frontend
npx serve .
```

Update `API_URL` in `frontend/app.js` if your backend runs on a different host/port. The page lets you pick an image, sends it to the API, and displays the results.

## Notes

- Ensure preprocessing matches training: images are resized to 384×384, converted to RGB, and passed through the standard ResNet50 preprocessing pipeline.
- For production, tighten CORS, add authentication, and log inference metadata for auditing.

