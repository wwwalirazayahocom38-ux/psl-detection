import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.models.database import init_db
from app.routers import detection, analytics, auth

app = FastAPI(
    title="Pakistani Sign Language (PSL) Detection System",
    description="Real-time PSL detection with MediaPipe hand tracking, Bi-LSTM neural inference, Urdu Text-to-Speech, and Admin Analytics.",
    version="1.0.0"
)

# Initialize database schema
init_db()

# Enable CORS for cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API and WebSocket routers
app.include_router(detection.router, prefix="/api/detection", tags=["Detection"])
app.include_router(detection.router, prefix="/ws", tags=["WebSocket"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])

# Static files directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "PSL Detection API is running. Build static UI in backend/app/static"}


@app.get("/health")
def health_check():
    from app.services.predictor import SignPredictor
    predictor = SignPredictor()
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "classes_count": len(predictor.label_map)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
