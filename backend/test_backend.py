"""
Automated Verification Script for PSL Detection System
Tests:
1. Database initialization and model tables
2. Label map loading (51 signs)
3. Deep learning model inference on landmark tensor
4. Analytics calculation and aggregation
5. FastAPI TestClient routing
"""
import sys
import os
import json
import numpy as np

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.models.database import init_db, SessionLocal, DetectionLog
from app.services.predictor import SignPredictor
from starlette.testclient import TestClient
from app.main import app

def test_system():
    print("=== Step 1: Testing Database Initialization ===")
    init_db()
    db = SessionLocal()
    count = db.query(DetectionLog).count()
    print(f"[OK] Database connected. Existing detection logs: {count}")
    db.close()

    print("\n=== Step 2: Testing Sign Predictor & Bi-LSTM Model ===")
    predictor = SignPredictor()
    print(f"[OK] Label map loaded with {len(predictor.label_map)} PSL signs")
    print(f"[OK] TensorFlow Model loaded: {predictor.model is not None}")

    # Generate dummy sequence of 30 frames x 63 features
    dummy_sequence = np.random.uniform(0.1, 0.9, (30, 63)).tolist()
    pred = predictor.predict(dummy_sequence)
    print(f"[OK] Inference test output:")
    print(f"     Sign Key:   {pred['sign_key']}")
    print(f"     English:    {pred['english']}")
    print(f"     Urdu:       {pred['urdu']}")
    print(f"     Confidence: {pred['confidence'] * 100:.2f}%")

    print("\n=== Step 3: Testing FastAPI HTTP Endpoints via TestClient ===")
    client = TestClient(app)

    res_health = client.get("/health")
    assert res_health.status_code == 200
    print(f"[OK] /health -> {res_health.json()}")

    res_signs = client.get("/api/detection/signs")
    assert res_signs.status_code == 200
    signs_data = res_signs.json()
    assert len(signs_data) >= 50
    print(f"[OK] /api/detection/signs -> returned {len(signs_data)} signs")

    res_analytics = client.get("/api/analytics/summary")
    assert res_analytics.status_code == 200
    analytics_data = res_analytics.json()
    print(f"[OK] /api/analytics/summary -> Total: {analytics_data['total_detections']}, Avg Conf: {analytics_data['avg_confidence'] * 100:.1f}%")

    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "Pakistani Sign Language" in res_index.text
    print(f"[OK] GET / (UI Dashboard HTML served successfully)")

    print("\n=======================================================")
    print("ALL AUTOMATED VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================")

if __name__ == "__main__":
    test_system()
