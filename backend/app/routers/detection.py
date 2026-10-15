import json
import collections
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session

from app.models.database import get_db, SessionLocal, DetectionLog
from app.services.predictor import SignPredictor
from app.models.schemas import DetectionResult

router = APIRouter()
predictor = SignPredictor()


@router.websocket("/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    buffer = collections.deque(maxlen=30)
    last_saved_sign = None
    last_saved_time = datetime.min

    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = json.loads(raw_text)

            # Support both array of landmarks or object with session_id
            session_id = payload.get("session_id", "web-session") if isinstance(payload, dict) else "web-session"
            landmarks = payload.get("landmarks", payload) if isinstance(payload, dict) else payload

            if not isinstance(landmarks, list) or len(landmarks) == 0:
                continue

            buffer.append(landmarks)

            # Predict when we have at least 10 frames
            if len(buffer) >= 10:
                prediction = predictor.predict(list(buffer))

                if prediction:
                    # Save to database if confidence exceeds threshold and not duplicate rapid chatter
                    now = datetime.utcnow()
                    elapsed = (now - last_saved_time).total_seconds()

                    if prediction["confidence"] >= 0.75 and (prediction["english"] != last_saved_sign or elapsed > 2.5):
                        db = SessionLocal()
                        try:
                            log_entry = DetectionLog(
                                sign_key=prediction["sign_key"],
                                english_label=prediction["english"],
                                urdu_label=prediction["urdu"],
                                phonetic=prediction["phonetic"],
                                category=prediction["category"],
                                confidence=prediction["confidence"],
                                session_id=session_id,
                                audio_spoken=True
                            )
                            db.add(log_entry)
                            db.commit()
                            last_saved_sign = prediction["english"]
                            last_saved_time = now
                        except Exception as e:
                            print(f"[WS Log Error] {e}")
                        finally:
                            db.close()

                    await websocket.send_json({
                        "status": "success",
                        "sign_key": prediction["sign_key"],
                        "english": prediction["english"],
                        "urdu": prediction["urdu"],
                        "phonetic": prediction["phonetic"],
                        "category": prediction["category"],
                        "confidence": prediction["confidence"],
                        "timestamp": datetime.utcnow().isoformat()
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS Error] {e}")


@router.post("/single", response_model=DetectionResult)
def predict_single(data: dict):
    landmarks = data.get("landmarks", [])
    seq = [landmarks] * 30
    res = predictor.predict(seq)
    return res


@router.get("/signs")
def list_available_signs():
    return predictor.get_all_signs()
