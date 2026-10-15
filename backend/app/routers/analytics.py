import random
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import get_db, DetectionLog
from app.models.schemas import AnalyticsSummary, DetectionLogResponse

router = APIRouter()


def seed_sample_data_if_empty(db: Session):
    count = db.query(DetectionLog).count()
    if count < 15:
        sample_signs = [
            ("0", "Alif", "الف", "Alif", "letter", 0.94),
            ("1", "Bay", "بے", "Bay", "letter", 0.92),
            ("2", "Pay", "پے", "Pay", "letter", 0.89),
            ("3", "Tay", "تے", "Tay", "letter", 0.95),
            ("6", "Jeem", "جیم", "Jeem", "letter", 0.91),
            ("10", "Daal", "دال", "Daal", "letter", 0.88),
            ("17", "Seen", "سین", "Seen", "letter", 0.93),
            ("30", "Meem", "میم", "Meem", "letter", 0.96),
            ("36", "0 (Sifar)", "۰ (صفر)", "Sifar", "digit", 0.97),
            ("37", "1 (Aik)", "۱ (ایک)", "Aik", "digit", 0.95),
            ("46", "Salam", "السلام علیکم", "Assalam o Alaikum", "phrase", 0.98),
            ("47", "Shukriya", "شکریہ", "Shukriya", "phrase", 0.94),
            ("48", "Haan (Yes)", "ہاں", "Haan", "phrase", 0.90),
            ("50", "Madad (Help)", "مدد", "Madad", "phrase", 0.96),
        ]
        now = datetime.utcnow()
        for i in range(45):
            sign = random.choice(sample_signs)
            days_ago = random.randint(0, 6)
            minutes_ago = random.randint(5, 720)
            entry_time = now - timedelta(days=days_ago, minutes=minutes_ago)
            conf = min(0.99, max(0.78, sign[5] + random.uniform(-0.06, 0.04)))

            log = DetectionLog(
                sign_key=sign[0],
                english_label=sign[1],
                urdu_label=sign[2],
                phonetic=sign[3],
                category=sign[4],
                confidence=round(conf, 4),
                timestamp=entry_time,
                session_id=f"session-{random.randint(101, 108)}",
                audio_spoken=True
            )
            db.add(log)
        db.commit()


@router.get("/summary", response_model=AnalyticsSummary)
def get_analytics_summary(db: Session = Depends(get_db)):
    seed_sample_data_if_empty(db)

    total = db.query(DetectionLog).count()
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.query(DetectionLog).filter(DetectionLog.timestamp >= today_start).count()

    avg_conf_raw = db.query(func.avg(DetectionLog.confidence)).scalar()
    avg_confidence = round(float(avg_conf_raw or 0.0), 4)

    unique_signs = db.query(func.count(func.distinct(DetectionLog.english_label))).scalar() or 0

    letters_count = db.query(DetectionLog).filter(DetectionLog.category == "letter").count()
    digits_count = db.query(DetectionLog).filter(DetectionLog.category == "digit").count()
    phrases_count = db.query(DetectionLog).filter(DetectionLog.category == "phrase").count()

    popular_query = (
        db.query(
            DetectionLog.english_label,
            DetectionLog.urdu_label,
            DetectionLog.category,
            func.count(DetectionLog.id).label("count"),
            func.avg(DetectionLog.confidence).label("avg_conf")
        )
        .group_by(DetectionLog.english_label)
        .order_by(func.count(DetectionLog.id).desc())
        .limit(8)
        .all()
    )

    popular_signs = [
        {
            "english": row.english_label,
            "urdu": row.urdu_label,
            "category": row.category or "letter",
            "count": row.count,
            "avg_confidence": round(float(row.avg_conf or 0.0), 3)
        }
        for row in popular_query
    ]

    recent_logs = (
        db.query(DetectionLog)
        .order_by(DetectionLog.timestamp.desc())
        .limit(10)
        .all()
    )

    # 7-day trend
    daily_trend = []
    for i in range(6, -1, -1):
        target_day = (datetime.utcnow() - timedelta(days=i)).date()
        day_start = datetime.combine(target_day, datetime.min.time())
        day_end = datetime.combine(target_day, datetime.max.time())

        day_count = (
            db.query(DetectionLog)
            .filter(DetectionLog.timestamp >= day_start, DetectionLog.timestamp <= day_end)
            .count()
        )
        day_conf = (
            db.query(func.avg(DetectionLog.confidence))
            .filter(DetectionLog.timestamp >= day_start, DetectionLog.timestamp <= day_end)
            .scalar()
        ) or 0.88

        daily_trend.append({
            "date": target_day.strftime("%b %d"),
            "detections": day_count,
            "avg_confidence": round(float(day_conf), 2)
        })

    return {
        "total_detections": total,
        "today_detections": today_count,
        "avg_confidence": avg_confidence,
        "unique_signs_detected": unique_signs,
        "letters_count": letters_count,
        "digits_count": digits_count,
        "phrases_count": phrases_count,
        "popular_signs": popular_signs,
        "recent_activity": recent_logs,
        "daily_trend": daily_trend
    }


@router.delete("/reset")
def reset_analytics(db: Session = Depends(get_db)):
    db.query(DetectionLog).delete()
    db.commit()
    seed_sample_data_if_empty(db)
    return {"message": "Analytics database reset and re-seeded with demo records successfully."}
