# Pakistani Sign Language (PSL) Real-Time AI Detection System
**Final Year Project (FYP) • Computer Science & Software Engineering**

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21+-FF6F00.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-00C7B7.svg)
![Status](https://img.shields.io/badge/Status-Fully%20Operational-brightgreen.svg)

---

## 🌟 Overview

The **Pakistani Sign Language (PSL) Real-Time AI Detection System** is an assistive computer vision and deep learning application designed to bridge communication barriers for the deaf and mute community in Pakistan.

The system extracts 21 3D hand landmarks in real-time from webcam streams, processes temporal sequence windows through a custom **Bidirectional Long Short-Term Memory (Bi-LSTM)** neural network, and translates gestures into:
1. **Urdu Calligraphic Text** (حروفِ تہجی، اعداد، جملے)
2. **English Transliteration & Phonetics**
3. **Spoken Urdu Audio** via browser Text-to-Speech (TTS)
4. **Admin Analytics Dashboard** with lifetime metrics, confidence distributions, and 7-day trend visualizations.

---

## 🏗️ Architecture

```
                                  [ Browser Client ]
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                [ Video / Canvas ]                   [ Audio & Display ]
             MediaPipe Hands (21 pts)                Urdu TTS & Calligraphy
                        │                                     ▲
                        │ (63 Floats / Frame JSON)            │
                        ▼                                     │
           [ WebSocket: /ws/detect ] ─────────────────────────┘
                        │
                        ▼
            [ FastAPI Asynchronous Backend ]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
[ Sliding Buffer (30 Frames) ]   [ Analytics API / DB ]
         │                             ▲
         ▼                             │
[ Bi-LSTM Neural Model (.h5) ] ────────┘
  51 Classes (Letters/Digits/Phrases)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Webcam for real-time video capture (or use the built-in `⚡ Quick Test Sign` simulation button)

### 2. Run the Application
Navigate to the project root directory and start the FastAPI server:

```powershell
cd C:\Users\pc\.gemini\antigravity-ide\scratch\psl-detection
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open your browser at:  
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📁 Repository Structure

```
psl-detection/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI server entry point & static file router
│   │   ├── models/
│   │   │   ├── database.py      # SQLAlchemy SQLite models (DetectionLog, AdminUser)
│   │   │   └── schemas.py       # Pydantic schemas
│   │   ├── routers/
│   │   │   ├── detection.py     # WebSocket (/ws/detect) & HTTP detection endpoints
│   │   │   ├── analytics.py     # Analytics & KPI endpoints (/api/analytics/summary)
│   │   │   └── auth.py          # Admin login verification
│   │   ├── services/
│   │   │   └── predictor.py     # Bi-LSTM inference, sliding window & smoothing
│   │   ├── ml/
│   │   │   ├── model.h5         # Trained Keras neural network model
│   │   │   └── label_map.json   # 51 PSL classes (Urdu, English, phonetic)
│   │   └── static/              # Glassmorphic web UI
│   │       ├── index.html       # Single-page interface (Detection, Dashboard, Dictionary)
│   │       ├── css/styles.css   # Dark-mode styling, gradients & animations
│   │       └── js/app.js        # MediaPipe client, WebSocket & TTS engine
│   └── test_backend.py          # Automated verification test suite
│
├── ml_pipeline/
│   ├── scripts/
│   │   ├── train_model.py       # Bi-LSTM model training & benchmark synthesis
│   │   └── collect_data.py      # Live webcam data collection utility
│   ├── models/saved/            # Model checkpoints
│   └── data/                    # Raw & processed landmark arrays (.npy)
│
├── docs/
│   ├── SRS.md                   # Software Requirements Specification
│   └── SDD.md                   # Software Design Document
├── docker-compose.yml           # Containerized orchestration
└── README.md                    # Project documentation
```

---

## 📊 Supported PSL Vocabulary (51 Classes)

- **36 Urdu Alphabet Letters (حروفِ تہجی):**  
  ا (Alif), ب (Bay), پ (Pay), ت (Tay), ٹ (Te), ث (Say), ج (Jeem), چ (Chay), ح (Hay), خ (Khay), د (Daal), ڈ (Daal-R), ذ (Zaal), ر (Ray), ڑ (Re), ز (Zay), ژ (Zzhe), س (Seen), ش (Sheen), ص (Saad), ض (Zaad), ط (Toay), ظ (Zoay), ع (Ain), غ (Ghain), ف (Fay), ق (Qaaf), ک (Kaaf), گ (Gaaf), ل (Laam), م (Meem), ن (Noon), و (Wao), ی (Choti Yeh), ے (Bari Yeh), ء (Hamza).

- **10 Urdu Digits (اعداد):**  
  ۰ (0), ۱ (1), ۲ (2), ۳ (3), ۴ (4), ۵ (5), ۶ (6), ۷ (7), ۸ (8), ۹ (9).

- **5 Conversational Phrases (جملے):**  
  السلام علیکم (Salam), شکریہ (Shukriya), ہاں (Haan), نہیں (Nahi), مدد (Madad).

---

## 🧪 Testing & Verification

Run the automated backend test suite:
```powershell
python backend/test_backend.py
```
Outputs validation for:
- Database creation & connectivity
- Label map structure
- Bi-LSTM model loading & tensor inference
- HTTP `/health`, `/api/detection/signs`, and `/api/analytics/summary` status
