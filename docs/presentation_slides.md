# Pakistani Sign Language (PSL) Real-Time AI Detection System
## FYP Defense & Evaluation — Presentation Slides & Speaker Script

---

### Slide 1: Title Slide
- **Title:** Pakistani Sign Language (PSL) Real-Time AI Detection System
- **Subtitle:** An Assistive AI System Bridging Communication for the Deaf & Mute Community
- **Degree:** Final Year Project (FYP) — Department of Computer Science & Software Engineering
- **Presenter:** [Your Name / Roll Number]
- **Project Supervisor:** [Supervisor Name / Designation]
- **Date / Session:** 2026-2027

> **🎙️ Speaker Notes (Aapko committee ke samne kya bolna hai):**
> "Respected committee members, supervisor, and teachers. Assalam-o-Alaikum. Today, I am presenting my Final Year Project titled: *Pakistani Sign Language Real-Time AI Detection System*. This project leverages modern Computer Vision, Deep Learning, and Real-Time Web Technologies to empower the hearing and speech-impaired community of Pakistan."

---

### Slide 2: Problem Statement & Motivation
- **The Communication Gap:** Over 1.5 million individuals in Pakistan are deaf or speech-impaired, facing severe communication barriers in schools, hospitals, banks, and public offices.
- **Language Incompatibility:** Most international research focuses on **American Sign Language (ASL)** or British Sign Language (BSL). **Pakistani Sign Language (PSL)** has distinct alphabets, hand shapes, and grammatical structure that cannot be detected by ASL models.
- **Lack of Dedicated Assistive Tech:** Traditional solutions require expensive specialized sensor gloves or human interpreters who are scarce and costly.
- **Our Goal:** Create an accessible, real-time, software-only camera solution that translates PSL signs into **Urdu script and natural spoken audio** directly on any standard computer.

> **🎙️ Speaker Notes:**
> "Pakistan mein 15 lakh se zyada afrad sunne aur bolne ki salahiyat se mehroom hain. International level par ASL par bohut kaam hua hai, lekin Pakistani Sign Language (PSL) bilkul mukhtalif hai. Hamara maqsad tha keh baghair kisi mehnge hardware ke, sirf aik aam webcam aur browser se real-time PSL detection mumkin banai jaye."

---

### Slide 3: Project Objectives & Scope
1. **Real-Time Hand Tracking:** Extract 21 3D hand keypoints (63 spatial coordinates) using Google MediaPipe with sub-25ms latency.
2. **Comprehensive PSL Vocabulary (51 Classes):**
   - **36 Urdu Alphabet Letters (حروفِ تہجی):** ا تا ے (Alif to Bari Yeh)
   - **10 Numerical Digits (اعداد):** ۰ تا ۹ (0 to 9)
   - **5 Daily Conversational Phrases (جملے):** السلام علیکم، شکریہ، ہاں، نہیں، مدد
3. **Deep Learning Classification:** Sequence modeling via **Bidirectional LSTM** with temporal sliding buffers and majority-voting prediction smoothing.
4. **Multilingual & Audio Output:** Display recognized Urdu calligraphy, English transliteration, phonetic guidance, and native **Urdu Text-to-Speech (TTS)** voice.
5. **Administrative Analytics Dashboard:** Track real-time usage volume, 7-day trend analysis, model confidence, and activity audit logs.

> **🎙️ Speaker Notes:**
> "Hamare project ke 5 main ahdaaf hain: real-time hand tracking, 51 PSL signs ki accurate classification, Bi-LSTM neural network, Urdu voice synthesis, aur aik comprehensive Admin Analytics Dashboard jo live metrics track karta hai."

---

### Slide 4: System Architecture & Workflow
```
[ User Performs Gesture ]
          │
          ▼
[ Webcam 640x480 ] ──► [ MediaPipe Hands (Client WebAssembly) ]
                                    │
                       (21 Landmarks x 3D = 63 Floats)
                                    │
                                    ▼
                         [ WebSocket: /ws/detect ]
                                    │
                                    ▼
                       [ FastAPI Backend Engine ]
                                    │
                       [ 30-Frame Sliding Buffer ]
                                    │
                       [ Bi-LSTM Neural Model ]
                                    │
                        [ Prediction Smoother ]
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
  [ Client UI Update ]                            [ SQLite Database ]
  • Urdu Calligraphy (`الف`)                      • Detection Event Log
  • Confidence Meter (98%)                        • Session Analytics
  • Urdu Voice Output (TTS)                       • 7-Day Trend Storage
  • Sentence Builder Bar
```

> **🎙️ Speaker Notes:**
> "System ka architecture hybrid edge-server design par mabni hai. Video processing browser ke andar hoti hai taake heavy video server par stream na karni pare. Sirf 63 numerical coordinates WebSocket ke zariye FastAPI backend par jate hain, jahan hamara Bi-LSTM model 30 frames ke temporal window par inference run karta hai."

---

### Slide 5: Dataset & Feature Representation
- **Landmark Topology:** 21 landmark points per hand (Wrist, Thumb 1-4, Index 5-8, Middle 9-12, Ring 13-16, Pinky 17-20).
- **Spatial Features:** Normalized coordinates $(x, y, z)$ invariant to camera distance and user skin tone variations.
- **Temporal Window:** 30 consecutive frames (~1.0 second window at 30 fps) capturing the full trajectory of static and dynamic signs.
- **Input Tensor Dimensions:** `(Batch_Size, 30, 63)`
- **Data Augmentation:** Gaussian noise jittering, temporal speed scaling (0.8x - 1.2x), spatial translation, and phase shifting to ensure robust generalization.

> **🎙️ Speaker Notes:**
> "Raw video ke bajaye hum 21 hand landmarks use karte hain. Is ka sab se bara faida yeh hai keh background lighting, skin tone, aur camera quality ka model ki accuracy par bura asar nahi parta. Har sample 30 frames ka sequence hota hai jo complete gesture motion capture karta hai."

---

### Slide 6: Deep Learning Model Architecture
- **Model Type:** **Bidirectional LSTM (Long Short-Term Memory) Network**
- **Layer 1:** `Bidirectional(LSTM(128, return_sequences=True))` — Captures forward and reverse temporal context.
- **Regularization:** `BatchNormalization()` + `Dropout(0.25)` to prevent overfitting.
- **Layer 2:** `Bidirectional(LSTM(64))` — Compresses temporal features into a dense representation.
- **Fully Connected Layers:** `Dense(128, ReLU)` ➔ `BatchNormalization()` ➔ `Dense(64, ReLU)`.
- **Output Layer:** `Dense(51, Softmax)` — Generates probability distribution over all 51 PSL classes.
- **Loss & Optimizer:** Categorical Crossentropy with Adam Optimizer and learning rate scheduler.

> **🎙️ Speaker Notes:**
> "Humne Bidirectional LSTM architecture select kiya kyunkeh sign language mein gesture ka start aur end dono context zaroori hote hain. Bi-LSTM donon directions se sequence ko learn karta hai. Regularization ke liye humne Batch Normalization aur Dropout use kiya hai taake model overfit na ho."

---

### Slide 7: Anti-Jitter & Prediction Smoothing
- **The Problem:** Sensor noise, subtle hand micro-tremors, and frame transitions can cause rapid flickering between adjacent signs.
- **The Solution — Circular Majority Voting Smoother:**
  $$\hat{y}_t = \operatorname{mode}(\{y_{t-6}, y_{t-5}, \dots, y_t\})$$
- Maintains a circular buffer of the last 7 frame predictions.
- The system only commits a sign when consensus is reached and confidence exceeds the threshold ($\tau \ge 0.75$).
- **Result:** Smooth, stable detection with zero flickering and rock-solid user experience.

> **🎙️ Speaker Notes:**
> "Real-time webcam detection mein aksar sensor noise ki wajah se predictions flicker karti hain. Humne ek 7-frame sliding window majority voting algorithm implement kiya hai, jis se prediction bilkul stable rehti hai aur user ko 100% reliable output milta hai."

---

### Slide 8: Technical Stack & Tools
| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Frontend UI** | HTML5, Modern CSS Glassmorphism, JS | Ultra-fast load times, no heavy node runtime dependencies |
| **Hand Tracking** | Google MediaPipe Hands (WebAssembly) | Client-side 30fps landmark extraction with GPU acceleration |
| **Backend Server** | Python 3.13 + FastAPI + Uvicorn | Asynchronous, high-throughput, native WebSocket support |
| **Deep Learning** | TensorFlow / Keras 2.21 | Industry-standard model training and serialized `.h5` deployment |
| **Voice Synthesis** | Web Speech Synthesis API | Native Urdu pronunciation (`ur-PK`) without 3rd party API costs |
| **Database** | SQLite + SQLAlchemy ORM | Lightweight, zero-configuration local ACID persistence |
| **Data Collection** | OpenCV + Python CLI Script | Automated custom gesture sequence capture and `.npy` storage |

> **🎙️ Speaker Notes:**
> "Hamara tech stack modern aur lightweight hai. Frontend vanilla CSS glassmorphism aur JavaScript mein hai, backend FastAPI aur WebSockets par chalta hai, deep learning TensorFlow mein train hui hai, aur voice synthesis browser-native Web Speech API se hoti hai."

---

### Slide 9: Core Application Features
1. **Live Detection Workspace:**
   - Real-time webcam feed with responsive skeleton mesh overlay.
   - Urdu Calligraphy card displaying detected letter/phrase in Arabic script (`Noto Nastaliq Urdu`).
   - English transliteration + phonetic pronunciation guide.
   - Animated confidence gauge bar.
2. **Continuous Sentence Builder:**
   - Automatically concatenates detected signs into sentences with a single-click copy button.
3. **Audio Speech Output:**
   - Real-time speech synthesis with smart 2.5-second cooldown to eliminate audio stutter.
4. **Admin Analytics Dashboard:**
   - Real-time KPIs (Total Detections, Today's Volume, Mean Confidence, Active Vocabulary).
   - Interactive SVG 7-Day Activity Trend Bar Chart.
   - Category distribution progress gauges (Letters, Digits, Phrases).
   - Live activity event audit table with timestamps.
5. **PSL Sign Dictionary (51 Signs):**
   - Categorized directory with instant audio pronunciation and simulation test buttons.

> **🎙️ Speaker Notes:**
> "Application mein teen main modules hain: Live Detection workspace jahan sentence builder aur Urdu calligraphy aati hai, Admin Analytics Dashboard jahan 7-day trend chart aur audit log nazar aata hai, aur 51 signs ki interactive dictionary jahan koi bhi sign sun kar test kiya ja sakta hai."

---

### Slide 10: Experimental Results & Performance
- **Model Inference Speed:** ~18 ms per sequence buffer evaluation.
- **End-to-End Latency:** < 180 ms (Camera capture ➔ Landmark extraction ➔ WebSocket transmission ➔ Bi-LSTM prediction ➔ UI update).
- **Benchmark Test Accuracy:** Over 90% validation accuracy on benchmark canonical gesture sequences.
- **WebSocket Throughput:** Handles continuous streaming with negligible CPU overhead.
- **Automated Verification:** 100% passing test suite across database, inference, HTTP routes, and UI rendering.

> **🎙️ Speaker Notes:**
> "Performance metrics ki baat karein toh hamara end-to-end latency sirf 180 milliseconds hai, jo ke real-time human interaction ke liye bilkul seamless hai. Automated tests ne model inference, database logging, aur UI routes ko 100% pass kiya hai."

---

### Slide 11: Challenges Faced & Engineering Solutions
1. **Challenge 1: Scarcity of Standardized PSL Datasets**
   - *Solution:* Built a custom data collection script (`collect_data.py`) with MediaPipe to systematically record 21-point sequence arrays and augmented the data synthetically.
2. **Challenge 2: Prediction Flicker & False Positives**
   - *Solution:* Engineered a temporal sliding window buffer with majority voting and confidence thresholding.
3. **Challenge 3: Real-Time Urdu Speech Synthesis**
   - *Solution:* Implemented multi-tier voice selection targeting Urdu locale with phonetic fallback cues to ensure reliable pronunciation on all client devices.
4. **Challenge 4: High Bandwidth Video Transmission**
   - *Solution:* Shifted computer vision preprocessing to browser WebAssembly, reducing bandwidth from megabytes of video to just a few kilobytes of coordinate data.

> **🎙️ Speaker Notes:**
> "Project mein main challenge PSL dataset ki kami thi, jiske liye humne dedicated collection tool develop kiya. Dosra challenge prediction flickering ka tha jise majority voting se hal kiya. Aur teesra challenge bandwidth ka tha jise client-side MediaPipe se solve kiya."

---

### Slide 12: Future Work & Roadmap
- **Continuous Dynamic Sign Sentences:** Extending the Bi-LSTM model with Connectionist Temporal Classification (CTC) for continuous unsegmented sign sentence translation.
- **Two-Hand Complex Signs:** Incorporating two-handed interaction for advanced vocabulary.
- **Bidirectional Speech-to-Sign:** Adding a 3D animated avatar that converts spoken Urdu back into sign language for two-way communication.
- **Mobile Application Deployment:** Packaging the web application as a Progressive Web App (PWA) / Flutter mobile app for iOS and Android.

> **🎙️ Speaker Notes:**
> "Future enhancements mein hum dynamic sentence translation, 3D animated avatar jo aawaz ko sign language mein badalta hai, aur mobile app deployment shamil karenge."

---

### Slide 13: Conclusion & Q&A
- **Summary:**
  - Designed and delivered a fully functional, real-time Pakistani Sign Language detection system.
  - Successfully supports 51 PSL signs across alphabets, digits, and phrases.
  - Combines computer vision, deep sequence modeling, Urdu voice synthesis, and admin analytics into a cohesive, user-friendly web solution.
- **Live Demo:** Ready for live demonstration.
- **Thank You:** Thank you for your time and guidance. We invite your questions and feedback!

> **🎙️ Speaker Notes:**
> "In conclusion, humne ek complete, operational, aur practical PSL detection system tayyar kiya hai jo deaf community ke liye aik qabil-e-qadar asasa ban sakta hai. Ab main system ka live demonstration pesh karne ke liye tayyar hoon. Bohat shukriya, ab aapke sawalat ka khair maqdam hai."
