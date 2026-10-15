"""
Pakistani Sign Language (PSL) Live Data Collection Script
Captures hand landmarks from webcam using MediaPipe and saves sequences as NumPy arrays (.npy)
"""
import os
import json
import time
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "label_map.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SEQ_LENGTH = 30
SAMPLES_PER_SIGN = 30


def load_classes():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(k, v["english"], v["urdu"]) for k, v in data.items()]


def collect_gestures(target_class_id=None):
    classes = load_classes()
    if target_class_id is not None:
        classes = [c for c in classes if c[0] == str(target_class_id)]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        for class_id, english_name, urdu_name in classes:
            class_folder = os.path.join(DATA_DIR, f"{class_id}_{english_name}")
            os.makedirs(class_folder, exist_ok=True)

            existing_samples = len([f for f in os.listdir(class_folder) if f.endswith(".npy")])
            sample_idx = existing_samples

            print(f"\n==========================================")
            print(f"Preparing to record: {english_name} ({urdu_name})")
            print(f"Target samples: {SAMPLES_PER_SIGN} (Already collected: {existing_samples})")
            print(f"Press 'r' to record sample, 's' to skip sign, 'q' to quit.")
            print(f"==========================================")

            while sample_idx < SAMPLES_PER_SIGN:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Draw landmark mesh
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                # Overlay UI
                cv2.rectangle(frame, (0, 0), (w, 80), (25, 25, 25), -1)
                cv2.putText(frame, f"Sign: {english_name} - {urdu_name}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
                cv2.putText(frame, f"Sample: {sample_idx + 1}/{SAMPLES_PER_SIGN} | Press 'R' to Record", (20, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

                cv2.imshow("PSL Data Collection - MediaPipe", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('s'):
                    break
                elif key == ord('r'):
                    # Countdown
                    for cd in range(3, 0, -1):
                        ret, c_frame = cap.read()
                        if ret:
                            c_frame = cv2.flip(c_frame, 1)
                            cv2.putText(c_frame, f"Starting in {cd}...", (w // 2 - 120, h // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                            cv2.imshow("PSL Data Collection - MediaPipe", c_frame)
                            cv2.waitKey(700)

                    # Record sequence of SEQ_LENGTH frames
                    sequence = []
                    frame_count = 0
                    while frame_count < SEQ_LENGTH:
                        ret, rec_frame = cap.read()
                        if not ret:
                            break
                        rec_frame = cv2.flip(rec_frame, 1)
                        rgb_rec = cv2.cvtColor(rec_frame, cv2.COLOR_BGR2RGB)
                        res = hands.process(rgb_rec)

                        landmarks_flat = np.zeros(63, dtype=np.float32)
                        if res.multi_hand_landmarks:
                            hand_lms = res.multi_hand_landmarks[0]
                            pts = []
                            for lm in hand_lms.landmark:
                                pts.extend([lm.x, lm.y, lm.z])
                            landmarks_flat = np.array(pts, dtype=np.float32)

                            mp_drawing.draw_landmarks(
                                rec_frame, hand_lms, mp_hands.HAND_CONNECTIONS
                            )

                        sequence.append(landmarks_flat)
                        frame_count += 1

                        cv2.circle(rec_frame, (w - 40, 40), 16, (0, 0, 255), -1)
                        cv2.putText(rec_frame, f"REC: {frame_count}/{SEQ_LENGTH}", (w - 180, 48),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("PSL Data Collection - MediaPipe", rec_frame)
                        cv2.waitKey(30)

                    # Save sequence as numpy array
                    arr = np.array(sequence, dtype=np.float32)
                    save_file = os.path.join(class_folder, f"sample_{sample_idx:03d}.npy")
                    np.save(save_file, arr)
                    print(f"Saved: {save_file}")
                    sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-id", type=int, default=None, help="Target specific class ID (0-50)")
    args = parser.parse_args()
    collect_gestures(args.class_id)
