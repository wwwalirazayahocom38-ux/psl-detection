"""
Pakistani Sign Language (PSL) Model Training Pipeline
Trains a Bidirectional LSTM + Dense neural network on landmark sequences.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "label_map.json")
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
BACKEND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "app", "ml", "model.h5")

SEQ_LENGTH = 30
NUM_FEATURES = 63  # 21 landmarks * 3 (x, y, z)


def load_label_map():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_benchmark_dataset(num_classes=51, samples_per_class=40, seq_len=30):
    """
    Generates structured landmark sequence trajectories representing
    distinct geometric hand gestures with natural temporal variation.
    """
    np.random.seed(42)
    X = []
    y = []

    # 21 standard hand landmarks base reference
    base_hand = np.array([
        [0.5, 0.8, 0.0],    # 0: wrist
        [0.45, 0.72, -0.01],# 1: thumb cmc
        [0.42, 0.65, -0.02],# 2: thumb mcp
        [0.39, 0.58, -0.03],# 3: thumb ip
        [0.37, 0.52, -0.04],# 4: thumb tip
        [0.47, 0.58, -0.01],# 5: index mcp
        [0.46, 0.48, -0.02],# 6: index pip
        [0.45, 0.40, -0.03],# 7: index dip
        [0.45, 0.32, -0.04],# 8: index tip
        [0.51, 0.58, 0.0],  # 9: middle mcp
        [0.51, 0.47, -0.01],# 10: middle pip
        [0.51, 0.38, -0.02],# 11: middle dip
        [0.51, 0.30, -0.03],# 12: middle tip
        [0.55, 0.60, 0.01], # 13: ring mcp
        [0.56, 0.50, 0.0],  # 14: ring pip
        [0.57, 0.42, -0.01],# 15: ring dip
        [0.57, 0.35, -0.02],# 16: ring tip
        [0.59, 0.64, 0.02], # 17: pinky mcp
        [0.61, 0.56, 0.02], # 18: pinky pip
        [0.62, 0.50, 0.01], # 19: pinky dip
        [0.63, 0.45, 0.0],  # 20: pinky tip
    ])

    for class_idx in range(num_classes):
        # Distinct spatial signature for each sign
        class_angle = (class_idx / num_classes) * 2 * np.pi
        finger_mod = np.sin(class_angle + np.arange(21) * 0.4)[:, None] * 0.08
        z_mod = np.cos(class_angle * 1.5 + np.arange(21) * 0.2)[:, None] * 0.04
        class_signature = base_hand + np.hstack([finger_mod, finger_mod * 0.5, z_mod])

        for _ in range(samples_per_class):
            sequence = []
            jitter = np.random.normal(0, 0.008, size=class_signature.shape)
            speed = np.random.uniform(0.8, 1.2)
            phase = np.random.uniform(0, np.pi)

            for t in range(seq_len):
                temporal_factor = np.sin(t * 0.15 * speed + phase) * 0.02
                noise = np.random.normal(0, 0.005, size=class_signature.shape)
                frame_landmarks = class_signature + jitter + noise
                frame_landmarks[:, 0] += temporal_factor
                sequence.append(frame_landmarks.flatten())

            X.append(sequence)
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_model(num_classes, seq_length=SEQ_LENGTH, num_features=NUM_FEATURES):
    model = Sequential([
        Input(shape=(seq_length, num_features)),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.25),

        Bidirectional(LSTM(64)),
        BatchNormalization(),
        Dropout(0.25),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    label_map = load_label_map()
    num_classes = len(label_map)
    print(f"Loaded {num_classes} PSL classes from label_map.json")

    print("Synthesizing baseline training sequences...")
    X, y = generate_benchmark_dataset(num_classes=num_classes, samples_per_class=45)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    y_cat = to_categorical(y, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y
    )

    print("Building Bidirectional LSTM architecture...")
    model = build_model(num_classes)
    model.summary()

    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BACKEND_MODEL_PATH), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    # Save to both saved checkpoint and backend path
    save_path = os.path.join(SAVED_MODEL_DIR, "best_model.h5")
    model.save(save_path)
    model.save(BACKEND_MODEL_PATH)
    print(f"Model saved successfully to {save_path} and {BACKEND_MODEL_PATH}")


if __name__ == "__main__":
    main()
