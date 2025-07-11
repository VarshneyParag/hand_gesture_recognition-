import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 🔧 Configuration
DATA_DIR = r"C:/Users/parag/OneDrive/Desktop/Prodigy_Infotech/Hand_gesture/data"
SEQ_LEN = 10
IMG_SIZE = 64
NUM_CLASSES = 10

# 📥 Load and process nested sequences
def load_sequences():
    X_seq, y_seq = [], []
    class_folders = sorted(os.listdir(DATA_DIR))

    for label_idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(DATA_DIR, class_folder)
        if not os.path.isdir(class_path):
            continue

        sequence_folders = sorted(os.listdir(class_path))

        for seq_folder in sequence_folders:
            seq_path = os.path.join(class_path, seq_folder)
            images = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            print(f"📁 Loading: {seq_path} - {len(images)} images")

            if len(images) < SEQ_LEN:
                print(f"⚠️ Skipping {seq_path} (not enough images)")
                continue

            frames = []
            for fname in images[:SEQ_LEN]:  # Ensure fixed length
                img_path = os.path.join(seq_path, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"❌ Error loading: {img_path}")
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                frames.append(img)

            if len(frames) == SEQ_LEN:
                X_seq.append(frames)
                y_seq.append(label_idx)

    if not X_seq:
        raise ValueError("🚫 No valid sequences found. Check folder structure.")

    X_seq = np.array(X_seq).reshape(-1, SEQ_LEN, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    y_seq = to_categorical(y_seq, num_classes=NUM_CLASSES)
    return train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 🧠 CNN-LSTM model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 🚀 Training
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_sequences()
    print(f"✅ Loaded: {X_train.shape} | Labels: {y_train.shape}")

    model = build_model(X_train.shape[1:], y_train.shape[1])
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

    model.save('../models/cnn_lstm_model.h5')
    print("✅ CNN-LSTM model trained and saved as cnn_lstm_model.h5")
