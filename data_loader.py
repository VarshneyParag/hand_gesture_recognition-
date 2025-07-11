import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, img_size=64):
    X, y = [], []
    gesture_folders = sorted(os.listdir(data_dir))  # 00 to 09
    label_map = {folder: idx for idx, folder in enumerate(gesture_folders)}

    for folder in gesture_folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for gesture_class in os.listdir(folder_path):  # 01_palm, 02_l, etc.
            gesture_path = os.path.join(folder_path, gesture_class)
            if not os.path.isdir(gesture_path):
                continue

            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label_map[folder])

    X = np.array(X).reshape(-1, img_size, img_size, 1).astype('float32') / 255.0
    y = to_categorical(np.array(y), num_classes=len(label_map))

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Test run block
if __name__ == "__main__":
    data_dir = "C:/Users/parag/OneDrive/Desktop/Prodigy_Infotech/Hand_gesture/data"
    X_train, X_test, y_train, y_test = load_data(data_dir)

    print("✅ Dataset loaded successfully!")
    print(f"📦 Total Samples: {len(X_train) + len(X_test)}")
    print(f"🧠 Training Samples: {len(X_train)}")
    print(f"🧪 Testing Samples: {len(X_test)}")
    print(f"🖼️ Input Shape: {X_train.shape[1:]}")
    print(f"🔢 Output Shape: {y_train.shape[1:]}")
