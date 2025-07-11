from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from data_loader import load_data  # Make sure this file exists and is implemented

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # ✅ Fix: Correct variable name
    data_path = "C:/Users/parag/OneDrive/Desktop/Prodigy_Infotech/Hand_gesture/data"
    X_train, X_test, y_train, y_test = load_data(data_path)

    model = build_model(X_train.shape[1:], y_train.shape[1])
    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

    model.save('../models/cnn_model.h5')
    print("✅ Model trained and saved as cnn_model.h5")
