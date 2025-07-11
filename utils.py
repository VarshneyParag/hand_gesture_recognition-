import matplotlib.pyplot as plt

def plot_metrics(history, save_path=None, dark_mode=False):
    """Plot training accuracy and loss curves."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    plt.figure(figsize=(10, 5))

    # 📈 Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy', color='green', linewidth=2)
    plt.plot(val_acc, label='Val Accuracy', color='orange', linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 📉 Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss', color='red', linewidth=2)
    plt.plot(val_loss, label='Val Loss', color='blue', linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📁 Plot saved to: {save_path}")
    plt.show()

# ✅ Self-test block (optional)
if __name__ == '__main__':
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    import numpy as np

    model = Sequential([
        Dense(10, activation='relu', input_shape=(5,)),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Dummy dataset
    X = np.random.rand(100, 5)
    y = np.zeros((100, 2))
    y[np.arange(100), np.random.randint(0, 2, 100)] = 1

    history = model.fit(X, y, validation_split=0.2, epochs=5)
    plot_metrics(history)
