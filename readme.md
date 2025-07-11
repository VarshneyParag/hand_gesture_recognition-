👨‍💻 Author
Parag Varshney
🎓 B.Tech (AI & ML) | 💡 Machine Learning Enthusiast
🔗 GitHub: @VarshneyParag
🔗 LinkedIn: parag-varshney

🌟 Credits
Dataset: Custom labeled dataset structured manually.
Libraries: TensorFlow, OpenCV, scikit-learn, numpy, matplotlib.

📜 License
This project is licensed under the MIT License.

# 🖐️ Hand Gesture Recognition using CNN

A Deep Learning-based project for recognizing hand gestures from images using Convolutional Neural Networks (CNN). This project utilizes a custom dataset of labeled hand gestures to train a model capable of classifying gestures in real-time for use in gesture-based control systems, accessibility tools, or HCI (Human-Computer Interaction) applications.

---

## 🚀 Project Overview

This repository contains code to:
- Load and preprocess a structured hand gesture image dataset.
- Train a CNN on gesture classes (`palm`, `fist`, `thumb`, `ok`, etc.).
- Evaluate performance on test data.
- Prepare the model for integration with real-time webcam-based recognition.

---

## 🧠 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Input Shape**: Grayscale images resized to `64x64`
- **Output**: Multi-class classification over 10 gesture types
