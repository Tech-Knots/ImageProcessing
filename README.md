
# 🧠 Image Processing using OpenCV and KNN (User-Trained)

This project is a hands-on implementation of **image classification** using Python, OpenCV, and a K-Nearest Neighbors (KNN) model. It’s fully customizable — you train it with your own images and use it in real-time or with static image prediction.

---

## 🚀 Features
- 🔍 Train your own image classifier using grayscale images
- 📦 Supports multi-class image training from custom folders
- 🖼️ Predicts a single test image using `predict.py`
- 🎥 Real-time object prediction with webcam via `webcam_predict.py`
- 💡 Simple and beginner-friendly — no deep learning needed
- 🧠 Built with scikit-learn’s KNN (k=1) algorithm

---

## 🧾 Project Structure

```
object_Detection/
├── dataset/             # Your training images organized into folders by label
│   ├── Apple/
│   ├── Banana/
│   └── etc.
├── load_Data.py         # Trains the model from dataset/
├── predict.py           # Predicts label from a test image
├── webcam_predict.py    # Real-time webcam prediction
├── knn_model.pkl        # Saved model after training
└── test.jpg             # Sample test image
```

---

## 🛠 Requirements

Install dependencies:

```bash
pip install opencv-python scikit-learn numpy joblib
```

---

## 🧠 How to Use

### 1️⃣ Prepare Your Dataset
Structure it like:
```
dataset/
├── Class1/
│   ├── img1.jpg
│   └── img2.jpg
├── Class2/
│   └── img1.jpg
```

### 2️⃣ Train the Model
```bash
python load_Data.py
```

### 3️⃣ Predict from Image
```bash
python predict.py
```

### 4️⃣ Predict from Webcam
```bash
python webcam_predict.py
```

---

## 🎯 Use Cases

✅ Educational Projects  
✅ Quick Demos in AI/ML  
✅ Real-Time Object Labeling  
✅ Workshop or Classroom Training  

---

## 👨‍💻 Developed By

**RishiKumar S.**  
🔗 [Tech-Knots GitHub](https://github.com/Tech-Knots)  
📅 Created: July 31, 2025

---

## 📃 License

MIT License - Free for personal and academic use.
