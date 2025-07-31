import cv2
import joblib
import numpy as np

# Load the trained model
model = joblib.load("knn_model.pkl")

# Load test image (change the file name if needed)
test_img_path = "test.jpg"
img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)  # Force grayscale

# Resize and flatten the image
img = cv2.resize(img, (100, 100))
img_flat = img.flatten().reshape(1, -1)  # Reshape to (1, 10000)

# Predict
prediction = model.predict(img_flat)
print("🔍 Prediction:", prediction[0])
