import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
# Folder where your images are stored
data_dir = 'dataset'

# Lists to store data and labels
X = []
y = []

# Resize all images to 100x100
img_size = (100, 100)

# Go through each class (folder)
for label in os.listdir(data_dir):
    class_folder = os.path.join(data_dir, label)

    # Go through each image in the class folder
    for file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, file)

        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize image to fixed size
        img = cv2.resize(img, img_size)

        # Flatten image (100x100 → 10,000)
        features = img.flatten()

        # Store features and label
        X.append(features)
        y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)
# Create KNN model (k = 1 means nearest neighbor)
model = KNeighborsClassifier(n_neighbors=1)

# Train the model with image data and labels
model.fit(X, y)
joblib.dump(model, "knn_model.pkl")
print("✅ Model training complete!")

print("Feature extraction completed!")
print("Total samples:", len(X))
print("Labels:", np.unique(y))
