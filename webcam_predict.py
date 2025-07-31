import cv2
import joblib
import numpy as np

# Load the trained KNN model
model = joblib.load('knn_model.pkl')

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Resize the frame to match model input (e.g., 100x100 if model trained with that)
    resized = cv2.resize(frame, (100, 100))  # Match your training input size
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Flatten and reshape
    img_flat = gray.flatten().reshape(1, -1)

    # Make prediction
    try:
        prediction = model.predict(img_flat)
        label = prediction[0]
    except Exception as e:
        label = f"Error: {e}"

    # Display label on video
    cv2.putText(frame, f'Prediction: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('KNN Prediction', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
