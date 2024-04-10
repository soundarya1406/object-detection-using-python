import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Load the pre-trained YOLOv3 model
model = load_model('yolov3.h5')

# Load the class labels
with open('coco_classes.txt', 'r', encoding='utf-8') as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

# Function to perform object detection on an image
def detect_objects(image):
    # Preprocess the image
    image = cv2.resize(image, (416, 416))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Perform object detection
    boxes, scores, classes = model.predict(image)

    # Display the detected objects
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x, y, w, h = boxes[i]
            class_id = int(classes[i])
            label = class_names[class_id]

            # Draw bounding box and label
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Define the image path
image_path = r"C:\Users\sound\Downloads\sharktank (19201080).jpg"

# Load test image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Perform object detection
result = detect_objects(image)

# Display the result
plt.imshow(result)
plt.axis('off')
plt.show()
