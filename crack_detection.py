import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_images(dataset_path):
    images = []
    labels = []
    categories = {"Positive": 1, "Negative": 0}
    for category, label in categories.items():
        folder_path = os.path.join(dataset_path, category, "Images")
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist. Skipping.")
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                img = img.flatten().astype(np.float32) / 255.0
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

dataset_path = r"C:\Users\ICTD\PycharmProjects\PythonProject\kaggle_datasets"
X, y = load_images(dataset_path)

print(f"Loaded {len(X)} images with labels: {set(y)}")
print(f"Label distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training SVM model...")
model = SVC(kernel="rbf", class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Unable to read image"

    img = cv2.resize(img, (100, 100))
    img = img.flatten().astype(np.float32) / 255.0
    img = img.reshape(1, -1)
    prediction = model.predict(img)

    return "Crack" if prediction[0] == 1 else "Non-Crack"

test_image_path = r"C:\Users\ICTD\PycharmProjects\PythonProject\test_images\test_006.jpg"
prediction = predict_image(test_image_path, model)
print(f"Prediction for {test_image_path}: {prediction}")