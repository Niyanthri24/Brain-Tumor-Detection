import os
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import cv2 # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Set parameters
img_size = 150
data_path = "dataset/"

# Load data
def load_data():
    images, labels = [], []
    for label in ["yes", "no"]:
        path = os.path.join(data_path, label)
        for img_name in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img_name))
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(1 if label == "yes" else 0)
            except:
                pass
    return np.array(images), np.array(labels)

X, y = load_data()
X = X / 255.0  # Normalize

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Augment training data
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                             width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, fill_mode="nearest")
datagen.fit(X_train)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=15, validation_data=(X_test, y_test))

# Evaluate
print(f"Test Accuracy: {model.evaluate(X_test, y_test)[1] * 100:.2f}%")
y_pred = model.predict(X_test) > 0.5
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
model.save("brain_tumor_detector.h5")

# Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()
plt.show()
