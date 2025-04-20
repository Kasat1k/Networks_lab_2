import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import filedialog

# --- 1. Завантаження MNIST ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (28,28) → (28,28,1)
x_test = np.expand_dims(x_test, -1)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# --- 2. Побудова моделі CNN ---
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# --- 3. Навчання моделі (на одній комбінації для демонстрації) ---
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=3, batch_size=128, validation_split=0.1)

# --- 4. Оцінка моделі ---
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"\nТочність на тестових даних: {test_acc:.4f}")

# --- 5. Матриця плутанини ---
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
plt.title("Матриця плутанини")
plt.show()

# --- 6. Класифікація довільного зображення через діалог ---
def predict_custom_image_via_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Виберіть зображення рукописної цифри",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if not file_path:
        print("⚠️ Файл не вибрано.")
        return

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f" Не вдалося відкрити файл: {file_path}")
        return

    img = cv2.resize(img, (28, 28))
    img = 255 - img  # інверсія, бо MNIST — білі цифри на чорному
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, 28, 28, 1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"\n📷 Обране зображення: {file_path}")
    print(f"✅ Передбачено: {predicted_class} з довірою {confidence:.2f}")

    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title(f"Передбачення: {predicted_class}")
    plt.axis('off')
    plt.show()

# --- 7. Виклик вибору довільного зображення ---
predict_custom_image_via_dialog()
