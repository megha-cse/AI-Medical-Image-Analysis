
import os
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. PROJECT SETTINGS

train_dir = "data/chest_xray/train"
val_dir = "data/chest_xray/val"
test_dir = "data/chest_xray/test"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 5

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 2. LOAD DATASET

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_data.class_names

print("Class mapping:", {
    name: i for i, name in enumerate(class_names)
})

# 3. NORMALIZE IMAGES

normalization = layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda x, y: (normalization(x), y)
).prefetch(tf.data.AUTOTUNE)

val_data = val_data.map(
    lambda x, y: (normalization(x), y)
).prefetch(tf.data.AUTOTUNE)

test_data = test_data.map(
    lambda x, y: (normalization(x), y)
).prefetch(tf.data.AUTOTUNE)

# 4. BUILD CNN MODEL

model = models.Sequential([
    layers.Input(shape=(256, 256, 1)),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")
model.summary()

# 5. TRAIN THE MODEL

print("\nStarting model training...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model.save("models/medical_image_cnn.keras")

print("\nModel saved successfully!")

# 6. EVALUATE MODEL

print("\nEvaluating on test dataset...")

test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 7. GET PREDICTIONS

y_true = []
y_pred = []

for images, labels in test_data:
    probabilities = model.predict(images, verbose=0)

    predictions = (
        probabilities >= 0.5
    ).astype("int32").flatten()

    y_true.extend(
        labels.numpy().flatten().astype("int32")
    )

    y_pred.extend(predictions)

print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0
))

# 8. SAVE CONFUSION MATRIX

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", values_format="d")

plt.title("Chest X-Ray Classification")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

print("\nConfusion matrix saved!")

# 9. SAVE ACCURACY GRAPH

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/accuracy_graph.png", dpi=300)
plt.close()

print("Accuracy graph saved!")

# 10. SAVE LOSS GRAPH

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/loss_graph.png", dpi=300)
plt.close()

print("Loss graph saved!")

# 11. COMPLETION

print("\nProject training and evaluation completed!")

print("Check the models and outputs folders.")

print("\nGenerated files:")
print("- models/medical_image_cnn.keras")
print("- outputs/confusion_matrix.png")
print("- outputs/accuracy_graph.png")
print("- outputs/loss_graph.png")
