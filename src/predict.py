
import sys
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/medical_image_cnn.keras"

if len(sys.argv) < 2:
    print('Usage: python src/predict.py "path/to/xray.jpeg"')
    sys.exit(1)

image_path = sys.argv[1]
model = tf.keras.models.load_model(MODEL_PATH)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not open the image. Check the file path.")
    sys.exit(1)

image = cv2.resize(image, (256, 256))
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=(0, -1))

probability = float(model.predict(image, verbose=0)[0][0])

if probability >= 0.5:
    print(f"Model prediction: PNEUMONIA ({probability * 100:.2f}%)")
else:
    print(f"Model prediction: NORMAL ({(1 - probability) * 100:.2f}%)")

print("Educational prototype only — not for medical diagnosis.")
