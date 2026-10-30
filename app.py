
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH = "models/medical_image_cnn.keras"
model = tf.keras.models.load_model(MODEL_PATH)

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Medical Image Analysis</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #eef4fb;
            text-align: center;
            padding: 35px 15px;
        }
        .card {
            background: white;
            max-width: 520px;
            margin: auto;
            padding: 28px;
            border-radius: 16px;
            box-shadow: 0 4px 18px #0001;
        }
        h1 { color: #174a7e; }
        input, button { margin: 14px; padding: 10px; }
        button {
            background: #1769aa;
            color: white;
            border: none;
            border-radius: 7px;
            cursor: pointer;
        }
        .result {
            background: #e5f4e8;
            padding: 15px;
            border-radius: 9px;
        }
        .warning { color: #9b3030; font-size: 14px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>AI Medical Image Analysis</h1>
        <p>Upload a chest X-ray image for an educational model prediction.</p>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <br>
            <button type="submit">Analyze Image</button>
        </form>
        {% if result %}
        <div class="result">
            <h2>Model Output: {{ result }}</h2>
            <p>Model score: {{ score }}%</p>
        </div>
        {% endif %}
        {% if error %}
        <p>{{ error }}</p>
        {% endif %}
        <p class="warning">
            Educational prototype only. This is not a medical diagnosis.
            Do not use it to make healthcare decisions.
        </p>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    score = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not file.filename:
            error = "Please select an image."
        else:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                error = "Could not read that image. Please upload a valid image."
            else:
                image = cv2.resize(image, (256, 256))
                image = image.astype("float32") / 255.0
                image = np.expand_dims(image, axis=(0, -1))

                probability = float(model.predict(image, verbose=0)[0][0])

                if probability >= 0.5:
                    result = "PNEUMONIA"
                    score = f"{probability * 100:.2f}"
                else:
                    result = "NORMAL"
                    score = f"{(1 - probability) * 100:.2f}"

            if os.path.exists(path):
                os.remove(path)

    return render_template_string(
        PAGE, result=result, score=score, error=error
    )

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
