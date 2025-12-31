from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = tf.keras.models.load_model("plant_disease_model.h5")

class_names = [
    "Pepper__bell__Bacterial_spot",
    "Pepper__bell__healthy"
]

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    index = np.argmax(predictions[0])

    disease = class_names[index]
    confidence = predictions[0][index]

    return disease, confidence

def treatment_advice(disease):
    disease = disease.lower()

    if "bacterial" in disease:
        return "Use copper-based fungicide. Remove infected leaves. Avoid overhead irrigation."
    elif "healthy" in disease:
        return "Plant is healthy 🌱 Continue proper watering and fertilization."
    else:
        return "Consult an agriculture expert."

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    disease, confidence = predict_disease(file_path)
    treatment = treatment_advice(disease)

    return render_template(
        "result.html",
        disease=disease,
        confidence=round(confidence * 100, 2),
        treatment=treatment,
        image=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)

