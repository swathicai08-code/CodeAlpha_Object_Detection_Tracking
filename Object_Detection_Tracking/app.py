from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():

    file = request.files["image"]

    input_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(input_path)

    image = cv2.imread(input_path)

    results = model(image)

    annotated = results[0].plot()

    output_path = os.path.join(
        RESULT_FOLDER,
        file.filename
    )

    cv2.imwrite(output_path, annotated)

    return render_template(
        "index.html",
        result_image=output_path
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)