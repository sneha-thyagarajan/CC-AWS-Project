import io
import os
import base64
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)
app.config["DETECTED_DIR"] = os.path.join("static", "detected")
os.makedirs(app.config["DETECTED_DIR"], exist_ok=True)

MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
DETECTION_THRESHOLD = 0.4
lock = threading.Lock()

COCO_LABELS = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print("🔄 Loading TensorFlow SSD model ...")
detector = hub.load(MODEL_URL)
print("✅ Model loaded successfully!")

def pil_from_bytes(image_bytes: bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def annotate_image_pil(image_pil: Image.Image, boxes, classes, scores):
    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except:
        font = ImageFont.load_default()

    for box, cls, score in zip(boxes, classes, scores):
        if score < DETECTION_THRESHOLD:
            continue
        ymin, xmin, ymax, xmax = box
        left, top, right, bottom = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        label = COCO_LABELS[cls] if 0 <= cls < len(COCO_LABELS) else str(cls)
        caption = f"{label}: {score:.2f}"
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([(left, top), (right, bottom)], outline="#00C3FF", width=3)
        draw.rectangle([(left, top - text_h - 4), (left + text_w + 4, top)], fill="#00C3FF")
        draw.text((left + 2, top - text_h - 2), caption, fill="white", font=font)
    return image_pil

def run_detector_on_numpy(image_np):
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)[tf.newaxis, ...]
    with lock:
        outputs = detector(input_tensor)
    boxes = outputs['detection_boxes'][0].numpy()
    scores = outputs['detection_scores'][0].numpy()
    classes = outputs['detection_classes'][0].numpy().astype(np.int32)
    return boxes, classes, scores

def pil_to_jpeg_bytes(pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect_upload", methods=["POST"])
def detect_upload():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.mimetype.startswith("image/"):
            return jsonify({"error": "Uploaded file is not an image"}), 400

        image_bytes = file.read()
        image_pil = pil_from_bytes(image_bytes)
        image_pil.thumbnail((800, 800))
        image_np = np.array(image_pil)
        boxes, classes, scores = run_detector_on_numpy(image_np)
        annotated = annotate_image_pil(image_pil, boxes, classes, scores)

        # Save result to history folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detected_{timestamp}.jpg"
        save_path = os.path.join(app.config["DETECTED_DIR"], filename)
        annotated.save(save_path)

        buf = pil_to_jpeg_bytes(annotated)
        return send_file(buf, mimetype="image/jpeg")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    data_url = request.json.get("image")
    if not data_url:
        return jsonify({"error": "No image data"}), 400
    header, encoded = data_url.split(",", 1)
    frame_bytes = base64.b64decode(encoded)
    image_pil = pil_from_bytes(frame_bytes)
    image_np = np.array(image_pil)
    boxes, classes, scores = run_detector_on_numpy(image_np)
    annotated = annotate_image_pil(image_pil, boxes, classes, scores)
    buf = pil_to_jpeg_bytes(annotated)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/get_detected_images", methods=["GET"])
def get_detected_images():
    """Return list of detected images (latest first)"""
    images = sorted(os.listdir(app.config["DETECTED_DIR"]), reverse=True)
    urls = [url_for('static', filename=f"detected/{img}") for img in images]
    return jsonify({"images": urls})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
