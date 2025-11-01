import io
import base64
import threading
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# ---------------- CONFIG ----------------
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

print("🔄 Loading TensorFlow SSD model (this may take ~1 minute)...")
detector = hub.load(MODEL_URL)
print("✅ Model loaded successfully!")

# ---------------- Helpers ----------------
def pil_from_bytes(image_bytes: bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def annotate_image_pil(image_pil: Image.Image, boxes, classes, scores, threshold=DETECTION_THRESHOLD):
    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except:
        font = ImageFont.load_default()

    for box, cls, score in zip(boxes, classes, scores):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        left, top, right, bottom = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        label = COCO_LABELS[cls] if 0 <= cls < len(COCO_LABELS) else str(cls)
        caption = f"{label}: {score:.2f}"

        # Draw bounding box
        draw.rectangle([(left, top), (right, bottom)], outline="#00C3FF", width=3)
        text_w, text_h = draw.textsize(caption, font=font)
        draw.rectangle([(left, top - text_h - 4), (left + text_w + 4, top)], fill="#00C3FF")
        draw.text((left + 2, top - text_h - 2), caption, fill="white", font=font)
    return image_pil

def run_detector_on_numpy(image_np: np.ndarray):
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)[tf.newaxis, ...]
    with lock:
        outputs = detector(input_tensor)
    boxes = outputs['detection_boxes'][0].numpy()
    scores = outputs['detection_scores'][0].numpy()
    classes = outputs['detection_classes'][0].numpy().astype(np.int32)
    return boxes, classes, scores

def pil_to_jpeg_bytes(pil_img: Image.Image, quality=85):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect_upload", methods=["POST"])
def detect_upload():
    try:
        # 1️⃣ Ensure file present
        if "image" not in request.files:
            print("⚠️ No 'image' key in request.files")
            return jsonify({"error": "No file field named 'image'"}), 400

        file = request.files["image"]

        # 2️⃣ Ensure valid file
        if not file or file.filename == "":
            print("⚠️ Empty file or filename")
            return jsonify({"error": "No file selected"}), 400

        # 3️⃣ Ensure correct mimetype
        if not file.mimetype.startswith("image/"):
            print(f"⚠️ Invalid mimetype: {file.mimetype}")
            return jsonify({"error": "Uploaded file is not an image"}), 400

        # 4️⃣ Read bytes once only
        image_bytes = file.read()
        if not image_bytes:
            print("⚠️ Empty file bytes")
            return jsonify({"error": "Empty upload"}), 400

        # 5️⃣ Verify with Pillow safely
        try:
            image_pil = Image.open(io.BytesIO(image_bytes))
            image_pil.verify()  # check integrity first
        except Exception as e:
            print("❌ Pillow verify failed:", e)
            return jsonify({"error": "Invalid image file"}), 400

        # 6️⃣ Reload the image (verify() closes file)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Optional resize to avoid TensorFlow OOM
        image_pil.thumbnail((800, 800))

        image_np = np.array(image_pil)
        boxes, classes, scores = run_detector_on_numpy(image_np)
        annotated = annotate_image_pil(image_pil, boxes, classes, scores)
        buf = pil_to_jpeg_bytes(annotated)
        print("✅ Uploaded image processed successfully")
        return send_file(buf, mimetype="image/jpeg")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    if "frame" in request.files:
        frame_bytes = request.files["frame"].read()
    else:
        data_url = request.json.get("image") if request.is_json else request.form.get("image")
        if not data_url:
            return jsonify({"error": "No image data"}), 400
        header, encoded = data_url.split(",", 1)
        frame_bytes = base64.b64decode(encoded)
    image_pil = pil_from_bytes(frame_bytes)
    image_np = np.array(image_pil)
    boxes, classes, scores = run_detector_on_numpy(image_np)
    annotated = annotate_image_pil(image_pil, boxes, classes, scores)
    buf = pil_to_jpeg_bytes(annotated, quality=70)
    return send_file(buf, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
