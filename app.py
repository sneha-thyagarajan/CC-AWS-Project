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
# Switch to more accurate model
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
DETECTION_THRESHOLD = 0.5  # Increase threshold to filter low-confidence detections
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

# ---------------- Helpers ----------------
def preprocess_image(image_pil: Image.Image):
    """Preprocess image for better detection accuracy"""
    # Resize to optimal size for the model (512x512 for EfficientDet)
    image_pil = image_pil.resize((512, 512), Image.Resampling.LANCZOS)

    # Enhance contrast and sharpness
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Sharpness(image_pil)
    image_pil = enhancer.enhance(1.1)

    return image_pil

def pil_from_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return preprocess_image(image)

def annotate_image_pil(image_pil: Image.Image, boxes, classes, scores, threshold=DETECTION_THRESHOLD):
    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size

    # Scale font size based on image size for better visibility
    base_font_size = max(24, min(w, h) // 20)  # Minimum 24, scales with image size

    # Try to load a bold font for better visibility
    font = None
    font_paths = [
        "Arial-Bold.ttf", "arial-bold.ttf", "ArialBold.ttf",
        "DejaVuSans-Bold.ttf", "dejavu-sans-bold.ttf",
        "Helvetica-Bold.ttf", "helvetica-bold.ttf",
        "Arial.ttf", "arial.ttf", "DejaVuSans.ttf"
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, size=base_font_size)
            break
        except:
            continue

    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

    # Sort detections by confidence score (highest first)
    sorted_indices = np.argsort(scores)[::-1]

    for idx in sorted_indices:
        box, cls, score = boxes[idx], classes[idx], scores[idx]
        if score < threshold:
            continue

        ymin, xmin, ymax, xmax = box
        left, top, right, bottom = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

        # Skip very small detections (likely noise)
        box_area = (right - left) * (bottom - top)
        if box_area < (w * h * 0.001):  # Less than 0.1% of image
            continue

        label = COCO_LABELS[cls] if 0 <= cls < len(COCO_LABELS) else str(cls)
        caption = f"{label}: {score:.2f}"

        # Use different colors for different confidence levels
        if score > 0.8:
            color = "#00FF00"  # Green for high confidence
            text_color = "white"
        elif score > 0.6:
            color = "#FFA500"  # Orange for medium confidence
            text_color = "black"
        else:
            color = "#FF0000"  # Red for low confidence
            text_color = "white"

        # Draw bounding box with thicker lines
        box_thickness = max(3, min(w, h) // 150)
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=box_thickness)

        # Get text dimensions using textbbox
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Add more padding around text
        padding = 10
        text_bg_left = left
        text_bg_top = top - text_h - padding * 2
        text_bg_right = left + text_w + padding * 2
        text_bg_bottom = top

        # Ensure text background doesn't go outside image bounds
        text_bg_left = max(0, text_bg_left)
        text_bg_top = max(0, text_bg_top)
        text_bg_right = min(w, text_bg_right)

        # If text would be cut off at top, place it below the box instead
        if text_bg_top < 0:
            text_bg_top = bottom
            text_bg_bottom = bottom + text_h + padding * 2
            text_bg_bottom = min(h, text_bg_bottom)

        # Draw text background
        draw.rectangle([(text_bg_left, text_bg_top), (text_bg_right, text_bg_bottom)], fill=color)

        # Draw text with outline for better visibility
        text_x = text_bg_left + padding
        text_y = text_bg_top + padding

        # Draw text outline (stroke effect) for better contrast
        outline_color = "black" if text_color == "white" else "white"
        outline_width = 2
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, text_y + dy), caption, fill=outline_color, font=font)

        # Draw main text
        draw.text((text_x, text_y), caption, fill=text_color, font=font)

    return image_pil

def run_detector_on_numpy(image_np: np.ndarray):
    # Convert to tensor and ensure proper format
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, 0)

    with lock:
        outputs = detector(input_tensor)

    # EfficientDet has different output format
    boxes = outputs['detection_boxes'][0].numpy()
    scores = outputs['detection_scores'][0].numpy()
    classes = outputs['detection_classes'][0].numpy().astype(np.int32)

    # Filter out background class (0) and very low scores
    valid_indices = (classes > 0) & (scores > 0.3)
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    classes = classes[valid_indices]

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

        # Store original size for display
        original_size = image_pil.size

        # Preprocess for detection
        processed_image = preprocess_image(image_pil.copy())
        image_np = np.array(processed_image)

        boxes, classes, scores = run_detector_on_numpy(image_np)

        # Resize back to original size for annotation
        image_pil = image_pil.resize(processed_image.size, Image.Resampling.LANCZOS)
        annotated = annotate_image_pil(image_pil, boxes, classes, scores)

        # Resize back to original size for display
        if original_size != processed_image.size:
            annotated = annotated.resize(original_size, Image.Resampling.LANCZOS)

        buf = pil_to_jpeg_bytes(annotated, quality=95)  # Higher quality
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
