
import io
import os
import base64
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for, current_app
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from werkzeug.utils import secure_filename

# --- NEW imports for S3 ---
import boto3
from botocore.exceptions import ClientError

# ---------------- App setup ----------------
app = Flask(__name__)
# local detected dir kept for compatibility (not used for saving if S3 is configured)
app.config["DETECTED_DIR"] = os.path.join("static", "detected")
os.makedirs(app.config["DETECTED_DIR"], exist_ok=True)

# ---------------- CONFIG ----------------
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
DETECTION_THRESHOLD = 0.5  # filter detections below this confidence
lock = threading.Lock()

# S3 configuration: read from environment variables
app.config["S3_BUCKET"] = os.environ.get("S3_BUCKET", "s3-objectdetection-folder")
app.config["AWS_REGION"] = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))

# Create an S3 client. boto3 will pick up credentials from env vars, shared credentials file, or IAM role.
s3_client = None
if app.config["S3_BUCKET"]:
    try:
        s3_client = boto3.client("s3", region_name=app.config["AWS_REGION"])
        app.logger.info("S3 client initialized for bucket: %s (region=%s)", app.config["S3_BUCKET"], app.config["AWS_REGION"])
    except Exception as e:
        app.logger.exception("Failed to initialize S3 client: %s", e)
        s3_client = None
else:
    app.logger.info("No S3_BUCKET configured. Falling back to local detected folder for persistence.")

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

# ---------------- Load model ----------------
app.logger.info("🔄 Loading TensorFlow SSD model ... (this may take a while)")
detector = hub.load(MODEL_URL)
app.logger.info("✅ Model loaded successfully!")

# ---------------- Helpers ----------------
def preprocess_image(image_pil: Image.Image, target_size=(512, 512)):
    """Resize and lightly enhance image for detection"""
    image_pil = image_pil.resize(target_size, Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(1.15)
    enhancer = ImageEnhance.Sharpness(image_pil)
    image_pil = enhancer.enhance(1.05)
    return image_pil

def pil_from_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return preprocess_image(image)

def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 85):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf

def _local_save_annotated_image(pil_img: Image.Image, prefix="detected"):
    """Fallback local save (kept for compatibility/testing)."""
    os.makedirs(app.config["DETECTED_DIR"], exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filename = secure_filename(filename)
    filepath = os.path.join(app.config["DETECTED_DIR"], filename)
    pil_img.save(filepath, format="JPEG", quality=90)
    url = url_for("static", filename=f"detected/{filename}", _external=False)
    return filepath, url

def save_annotated_image(pil_img: Image.Image, prefix="detected"):
    """
    Upload the annotated PIL image to S3 and return (s3_key, url).
    If S3 is not configured, save to local detected folder and return local path+url.
    """
    # prepare image bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    image_bytes = buf.read()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filename = secure_filename(filename)
    s3_bucket = app.config.get("S3_BUCKET", "")

    if s3_client and s3_bucket:
        s3_key = f"detected/{filename}"
        try:
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=image_bytes,
                ContentType="image/jpeg",
                ACL="private"  # prefer private + presigned URLs
            )
            # create presigned URL
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_bucket, "Key": s3_key},
                ExpiresIn=3600
            )
            return s3_key, presigned_url
        except ClientError as e:
            current_app.logger.exception("Failed to upload to S3: %s", e)
            # fall back to local save
            try:
                return _local_save_annotated_image(pil_img, prefix=prefix)
            except Exception:
                raise
    else:
        # S3 not configured: local save
        return _local_save_annotated_image(pil_img, prefix=prefix)

def annotate_image_pil(image_pil: Image.Image, boxes, classes, scores, threshold=DETECTION_THRESHOLD):
    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size

    base_font_size = max(18, min(w, h) // 22)
    font = None
    font_paths = [
        "Arial-Bold.ttf", "arial-bold.ttf", "ArialBold.ttf",
        "DejaVuSans-Bold.ttf", "dejavu-sans-bold.ttf",
        "Helvetica-Bold.ttf", "helvetica-bold.ttf",
        "Arial.ttf", "arial.ttf", "DejaVuSans.ttf"
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, size=base_font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    # draw detections sorted by score descending
    sorted_idx = np.argsort(scores)[::-1]
    for idx in sorted_idx:
        score = float(scores[idx])
        if score < threshold:
            continue
        box = boxes[idx]
        cls = int(classes[idx])

        ymin, xmin, ymax, xmax = box
        left, top, right, bottom = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

        # skip tiny boxes
        if (right - left) <= 1 or (bottom - top) <= 1:
            continue

        # label
        label = COCO_LABELS[cls] if 0 <= cls < len(COCO_LABELS) else str(cls)
        caption = f"{label}: {score:.2f}"

        # color by confidence
        if score > 0.8:
            color = "#00FF00"
            text_color = "white"
        elif score > 0.6:
            color = "#FFA500"
            text_color = "black"
        else:
            color = "#FF0000"
            text_color = "white"

        box_thickness = max(2, min(w, h) // 180)
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=box_thickness)

        # measure text
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 6

        text_bg_left = max(0, left)
        text_bg_top = top - text_h - padding * 2
        text_bg_right = min(w, left + text_w + padding * 2)
        text_bg_bottom = top

        # if out of top bounds, place under the box
        if text_bg_top < 0:
            text_bg_top = bottom
            text_bg_bottom = min(h, bottom + text_h + padding * 2)

        draw.rectangle([(text_bg_left, text_bg_top), (text_bg_right, text_bg_bottom)], fill=color)

        text_x = text_bg_left + padding
        text_y = text_bg_top + padding

        # outline for readability
        outline_color = "black" if text_color == "white" else "white"
        outline_width = 1
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((text_x + dx, text_y + dy), caption, fill=outline_color, font=font)

        draw.text((text_x, text_y), caption, fill=text_color, font=font)

    return image_pil

def run_detector_on_numpy(image_np: np.ndarray):
    """
    image_np expected in HxWxC (uint8)
    returns boxes, classes, scores (numpy arrays)
    """
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, 0)

    with lock:
        outputs = detector(input_tensor)

    # outputs are tensors — get numpy arrays
    boxes = outputs['detection_boxes'][0].numpy()
    scores = outputs['detection_scores'][0].numpy()
    classes = outputs['detection_classes'][0].numpy().astype(np.int32)

    # filter by threshold (retain >= threshold)
    valid = scores >= DETECTION_THRESHOLD
    boxes = boxes[valid]
    scores = scores[valid]
    classes = classes[valid]

    return boxes, classes, scores

# ---------------- Routes ----------------
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

        if not file.mimetype or not file.mimetype.startswith("image/"):
            return jsonify({"error": "Uploaded file is not an image"}), 400

        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty upload"}), 400

        # Verify image integrity
        try:
            img_check = Image.open(io.BytesIO(image_bytes))
            img_check.verify()
        except Exception as e:
            current_app.logger.warning("Pillow verify failed: %s", e)
            return jsonify({"error": "Invalid image file"}), 400

        # Reload the image and process
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image_pil.size

        processed = preprocess_image(image_pil.copy())
        image_np = np.array(processed)

        boxes, classes, scores = run_detector_on_numpy(image_np)

        # Annotate on processed size for consistent coords
        annotated = annotate_image_pil(processed.copy(), boxes, classes, scores, threshold=DETECTION_THRESHOLD)

        # Resize back to original if needed
        if original_size != processed.size:
            annotated = annotated.resize(original_size, Image.Resampling.LANCZOS)

        # --- SAVE annotated image (upload to S3 or local fallback) ---
        try:
            storage_key, url = save_annotated_image(annotated, prefix="upload")
            current_app.logger.info("Saved annotated upload: %s (key=%s)", url, storage_key)
        except Exception as e:
            current_app.logger.warning("Failed to persist annotated upload: %s", e)

        buf = pil_to_jpeg_bytes(annotated, quality=95)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        current_app.logger.exception("Error in /detect_upload")
        return jsonify({"error": str(e)}), 500

@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image data"}), 400

        data_url = data.get("image")
        try:
            header, encoded = data_url.split(",", 1)
        except Exception:
            return jsonify({"error": "Invalid data URL"}), 400

        frame_bytes = base64.b64decode(encoded)
        # preprocess to model size but DO NOT save the annotated frame
        image_pil = pil_from_bytes(frame_bytes)  # already preprocesses to target size

        image_np = np.array(image_pil)
        boxes, classes, scores = run_detector_on_numpy(image_np)

        annotated = annotate_image_pil(image_pil.copy(), boxes, classes, scores, threshold=DETECTION_THRESHOLD)

        # IMPORTANT: do NOT call save_annotated_image here — webcam frames are not saved per your request

        buf = pil_to_jpeg_bytes(annotated, quality=85)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        current_app.logger.exception("Error in /detect_frame")
        return jsonify({"error": str(e)}), 500

@app.route("/get_detected_images", methods=["GET"])
def get_detected_images():
    try:
        s3_bucket = app.config.get("S3_BUCKET", "")
        if s3_client and s3_bucket:
            prefix = "detected/"
            paginator = s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=prefix)

            all_objs = []
            for page in page_iterator:
                contents = page.get("Contents", [])
                for obj in contents:
                    all_objs.append(obj)

            # sort all objects by LastModified desc
            all_objs.sort(key=lambda o: o.get("LastModified", datetime.min), reverse=True)

            urls = []
            for obj in all_objs:
                key = obj["Key"]
                try:
                    url = s3_client.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": s3_bucket, "Key": key},
                        ExpiresIn=3600
                    )
                except ClientError:
                    url = f"https://{s3_bucket}.s3.{app.config['AWS_REGION']}.amazonaws.com/{key}"
                urls.append(url)
            return jsonify({"images": urls})
        else:
            # fallback to local folder listing
            images = sorted(os.listdir(app.config["DETECTED_DIR"]), reverse=True)
            urls = [url_for('static', filename=f"detected/{img}") for img in images]
            return jsonify({"images": urls})
    except Exception as e:
        current_app.logger.exception("Error listing detected images")
        return jsonify({"error": str(e)}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    # Helpful startup logs
    if app.config.get("S3_BUCKET"):
        app.logger.info("App configured to upload detected images to S3 bucket: %s", app.config.get("S3_BUCKET"))
    else:
        app.logger.info("App configured to store detected images locally in: %s", app.config.get("DETECTED_DIR"))

    app.run(debug=True, host="127.0.0.1", port=5000)
