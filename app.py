"""
Bhoomvist Sketch  ·  app.py
Render-hosted Flask app — full server, gunicorn production.
OpenCV + numpy for professional pencil sketch layers.
Stores compressed source images directly to Supabase Storage.
Includes /ping keepalive endpoint for Render free tier.
"""

import os
import base64
import threading
import time
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from supabase import create_client, Client
from dotenv import load_dotenv

# ── Load Environment Variables ──────────────────────────────────────
# Loads variables from a local .env file (for local development).
# In production on Render, system environment variables take precedence.
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED = {"jpg", "jpeg", "png", "bmp", "webp"}

# ── Supabase Configuration ─────────────────────────────────────────
RAW_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_URL = RAW_URL.replace("/rest/v1", "").rstrip("/") if RAW_URL else None
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "sketch-data")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"[Supabase] Client initialized successfully for bucket '{SUPABASE_BUCKET}'.")
    except Exception as e:
        print(f"[Supabase Init Warning] Failed to initialize client: {e}")
else:
    print("[Supabase Warning] SUPABASE_URL or SUPABASE_KEY missing. Storage upload disabled.")

# ── keepalive for Render free tier (self-ping every 14 min) ───────
def _keepalive():
    """Ping self every 14 minutes so Render free instance stays warm."""
    import urllib.request
    time.sleep(60)
    while True:
        try:
            port = os.environ.get("PORT", "8080")
            urllib.request.urlopen(f"http://localhost:{port}/ping", timeout=10)
        except Exception:
            pass
        time.sleep(14 * 60)

if os.environ.get("RENDER"):
    t = threading.Thread(target=_keepalive, daemon=True)
    t.start()

# ── helpers ────────────────────────────────────────────────────────

def _ok(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED

def _enc(arr, quality=80):
    """Encodes numpy array into compressed Base64 JPEG or PNG format."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        _, buf = cv2.imencode(".png", arr, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    else:
        _, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode()

def _compress_and_decode(raw_bytes, max_dim=900, quality=80):
    """Downscales and compresses raw uploaded image bytes, returning both array and JPEG bytes."""
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image.")

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    # Encode with lossy JPEG compression
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buf = cv2.imencode(".jpg", img, encode_params)
    if not success:
        raise ValueError("Failed to encode compressed image.")

    compressed_bytes = buf.tobytes()
    compressed_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return compressed_img, compressed_bytes

def _upload_to_supabase(image_bytes, original_filename):
    """Uploads in-memory compressed image to Supabase Storage and returns the public URL."""
    if not supabase:
        return None
    try:
        # Create a unique filename path: uploads/timestamp_uuid.jpg
        ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else "jpg"
        unique_name = f"uploads/{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"

        # Upload byte buffer to Supabase bucket
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=unique_name,
            file=image_bytes,
            file_options={"content-type": "image/jpeg", "x-upsert": "true"}
        )

        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_name)
        return public_url
    except Exception as e:
        print(f"[Supabase Upload Error] Failed to upload image: {e}")
        return None

def _contour_paths(contours):
    paths = []
    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim < 2 or len(pts) < 4:
            continue
        simp = cv2.approxPolyDP(cnt, 1.2, closed=False).squeeze()
        if simp.ndim < 2 or len(simp) < 2:
            continue
        coords = simp.tolist()
        d = f"M {coords[0][0]:.1f} {coords[0][1]:.1f}"
        for x, y in coords[1:]:
            d += f" L {x:.1f} {y:.1f}"
        paths.append(d)
    paths.sort(key=lambda p: -p.count("L"))
    return paths

# ── layer extraction ───────────────────────────────────────────────

def process_image(img, blur=51, detail=False, clo=30, chi=100, quality=80):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Layer 1: mid-tone shading (dodge blend)
    k = max(3, blur | 1)
    blurred = cv2.GaussianBlur(cv2.bitwise_not(gray), (k, k), 0)
    shading = cv2.divide(gray, 255 - blurred, scale=256.0)
    if detail:
        gauss = cv2.GaussianBlur(shading, (9, 9), 10.0)
        shading = np.clip(cv2.addWeighted(shading, 1.5, gauss, -0.5, 0), 0, 255).astype(np.uint8)

    # Layer 2: deep shadow mask
    shadow = cv2.bitwise_not(
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    )
    shadow = cv2.dilate(shadow, np.ones((2, 2), np.uint8), iterations=1)

    # Layer 3: Canny edges → SVG paths
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), clo, chi)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    svg_paths = _contour_paths(cnts)

    # Layer 4: shading with alpha
    bgra = cv2.cvtColor(shading, cv2.COLOR_GRAY2BGRA)
    bgra[:, :, 3] = 255 - shading

    return {
        "width": w,
        "height": h,
        "outline_svg": svg_paths,
        "shading_b64": _enc(shading, quality=quality),
        "shadow_b64": _enc(shadow, quality=quality),
        "original_b64": _enc(img, quality=quality),
        "shading_alpha_b64": _enc(bgra),
    }

# ── routes ─────────────────────────────────────────────────────────

@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/convert", methods=["POST"])
def convert():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400
    f = request.files["image"]
    if not f.filename or not _ok(f.filename):
        return jsonify({"error": "Invalid or unsupported file."}), 400
    try:
        blur = int(request.form.get("intensity", 51))
        clo = int(request.form.get("canny_lo", 30))
        chi = int(request.form.get("canny_hi", 100))
        detail = request.form.get("detail_boost", "false").lower() == "true"
        original_filename = secure_filename(f.filename)

        raw_bytes = f.read()

        # 1. Compress once in-memory
        compressed_img, compressed_bytes = _compress_and_decode(
            raw_bytes, max_dim=900, quality=80
        )

        # 2. Upload compressed bytes to Supabase Storage
        supabase_image_url = _upload_to_supabase(compressed_bytes, original_filename)

        # 3. Process sketch layers on compressed image
        result = process_image(compressed_img, blur=blur, detail=detail, clo=clo, chi=chi)
        result["filename"] = original_filename
        result["supabase_url"] = supabase_image_url

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)