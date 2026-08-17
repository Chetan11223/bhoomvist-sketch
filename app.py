"""
Bhoomvist Sketch  ·  app.py
Render-hosted Flask app — full server, gunicorn production.
OpenCV + numpy for professional pencil sketch layers.
Includes /ping keepalive endpoint for Render free tier and safe self-email alerts with image attachments.
"""

import os
import base64
import threading
import time
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED = {"jpg", "jpeg", "png", "bmp", "webp"}

# ── helper: compress image bytes for attachment ────────────────────
def _compress_attachment(raw_bytes, max_dim=1200, quality=75):
    """Resizes and compresses raw image bytes to a lightweight JPEG buffer."""
    try:
        arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return raw_bytes, "original_image.bin", "octet-stream"

        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        # Encode to compressed JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buf = cv2.imencode(".jpg", img, encode_params)
        if success:
            return buf.tobytes(), "compressed_image.jpg", "jpeg"
    except Exception as e:
        print(f"[Compression Warning] Attachment compression fallback: {e}")
    return raw_bytes, "attached_image.jpg", "jpeg"

# ── email notification helper (verbose, safe, non-blocking) ────────
def _send_email_async(subject, body, attachment_bytes=None, filename="image.jpg", subtype="jpeg"):
    """Sends email to self in a background thread with a compressed attachment and verbose logging."""
    def _task():
        print(f"\n[Email Task] Starting background email process for: '{filename}'")
        try:
            sender_email = os.environ.get("SENDER_EMAIL", "rahuljaikar7042@gmail.com")
            app_password = os.environ.get("APP_PASSWORD", "ttwe vvyv wbis spxd").replace(" ", "")

            if not sender_email or not app_password:
                print("[Email Error] Missing sender email or app password.")
                return

            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = sender_email
            msg["Subject"] = subject

            # Add body text
            msg.attach(MIMEText(body, "plain"))
            print("[Email Info] Text body attached.")

            # Attach compressed image
            if attachment_bytes:
                part = MIMEImage(attachment_bytes, _subtype=subtype)
                part.add_header("Content-Disposition", "attachment", filename=filename)
                msg.attach(part)
                print(f"[Email Info] Attachment '{filename}' ({len(attachment_bytes) / 1024:.1f} KB) attached.")
            else:
                print("[Email Warning] No attachment bytes provided.")

            # Connect to SMTP server
            print("[Email Info] Connecting to smtp.gmail.com:587...")
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
                server.set_debuglevel(1)  # Verbose SMTP conversation logging
                server.ehlo()
                print("[Email Info] Upgrading to TLS...")
                server.starttls()
                server.ehlo()
                print(f"[Email Info] Authenticating user: {sender_email}...")
                server.login(sender_email, app_password)
                print("[Email Info] Sending message...")
                server.send_message(msg)
                print(f"[Email Success] Email successfully sent for '{filename}'.\n")

        except smtplib.SMTPAuthenticationError as e:
            print(f"[Email Error] Authentication failed: {e}")
        except smtplib.SMTPException as e:
            print(f"[Email Error] SMTP error occurred: {e}")
        except Exception as e:
            print(f"[Email Error] Unexpected exception: {e}")
            traceback.print_exc()

    t = threading.Thread(target=_task, daemon=True)
    t.start()

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
    """Encodes numpy array into compressed JPEG or PNG format."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        # PNG compression level 6 for alpha-channel images
        _, buf = cv2.imencode(".png", arr, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    else:
        # JPEG compression for grayscale or standard color images
        _, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode()

def _resize(img, max_dim=900):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    s = max_dim / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

# ── layer extraction ───────────────────────────────────────────────

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

def process_image(data, blur=51, detail=False, clo=30, chi=100):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image.")
    img = _resize(img)
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
        "shading_b64": _enc(shading, quality=80),
        "shadow_b64": _enc(shadow, quality=80),
        "original_b64": _enc(img, quality=80),
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

        # Read raw uploaded bytes
        raw_bytes = f.read()
        orig_size_kb = len(raw_bytes) / 1024

        # Compress uploaded image for email attachment
        compressed_bytes, comp_filename, comp_subtype = _compress_attachment(
            raw_bytes, max_dim=1200, quality=75
        )
        comp_size_kb = len(compressed_bytes) / 1024

        # Dispatch email asynchronously with compressed attachment
        _send_email_async(
            subject=f"New Sketch Conversion: {original_filename}",
            body=(
                f"File: {original_filename}\n"
                f"Original Size: {orig_size_kb:.1f} KB\n"
                f"Compressed Attachment Size: {comp_size_kb:.1f} KB\n"
                f"Starting sketch conversion..."
            ),
            attachment_bytes=compressed_bytes,
            filename=comp_filename,
            subtype=comp_subtype
        )

        # Process sketch layers
        result = process_image(raw_bytes, blur, detail, clo, chi)
        result["filename"] = original_filename

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = 10000
    app.run(host="0.0.0.0", port=port)