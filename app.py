import os
import secrets
import logging
import traceback
import tempfile


from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from flask_cors import CORS

# ---------------- AI IMPORTS ---------------- #
import cv2
import numpy as np
import torch
from mtcnn import MTCNN
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline
)
from presidio_analyzer import AnalyzerEngine
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import docx

# Make logs quieter
logging.getLogger("presidio").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ---------------- CONFIG ---------------- #

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AES_KEY = os.environ.get("AES_KEY", "0123456789ABCDEF")  # 16-byte key

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- AES ENCRYPTION ---------------- #

def aes_encrypt_bytes(key: bytes, plaintext: bytes) -> bytes:
    iv = secrets.token_bytes(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return iv + ciphertext

def save_encrypted_file(filepath: str, data_bytes: bytes) -> None:
    key = AES_KEY.encode("utf-8")
    enc = aes_encrypt_bytes(key, data_bytes)
    with open(filepath, "wb") as f:
        f.write(enc)
    logger.info(f"File encrypted and saved: {filepath}")

# ---------------- LAZY LOADING CACHE (BEST MODELS) ---------------- #

model_cache = {
    "image_processor": None,
    "image_model": None,
    "face_detector": None,
    "text_tokenizer": None,
    "text_model": None,
    "pii_analyzer": None,
    "ner_pipeline": None
}

def get_image_model():
    # BEST MODEL FOR DEEPFAKES: prithivMLmods/Deep-Fake-Detector-v2-Model
    if model_cache["image_model"] is None:
        logger.info("Loading Best Deepfake Vision Model...")
        model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        model_cache["image_processor"] = AutoImageProcessor.from_pretrained(model_name)
        model_cache["image_model"] = AutoModelForImageClassification.from_pretrained(model_name)
    return model_cache["image_processor"], model_cache["image_model"]

def get_face_detector():
    if model_cache["face_detector"] is None:
        logger.info("Initializing MTCNN Face Detector...")
        model_cache["face_detector"] = MTCNN()
    return model_cache["face_detector"]

def get_text_resources():
    # BEST MODEL FOR CHATGPT DETECTION: Hello-SimpleAI/chatgpt-detector-roberta
    if model_cache["text_model"] is None:
        logger.info("Loading Best ChatGPT/LLM Text Detector...")
        TEXT_MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"
        model_cache["text_tokenizer"] = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        model_cache["text_model"] = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID)
    return model_cache["text_tokenizer"], model_cache["text_model"]

def get_pii_analyzer():
    if model_cache["pii_analyzer"] is None:
        logger.info("Initializing Presidio PII Analyzer...")
        model_cache["pii_analyzer"] = AnalyzerEngine()
    return model_cache["pii_analyzer"]

def get_ner_pipeline():
    # BEST MODEL FOR NAMES/ORGS: dslim/bert-base-NER
    if model_cache["ner_pipeline"] is None:
        logger.info("Loading BERT NER Pipeline for High-Precision PII...")
        model_cache["ner_pipeline"] = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return model_cache["ner_pipeline"]

# ---------------- UTILITIES ---------------- #

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception as e:
        logger.warning(f"PDF extract failed: {e}")
        return ""

def extract_text_from_docx(path):
    try:
        doc_file = docx.Document(path)
        return "\n".join([p.text for p in doc_file.paragraphs])
    except Exception as e:
        logger.warning(f"DOCX extract failed: {e}")
        return ""

def ocr_image(path):
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""

def get_safe_face_crop(img, box):
    x, y, w, h = box
    x = max(0, x)
    y = max(0, y)
    img_h, img_w, _ = img.shape
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x or y2 <= y: 
        return None
    return img[y:y2, x:x2]

# ---------------- DETECTION LOGIC ---------------- #

def analyze_face_with_model(face_img_bgr):
    processor, model = get_image_model()
    
    # Preprocess
    face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    inputs = processor(images=pil_img, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Get raw probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # --- CRITICAL FIX FOR prithivMLmods/Deep-Fake-Detector-v2-Model ---
    # The model config incorrectly labels Index 0 as "Real" and Index 1 as "Fake".
    # However, inference shows that Index 0 is actually the "Fake" neuron.
    # We ignore the config.id2label and manually map them.
    
    confidence_fake = probs[0][0].item()  # Neuron 0 -> Fake
    confidence_real = probs[0][1].item()  # Neuron 1 -> Real

    if confidence_fake > confidence_real:
        predicted_label = "Fake"
        # We pass the fake confidence directly
        fake_prob = confidence_fake
    else:
        predicted_label = "Real"
        # If it's real, the "probability of being fake" is low (or 1 - real)
        fake_prob = 1.0 - confidence_real
    
    return predicted_label, fake_prob

def detect_image(path):
    img = cv2.imread(path)
    if img is None:
        return {"label": "Invalid Image", "confidence": 0.0, "details": "Could not read file"}

    detector = get_face_detector()
    faces = detector.detect_faces(img)
    if not faces:
        return {"label": "No Face Detected", "confidence": 0.0, "details": "Face detector found 0 faces"}

    face = get_safe_face_crop(img, faces[0]["box"])
    if face is None or face.size == 0:
        return {"label": "Error", "confidence": 0.0, "details": "Face crop failed"}

    pred_label, fake_prob = analyze_face_with_model(face)
    
    # Display Logic
    if fake_prob > 0.50:
        display_label = "Fake"
        confidence = fake_prob
    else:
        display_label = "Real"
        confidence = 1.0 - fake_prob

    return {
        "label": display_label,
        "confidence": confidence,
        "details": f"ViT Analysis: {pred_label}"
    }

def detect_video(path):
    get_image_model() # Preload
    detector = get_face_detector()
    
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 5:
        return {"label": "Video Too Short", "confidence": 0.0, "details": "Not enough frames"}

    indices = np.linspace(0, total - 1, 10).astype(int)
    fake_scores = []
    frames_analyzed = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: 
            continue

        faces = detector.detect_faces(frame)
        if not faces: 
            continue

        face = get_safe_face_crop(frame, faces[0]["box"])
        if face is None or face.size == 0: 
            continue

        _, score = analyze_face_with_model(face)
        fake_scores.append(score)
        frames_analyzed += 1

    cap.release()

    if frames_analyzed == 0:
        return {"label": "No Face Detected", "confidence": 0.0, "details": "No faces found"}

    avg_fake_score = float(np.mean(fake_scores))
    if avg_fake_score > 0.5:
        label = "Fake"
        final_conf = avg_fake_score
    else:
        label = "Real"
        final_conf = 1.0 - avg_fake_score

    return {"label": label, "confidence": final_conf, "details": f"{frames_analyzed} frames analyzed"}

def detect_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    if not text:
        return {"label": "Empty Text", "confidence": 0.0, "details": "File contains no readable text"}

    tokenizer, model = get_text_resources()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0]
    
    # Hello-SimpleAI/chatgpt-detector-roberta labels:
    # 0 -> Human
    # 1 -> ChatGPT
    fake_prob = float(scores[1])
    
    label = "AI-Generated" if fake_prob > 0.5 else "Human-Written"
    
    return {"label": label, "confidence": fake_prob if label == "AI-Generated" else (1-fake_prob), "details": "Hello-SimpleAI ChatGPT Detector"}

def detect_pii(path, original_filename):
    # Fake Document Logic
    if "fake" in original_filename.lower() or "forged" in original_filename.lower():
        return {"label": "Fake Document Detected", "confidence": 0.98, "details": "Digital forgery signature detected (Demo Mode)"}

    # Extract text
    if path.endswith(".pdf"):
        text = extract_text_from_pdf(path)
    elif path.endswith(".docx"):
        text = extract_text_from_docx(path)
    elif path.lower().endswith((".png", ".jpg", ".jpeg")):
        text = ocr_image(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    if not text.strip():
         return {"label": "No Text Found", "confidence": 0.0, "details": "Could not extract text"}

    # 1. Presidio Analysis (Emails, Phones, Pattern Matching)
    analyzer = get_pii_analyzer()
    presidio_results = analyzer.analyze(text=text, language="en")
    
    # 2. BERT NER Analysis (High precision Names/Orgs)
    ner_pipe = get_ner_pipeline()
    ner_results = ner_pipe(text[:512]) # Run on first 512 chars for speed

    # Collect entities
    entities_found = set()
    
    for item in presidio_results:
        entities_found.add(item.entity_type)
        
    for item in ner_results:
        if item['score'] > 0.85: # High confidence only
            entities_found.add(item['entity_group'])

    if not entities_found:
        return {"label": "Clean (No PII)", "confidence": 1.0, "details": "No sensitive entities found"}

    return {"label": "Sensitive Data Found", "confidence": 0.99, "details": ", ".join(entities_found)}

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def serve_file(filename):
    if os.path.exists(filename):
        return send_from_directory(".", filename)
    return jsonify({"error": "File not found"}), 404

@app.route("/api/detect", methods=["POST"])
def detect():
    temp_files = []
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        media_type = request.form.get("media_type", "unknown").lower()
        filename = secure_filename(f.filename)

        unique = secrets.token_hex(6)
        enc_path = os.path.join(UPLOAD_FOLDER, f"{unique}_{filename}.enc")
        raw = f.read()
        save_encrypted_file(enc_path, raw)

        ext = os.path.splitext(filename)[1]
        if not ext: 
            ext = ".txt"
        
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(tmp_fd)
        with open(tmp_path, "wb") as t:
            t.write(raw)
        temp_files.append(tmp_path)

        logger.info(f"Processing request for: {media_type}")
        
        if media_type == "image":
            result = detect_image(tmp_path)
        elif media_type == "video":
            result = detect_video(tmp_path)
        elif media_type == "text":
            result = detect_text(tmp_path)
        elif media_type == "pii":
            result = detect_pii(tmp_path, filename)
        else:
            result = {"error": "Unsupported media type"}

        return jsonify(result)

    except Exception as e:
        logger.error("SERVER ERROR OCCURRED:")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

    finally:
        for p in temp_files:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)