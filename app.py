import streamlit as st
import cv2
import numpy as np
import torch
import timm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import easyocr
from presidio_analyzer import AnalyzerEngine
from io import BytesIO
from PIL import Image

# ==========================================
# CONFIG
# ==========================================
ROOT = Path.cwd()
PII_MODEL_DIR = ROOT / "models" / "pii_classifier"
REGION_MODEL_PATH = ROOT / "models" / "efficientnet_adaptive_masker_best.pth"

CLASS_MAP = {0: "blur", 1: "pixelate", 2: "black"}
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_pii_model():
    tokenizer = AutoTokenizer.from_pretrained(str(PII_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(PII_MODEL_DIR))
    model.to(device).eval()
    return tokenizer, model

@st.cache_resource
def load_region_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(REGION_MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

tokenizer, pii_model = load_pii_model()
region_model = load_region_model()

reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
pii_analyzer = AnalyzerEngine()


# ==========================================
# HELPERS
# ==========================================
def is_pii_text(text):
    text = text.strip()
    if any(ch.isdigit() for ch in text):   # HARD RULE → Numbers = PII
        return True

    try:
        results = pii_analyzer.analyze(text=text, language='en')
        return len(results) > 0
    except:
        return False


def predict_region(crop):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))
    img = torch.tensor(img).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = region_model(img)

    return int(out.argmax(1).cpu().item())


def apply_mask(crop, cls):
    if cls == 0:  # blur
        k = max(11, (min(crop.shape[:2]) // 2) | 1)
        return cv2.GaussianBlur(crop, (k, k), 0)

    elif cls == 1:  # pixelate
        small = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    else:  # black
        return np.zeros_like(crop)


def anonymize_image(image_np):
    img = image_np.copy()
    H, W = img.shape[:2]
    output = img.copy()

    ocr = reader.readtext(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for bbox, text, conf in ocr:
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        x1 = max(0,x1); y1=max(0,y1)
        x2 = min(W,x2); y2=min(H,y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # --- PII DECISION ---
        if not is_pii_text(text):
            continue

        # --- MASK TYPE PREDICTION ---
        cls = predict_region(crop)
        masked = apply_mask(crop, cls)
        output[y1:y2, x1:x2] = masked

    return output


# ==========================================
# STREAMLIT FRONTEND UI
# ==========================================
st.set_page_config(page_title="AI Document Anonymizer", layout="wide")
st.title("🔐 AI-Powered Document Anonymizer")

st.write("Upload an image and the system will automatically detect faces, PII text and mask them using ML models.")

uploaded = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.spinner("Processing... Please wait ⏳"):
        output = anonymize_image(image)

    with col2:
        st.subheader("🔒 Anonymized Image")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

    # ---- DOWNLOAD BUTTON ----
    pil_out = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil_out.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download Anonymized Image",
        data=buf.getvalue(),
        file_name="anonymized.png",
        mime="image/png"
    )
