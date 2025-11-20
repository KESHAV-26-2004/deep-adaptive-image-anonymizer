import cv2
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path.cwd().parent
MODEL_PATH = ROOT / "models" / "efficientnet_adaptive_masker_best.pth"
IMG_SIZE = 224
CLASS_MAP = {0: "blur", 1: "pixelate", 2: "black"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
print("Model loaded.")

# -----------------------------
# PREPROCESS & PREDICT REGION
# -----------------------------
def predict_region(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(tensor)
        pred = int(out.argmax(1).cpu().item())

    return pred

# -----------------------------
# LOAD ANY IMAGE & TEST MODEL
# -----------------------------
def evaluate_image(img_path, x1, y1, x2, y2):
    """
    Provide an image & bounding box of region you want to test.
    Shows side-by-side: original crop vs class prediction.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Invalid crop region")

    pred = predict_region(crop)

    # -- VISUALIZE --
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(crop_rgb)
    plt.title("Cropped Region")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(crop_rgb)
    plt.title(f"Predicted: {CLASS_MAP[pred]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Prediction: {pred} → {CLASS_MAP[pred]}")
    return pred

# -----------------------------
# HOW TO USE
# -----------------------------
if __name__ == "__main__":
    # Example: Modify this in Jupyter or run the .py directly
    test_image = ROOT / "test" / "testimage.png"

    # Provide ANY region manually (x1, y1, x2, y2)
    # Example: crop center of image
    img = cv2.imread(str(test_image))
    h, w = img.shape[:2]
    crop_coords = (w//4, h//4, (w//4)*3, (h//4)*3)

    print("Testing:", test_image)
    evaluate_image(test_image, *crop_coords)
