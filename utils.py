import io
import numpy as np
import cv2
from PIL import Image
import pydicom
from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
def load_model(path):
    """
    Load YOLO model.
    Streamlit will cache this function so model loads only once.
    """
    return YOLO(path)

# =========================
# READ IMAGE (JPG / PNG)
# =========================
def read_image(file):
    """
    Convert uploaded image to RGB numpy array.
    Ensures compatibility with YOLO (3-channel input).
    """
    image = Image.open(file)

    # Convert everything to RGB (handles RGBA, grayscale, palette)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return np.array(image)

# =========================
# READ DICOM (.dcm)
# =========================
def read_dicom(file):
    """
    Read DICOM safely from Streamlit upload.
    Converts to normalized RGB numpy array.
    """

    # IMPORTANT: reset pointer (Streamlit uploads require this)
    file.seek(0)

    dicom = pydicom.dcmread(file)

    img = dicom.pixel_array.astype(np.float32)

    # Normalize safely
    min_val, max_val = img.min(), img.max()

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    img = (img * 255).astype(np.uint8)

    # Ensure 3-channel RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)

    return img

# =========================
# RUN YOLO INFERENCE
# =========================
def run_inference(model, image, conf_thres):
    """
    Run YOLO prediction on image.

    Returns:
        annotated image (numpy array)
        fracture (bool)
        confidence (float)
    """

    results = model(image, conf=conf_thres)[0]

    # Annotated output image
    annotated = results.plot()

    boxes = results.boxes

    # Safe confidence extraction
    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        confidence = float(np.max(confs))
        fracture = True
    else:
        confidence = 0.0
        fracture = False

    return annotated, fracture, confidence