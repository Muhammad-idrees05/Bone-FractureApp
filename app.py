import streamlit as st
import time
import os
import gdown
from ultralytics import YOLO

from utils import read_image, read_dicom, run_inference
from pdf_generator import generate_pdf


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Radiology Assistant",
    page_icon="🏥",
    layout="wide"
)


# =========================
# CSS LOAD
# =========================
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()


# =========================
# HEADER
# =========================
st.markdown("<div class='title'>🏥 AI Radiology Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Bone Fracture Detection System (YOLO AI)</div>", unsafe_allow_html=True)
st.write("---")


# =========================
# MODEL DOWNLOAD + LOAD
# =========================
MODEL_PATH = "best.pt"
GDRIVE_URL = "https://drive.google.com/uc?id=1GEdORfto5yqARqxz5Jb9VxIPrXcz3uPn"


def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading AI model...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")


@st.cache_resource
def get_model():
    download_model()
    return YOLO(MODEL_PATH)


model = get_model()


# =========================
# SIDEBAR PATIENT INFO
# =========================
st.sidebar.header("👤 Patient Information")

patient_name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 0, 120, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

hospital_name = st.sidebar.text_input("Hospital Name")
doctor_name = st.sidebar.text_input("Doctor Name")
notes = st.sidebar.text_area("Doctor Notes")

st.sidebar.header("⚙ AI Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)


# =========================
# FILE UPLOAD
# =========================
uploaded_files = st.file_uploader(
    "📤 Upload X-ray / DICOM Images",
    type=["jpg", "jpeg", "png", "dcm"],
    accept_multiple_files=True
)

results_list = []


# =========================
# PROCESS IMAGES
# =========================
if uploaded_files:

    for file in uploaded_files:

        st.subheader(f"📁 {file.name}")

        # LOAD IMAGE
        file_type = file.name.split(".")[-1].lower()

        if file_type == "dcm":
            image = read_dicom(file)
        else:
            image = read_image(file)

        st.image(image, caption="Original X-ray")

        # SCANNING EFFECT
        progress = st.progress(0)
        status = st.empty()

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
            status.markdown("🔍 AI analyzing...")

        # INFERENCE
        try:
            result_img, fracture, confidence = run_inference(
                model, image, conf_threshold
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
            continue

        # RESULTS UI
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(result_img, caption="Prediction")

        if fracture:
            st.error("⚠ FRACTURE DETECTED")
        else:
            st.success("✅ NO FRACTURE DETECTED")

        st.info(f"Confidence: {confidence:.2%}")

        # STORE RESULT FOR PDF
        results_list.append({
            "file": file.name,
            "fracture": fracture,
            "confidence": confidence,
            "image": result_img
        })

        st.divider()


# =========================
# PDF GENERATION (FIXED CALL)
# =========================
if results_list:

    pdf = generate_pdf(
        patient_name,
        age,
        gender,
        hospital_name,
        doctor_name,
        notes,
        results_list
    )

    st.download_button(
        "📄 Download Medical Report",
        data=pdf,
        file_name="radiology_report.pdf",
        mime="application/pdf"
    )