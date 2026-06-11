import io
import datetime
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# =========================
# IMAGE CONVERTER
# =========================
def image_to_buffer(img):

    img = np.array(img)

    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    img = img.astype(np.uint8)

    pil_img = Image.fromarray(img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


# =========================
# PDF GENERATOR (FIXED)
# =========================
def generate_pdf(
    patient_name,
    patient_id,
    age,
    gender,
    notes,
    results
):

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ================= HEADER =================
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 50, "AI RADIOLOGY REPORT")

    c.setFont("Helvetica", 10)
    c.drawString(
        50,
        height - 80,
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ================= PATIENT INFO =================
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 120, "Patient Information")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 140, f"Name: {patient_name or 'N/A'}")
    c.drawString(50, height - 155, f"Patient ID: {patient_id or 'N/A'}")
    c.drawString(50, height - 170, f"Age: {age}")
    c.drawString(50, height - 185, f"Gender: {gender}")

    # ================= NOTES =================
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 215, "Doctor Notes:")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 230, (notes or "No notes provided")[:120])

    # ================= RESULTS =================
    y = height - 270

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Scan Results")
    y -= 25

    for r in results:

        status = "FRACTURE DETECTED ⚠" if r["fracture"] else "NO FRACTURE ✓"

        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, f"File: {r['file']}")

        c.setFont("Helvetica", 10)
        c.drawString(50, y - 15, f"Result: {status}")
        c.drawString(50, y - 30, f"Confidence: {r['confidence']:.2%}")

        # IMAGE
        try:
            img_buffer = image_to_buffer(r["image"])
            c.drawImage(img_buffer, 320, y - 90, width=200, height=130)
        except:
            c.drawString(320, y - 40, "Image not rendered")

        y -= 170

        if y < 120:
            c.showPage()
            y = height - 100

    c.save()
    buffer.seek(0)

    return buffer