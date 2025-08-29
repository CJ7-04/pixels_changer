import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import fitz  # PyMuPDF

st.title("PDF/Image Black Pixels Extractor (Preserve Diagrams)")

uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    images = []

    if uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        zoom = 3.0
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img_pil)  # Keep original diagrams intact
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        images = [img_pil]

    black_thresh = st.slider("Black Threshold", 0, 255, 70)
    processed_pages = []

    for i, img in enumerate(images):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Optional: Detect text areas using thresholding
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)

        # Only apply black pixel inversion to text regions
        output = img_cv.copy()
        output[thresh > 0] = 255 - output[thresh > 0]  # invert text pixels

        output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        processed_pages.append(output_pil)
        st.image(output_pil, caption=f"Processed Page {i+1}")

    if processed_pages:
        pdf_buf = io.BytesIO()
        processed_pages[0].save(pdf_buf, format="PDF", save_all=True, append_images=processed_pages[1:])
        st.download_button(
            "Download All Pages as PDF",
            data=pdf_buf.getvalue(),
            file_name="processed_preserve_diagrams.pdf",
            mime="application/pdf"
        )











