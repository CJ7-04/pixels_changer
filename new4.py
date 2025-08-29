import cv2
import numpy as np
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
import io

st.title("PDF/Image Black Pixels Extractor")

# File uploader: images + PDFs
uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    images = []

    # Handle PDF
    if uploaded_file.type == "application/pdf":
        try:
            # Preserve original size by setting dpi (default 200)
            pdf_pages = convert_from_bytes(uploaded_file.read(), dpi=200, poppler_path="/usr/bin")
            images = pdf_pages
        except Exception as e:
            st.error(f"Failed to process PDF. Make sure poppler is installed. Error: {e}")
    else:
        # Handle single image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        images = [img_pil]

    # Black threshold slider
    black_thresh = st.slider("Black Threshold", 0, 255, 70)

    processed_pages = []

    for i, img in enumerate(images):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Black pixel conversion logic
        black_mask = cv2.inRange(img_cv, np.array([0, 0, 0]), np.array([black_thresh, black_thresh, black_thresh]))
        output = np.zeros_like(img_cv)              # start all black
        output[black_mask > 0] = [255, 255, 255]   # convert black pixels to white

        # Convert to PIL for display and PDF
        output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        processed_pages.append(output_pil)

        # Display processed page
        st.image(output_pil, caption=f"Processed Page {i+1}")

        # Download individual page as PNG
        buf = io.BytesIO()
        output_pil.save(buf, format="PNG")
        st.download_button(
            f"Download Page {i+1} as PNG",
            data=buf.getvalue(),
            file_name=f"page_{i+1}_black.png",
            mime="image/png"
        )

    # Combine all processed pages into single PDF
    if processed_pages:
        pdf_buf = io.BytesIO()
        processed_pages[0].save(pdf_buf, format="PDF", save_all=True, append_images=processed_pages[1:])
        st.download_button(
            "Download All Pages as PDF",
            data=pdf_buf.getvalue(),
            file_name="processed_black_pages.pdf",
            mime="application/pdf"
        )

# To run: streamlit run new4.py


