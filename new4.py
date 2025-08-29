import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import fitz  # PyMuPDF

st.title("PDF/Image--Black Pixels Extractor")

# File uploader: images + PDFs
uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    images = []

    # Handle PDF
    if uploaded_file.type == "application/pdf":
        try:
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []

            zoom = 4.0  # ~216 DPI for higher quality
            matrix = fitz.Matrix(zoom, zoom)

            for page in doc:
                pix = page.get_pixmap(matrix=matrix)
                img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Convert to OpenCV
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # Auto-crop white/margin areas
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(np.vstack(contours))
                    img_cv = img_cv[y:y+h, x:x+w]

                # Convert back to PIL
                img_pil_cropped = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                images.append(img_pil_cropped)

        except Exception as e:
            st.error(f"Failed to process PDF. Error: {e}")
    else:
        # Handle single image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        images = [Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))]

    # Black threshold slider
    black_thresh = st.slider("Black Threshold", 0, 255, 107)

    processed_pages = []

    for i, img in enumerate(images):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Black pixel conversion logic
        black_mask = cv2.inRange(img_cv, np.array([0, 0, 0]), np.array([black_thresh, black_thresh, black_thresh]))
        output = np.zeros_like(img_cv)              # start all black
        output[black_mask > 0] = [255, 255, 255]   # convert black pixels to white

        # --- Invert only large true rectangles ---
        gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, thresh_rect = cv2.threshold(gray_output, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:  # skip small noise
                continue

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Check if polygon has 4 vertices → likely rectangle
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                roi = output[y:y+h, x:x+w]
                roi = 255 - roi  # invert colors inside rectangle
                output[y:y+h, x:x+w] = roi

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













