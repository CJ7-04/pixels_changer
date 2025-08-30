import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import fitz  # PyMuPDF

st.title("PDF/Image Black Pixels Extractor with Crop & Rectangle Inversion")

# File uploader
temp_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if temp_file is not None:
    black_thresh = st.slider("Black Threshold", 0, 255, 70)
    processed_pages = []

    # Handle PDF one page at a time
    if temp_file.type == "application/pdf":
        pdf_bytes = temp_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for i, page in enumerate(doc):
            try:
                zoom = 2.0  # reduce memory usage
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix)
                img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Convert to OpenCV
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # Crop white margins if possible
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(np.vstack(contours))
                    img_cv = img_cv[y:y+h, x:x+w]

                # Convert black pixels to white
                black_mask = cv2.inRange(img_cv, np.array([0, 0, 0]), np.array([black_thresh, black_thresh, black_thresh]))
                output = np.zeros_like(img_cv)
                output[black_mask > 0] = [255, 255, 255]

                # Invert large rectangles only
                gray_out = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, thresh_rect = cv2.threshold(gray_out, 127, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cv2.contourArea(cnt) < 5000:
                        continue
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        roi = output[y:y+h, x:x+w]
                        roi = 255 - roi
                        output[y:y+h, x:x+w] = roi

                # Show result
                output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                st.image(output_pil, caption=f"Processed Page {i+1}")

                # Save page for final PDF
                processed_pages.append(output_pil)

                # Allow download of page as PNG
                buf = io.BytesIO()
                output_pil.save(buf, format="PNG")
                st.download_button(
                    f"Download Page {i+1} as PNG",
                    data=buf.getvalue(),
                    file_name=f"page_{i+1}_processed.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Error processing page {i+1}: {e}")

        # Combine into single PDF
        if processed_pages:
            pdf_buf = io.BytesIO()
            processed_pages[0].save(
                pdf_buf,
                format="PDF",
                save_all=True,
                append_images=processed_pages[1:]
            )
            st.download_button(
                "Download All Pages as PDF",
                data=pdf_buf.getvalue(),
                file_name="processed_document.pdf",
                mime="application/pdf"
            )

    else:
        # Handle single image
        file_bytes = np.asarray(bytearray(temp_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        black_mask = cv2.inRange(img_cv, np.array([0, 0, 0]), np.array([black_thresh, black_thresh, black_thresh]))
        output = np.zeros_like(img_cv)
        output[black_mask > 0] = [255, 255, 255]

        gray_out = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, thresh_rect = cv2.threshold(gray_out, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 5000:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                roi = output[y:y+h, x:x+w]
                roi = 255 - roi
                output[y:y+h, x:x+w] = roi

        output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        st.image(output_pil, caption="Processed Image")

        buf = io.BytesIO()
        output_pil.save(buf, format="PNG")
        st.download_button(
            "Download Processed Image",
            data=buf.getvalue(),
            file_name="processed_image.png",
            mime="image/png"
        )
















