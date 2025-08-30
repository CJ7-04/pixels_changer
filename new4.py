import cv2
import numpy as np 
import streamlit as st 
from PIL import Image
import io 
import fitz  # PyMuPDF

st.title("PDF/Image Black Pixels Extractor with Crop & Rectangle Inversion")

File uploader: images + PDFs

uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

Slider outside loop so it applies to all pages

black_thresh = st.slider("Black Threshold", 0, 255, 70)

if uploaded_file is not None: processed_pages = []

# ---------------- Handle PDF ----------------
if uploaded_file.type == "application/pdf":
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        zoom = 2.0  # lower zoom to save memory
        matrix = fitz.Matrix(zoom, zoom)

        for page_num, page in enumerate(doc):
            st.write(f"Processing Page {page_num+1}...")

            # Render PDF page as image
            pix = page.get_pixmap(matrix=matrix)
            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to OpenCV
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Auto-crop margins
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(np.vstack(contours))
                img_cv = img_cv[y:y+h, x:x+w]

            # --- Black pixel extraction ---
            black_mask = cv2.inRange(img_cv, np.array([0, 0, 0]), np.array([black_thresh, black_thresh, black_thresh]))
            output = np.zeros_like(img_cv)
            output[black_mask > 0] = [255, 255, 255]

            # --- Invert rectangles ---
            gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            _, thresh_rect = cv2.threshold(gray_output, 127, 255, cv2.THRESH_BINARY_INV)
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

            # Convert to PIL
            output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            processed_pages.append(output_pil)

            # Show & allow download for this page
            st.image(output_pil, caption=f"Processed Page {page_num+1}")
            buf = io.BytesIO()
            output_pil.save(buf, format="PNG")
            st.download_button(
                f"Download Page {page_num+1} as PNG",
                data=buf.getvalue(),
                file_name=f"page_{page_num+1}_black.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"Failed to process PDF. Error: {e}")

# ---------------- Handle Single Image ----------------
else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    black_mask = cv2.inRange(img_cv, np.array([0, 0, 0]), np.array([black_thresh, black_thresh, black_thresh]))
    output = np.zeros_like(img_cv)
    output[black_mask > 0] = [255, 255, 255]

    gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, thresh_rect = cv2.threshold(gray_output, 127, 255, cv2.THRESH_BINARY_INV)
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
    processed_pages.append(output_pil)

    st.image(output_pil, caption="Processed Image")
    buf = io.BytesIO()
    output_pil.save(buf, format="PNG")
    st.download_button(
        "Download Processed Image",
        data=buf.getvalue(),
        file_name="processed_black.png",
        mime="image/png"
    )

# ---------------- Save All to PDF ----------------
if processed_pages:
    pdf_buf = io.BytesIO()
    with processed_pages[0] as first_page:
        first_page.save(pdf_buf, format="PDF", save_all=True, append_images=processed_pages[1:])
    pdf_buf.seek(0)
    st.download_button(
        "Download All Pages as PDF",
        data=pdf_buf,
        file_name="processed_black_pages.pdf",
        mime="application/pdf"
    )















