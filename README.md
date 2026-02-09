# pixels_changer
 # PDF / Image Dark–White Pixel Converter

This project is a Streamlit-based image & PDF processing tool that performs pixel-level color transformation using OpenCV.
It allows users to interactively convert dark pixels to white and white pixels to a darker color, while keeping all other colors unchanged.

The tool works for:

Single images (.jpg, .jpeg, .png)

Multi-page PDFs (each page processed individually)

🚀 Features

📄 PDF Support

Converts each PDF page into an image

Processes all pages automatically

Exports results as a single PDF with original page size preserved

🖼️ Image Support

Works with JPG, JPEG, and PNG files

🎚️ Adjustable Thresholds

Dark Pixel Threshold
Pixels darker than the selected value are converted to white

White Pixel Threshold
Pixels brighter than the selected value are converted to a dark color

🎛️ Dark Color Intensity Slider

Control how dark the replacement color should be

0 = pure black, higher values = lighter gray

📐 Preserves Image Size

Output PDF pages match the original image dimensions

No stretching, padding, or scaling issues

⬇️ Download Options

Download each processed page as PNG

Download all pages combined into a single PDF

🧠 How It Works

Upload an image or PDF

PDF pages are converted into images using pdf2image

For each image/page:

Dark pixels (RGB ≤ dark threshold) → White

White pixels (RGB ≥ white threshold) → Dark color

Other pixels remain unchanged

Processed images are displayed in the browser

Output can be downloaded as PNG or a combined PDF

🛠️ Tech Stack

Python

OpenCV (cv2) – pixel-level image processing

NumPy – fast array operations

Streamlit – interactive web UI

Pillow (PIL) – image handling & PDF creation

pdf2image – PDF to image conversion

📦 Installation:-
pip install streamlit opencv-python numpy pillow pdf2image

Poppler Requirement (for PDF support)

Windows:
Download from
https://github.com/oschwartz10612/poppler-windows/releases

Add bin/ to PATH

Linux:

sudo apt install poppler-utils


Mac:

brew install poppler

▶️ Run the App
streamlit run app.py

📌 Use Cases

Cleaning scanned documents

Inverting document backgrounds

Preprocessing PDFs for OCR

Highlighting dark regions

Image preprocessing for computer vision tasks

📄 Output Example

Dark text → White

White background → Dark gray / black

Image size preserved in PDF output


# THANK YOU


