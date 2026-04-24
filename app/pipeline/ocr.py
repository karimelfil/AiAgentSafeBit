from typing import List
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import io
import os
from pdf2image import convert_from_bytes

# Set up Tesseract and Poppler paths
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
POPLER_BIN = os.getenv(
    "POPLER_BIN",
    r"C:\Users\victus\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin",
)

# English-only OCR keeps the extraction aligned with the English knowledge base.
OCR_LANG = "eng"

# OCR configuration oem3 ocr engine and psm6 assume a single uniform block of text
OCR_CONFIGS = [
    "--oem 3 --psm 6",
    "--oem 3 --psm 11",
]


#Improve image quality for better OCR results
def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image) #fix orientation 
    image = image.convert("L") #remove colors for better result 
    image = ImageOps.autocontrast(image) #improve contrast

    #Upscale small images to improve OCR accuracy if < 1800px 
    if image.width < 1800:
        scale = 1800 / float(image.width)
        image = image.resize(
            (1800, max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN) #sharpen to make text clearer
    image = image.point(lambda pixel: 255 if pixel > 160 else 0) #binarize to black and white for better OCR accuracy
    return image

#Run OCR on the preprocessed image using multiple configurations and return the best result
def _run_ocr(image: Image.Image) -> str:
    best_text = ""
    for config in OCR_CONFIGS:
        text = pytesseract.image_to_string(image, lang=OCR_LANG, config=config).strip()
        if len(text) > len(best_text):
            best_text = text
        if len(best_text) >= 40:
            break
    return best_text


#Extract text from image bytes using OCR
def ocr_image_bytes(image_bytes: bytes) -> str:
    image = preprocess_for_ocr(Image.open(io.BytesIO(image_bytes))) #convert bytes to image and preprocess for better OCR results
    return _run_ocr(image) #return extracted text


#Extract text from PDF by converting each page to an image 
def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(
        pdf_bytes,
        poppler_path=POPLER_BIN,
        dpi=300,
        fmt="png",
    ) #convert each page to image 
    texts: List[str] = []
    for page in pages:
        texts.append(_run_ocr(preprocess_for_ocr(page)))
    return "\n".join(texts) #return text from all pages into a single string


#Pdf or Image 
def extract_text(file_bytes: bytes, content_type: str, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1] #extarct the extension
    if content_type == "application/pdf" or ext == ".pdf":
        return ocr_pdf_bytes(file_bytes)
    return ocr_image_bytes(file_bytes)
