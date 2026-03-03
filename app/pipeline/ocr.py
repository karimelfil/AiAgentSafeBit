from typing import List
from PIL import Image
import pytesseract
import io
import os
from pdf2image import convert_from_bytes

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPLER_BIN = r"C:\Users\victus\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

OCR_LANG = "eng+ara+fra"

def ocr_image_bytes(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(image, lang=OCR_LANG)

def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(pdf_bytes, poppler_path=POPLER_BIN)
    texts: List[str] = []
    for page in pages:
        texts.append(pytesseract.image_to_string(page, lang=OCR_LANG))
    return "\n".join(texts)

def extract_text(file_bytes: bytes, content_type: str, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if content_type == "application/pdf" or ext == ".pdf":
        return ocr_pdf_bytes(file_bytes)
    return ocr_image_bytes(file_bytes)
