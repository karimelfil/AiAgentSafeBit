from typing import List
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import io
import os
from pdf2image import convert_from_bytes

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPLER_BIN = r"C:\Users\victus\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

OCR_LANG = "eng+ara+fra"
OCR_CONFIG = "--oem 3 --psm 6"

def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    if image.width < 1800:
        scale = 1800 / float(image.width)
        image = image.resize(
            (1800, max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda pixel: 255 if pixel > 160 else 0)
    return image

def ocr_image_bytes(image_bytes: bytes) -> str:
    image = preprocess_for_ocr(Image.open(io.BytesIO(image_bytes)))
    return pytesseract.image_to_string(image, lang=OCR_LANG, config=OCR_CONFIG)

def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(
        pdf_bytes,
        poppler_path=POPLER_BIN,
        dpi=300,
        fmt="png",
    )
    texts: List[str] = []
    for page in pages:
        texts.append(
            pytesseract.image_to_string(
                preprocess_for_ocr(page),
                lang=OCR_LANG,
                config=OCR_CONFIG,
            )
        )
    return "\n".join(texts)

def extract_text(file_bytes: bytes, content_type: str, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if content_type == "application/pdf" or ext == ".pdf":
        return ocr_pdf_bytes(file_bytes)
    return ocr_image_bytes(file_bytes)
