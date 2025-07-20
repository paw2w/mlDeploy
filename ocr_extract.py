import easyocr
from PIL import Image

reader = easyocr.Reader(['en'])  # Can add other languages like ['en', 'de']

def extract_text_from_image(image_path: str) -> str:
    """Extract text using EasyOCR instead of Tesseract."""
    results = reader.readtext(image_path, detail=0)
    return "\n".join(results)
