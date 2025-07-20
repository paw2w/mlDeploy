import requests
import os

def extract_text_from_image(image_path):
    api_key = os.getenv("OCR_SPACE_API_KEY", "helloworld")  # Replace with real key in production
    with open(image_path, 'rb') as image_file:
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'filename': image_file},
            data={
                'apikey': api_key,
                'language': 'eng',
                'OCREngine': 2,
            },
        )
    result = response.json()
    return result['ParsedResults'][0]['ParsedText'] if result.get('ParsedResults') else ''
