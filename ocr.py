import cv2
import pytesseract
import numpy as np

# Configure path
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def advanced_ocr(image_path):
    # 1. Load image with OpenCV
    image = cv2.imread(image_path)
    
    # 2. Pre-processing
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding (Binarization) to make text pop
    # Using OTSU + Gaussian Blur to handle shadows and uneven lighting
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. OCR Configuration
    # --psm 6: Assume a single uniform block of text
    # --oem 3: Default OCR Engine Mode (LSTM)
    custom_config = r'--oem 3 --psm 6'
    
    # 4. Extract Text
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    return text

if __name__ == "__main__":
    print("--- EXTRACTED TEXT ---")
    print(advanced_ocr("com.jpeg"))