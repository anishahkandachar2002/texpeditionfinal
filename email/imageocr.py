import easyocr
import cv2
import numpy as np
from PIL import Image

# Install: pip install easyocr

def extract_text_easyocr(image_path, languages=['en']):
    """Extract text using EasyOCR"""
    # Initialize reader
    reader = easyocr.Reader(languages)
    
    # Read image and extract text
    results = reader.readtext(image_path)
    
    # Extract just the text
    text_lines = []
    for (bbox, text, confidence) in results:
        if confidence > 0.5:  # Filter low confidence
            text_lines.append(text)
    
    return '\n'.join(text_lines), results

def extract_text_with_details(image_path, languages=['en']):
    """Extract text with bounding boxes and confidence scores"""
    reader = easyocr.Reader(languages)
    results = reader.readtext(image_path)
    
    detailed_results = []
    for (bbox, text, confidence) in results:
        detailed_results.append({
            'text': text,
            'bbox': bbox,
            'confidence': confidence
        })
    
    return detailed_results

def draw_boxes_on_image(image_path, output_path, languages=['en']):
    """Draw bounding boxes around detected text"""
    reader = easyocr.Reader(languages)
    img = cv2.imread(image_path)
    results = reader.readtext(image_path)
    
    for (bbox, text, confidence) in results:
        if confidence > 0.5:
            # Convert bbox to integers
            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw bounding box
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            
            # Add text label
            cv2.putText(img, f'{text} ({confidence:.2f})', 
                       (int(bbox[0][0]), int(bbox[0][1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, img)
    return img

# Example usage
if __name__ == "__main__":
    image_path = "image.png"
    
    try:
        # Basic text extraction
        print("EasyOCR Text Extraction:")
        text, results = extract_text_easyocr(image_path)
        print(text)
        print("-" * 50)
        
        # Detailed results
        print("Detailed Results:")
        detailed = extract_text_with_details(image_path)
        for item in detailed:
            print(f"Text: '{item['text']}', Confidence: {item['confidence']:.2f}")
        print("-" * 50)
        
        # Multi-language support
        print("Multi-language extraction (English + Spanish):")
        text_multi, _ = extract_text_easyocr(image_path, languages=['en', 'es'])
        print(text_multi)
        
        # Draw bounding boxes (optional)
        # draw_boxes_on_image(image_path, "output_with_boxes.jpg")
        
    except Exception as e:
        print(f"Error: {e}")