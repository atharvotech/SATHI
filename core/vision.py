"""
JARVIS Vision Engine — Screenshot capture + OCR element detection
Finds UI elements on screen by text and returns their coordinates.
"""

import os
from PIL import Image, ImageGrab
import base64
import io


def capture_screen() -> Image.Image:
    """Capture the full screen as a PIL Image."""
    return ImageGrab.grab()


def capture_screen_to_file(path: str = None) -> str:
    """Capture screen and save to file. Returns the file path."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', 'memory', 'latest_screenshot.png')
        path = os.path.abspath(path)
    img = capture_screen()
    img.save(path)
    return path


def encode_image_base64(image: Image.Image, max_size: tuple = (1024, 768)) -> str:
    """Resize and encode a PIL Image to base64 string for LLM input."""
    # Resize for efficiency (smaller image = faster processing)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def find_text_on_screen(search_text: str) -> dict:
    """
    Find text on screen using OCR (pytesseract).
    Returns bounding box coordinates of matched text.
    """
    try:
        import pytesseract
        
        img = capture_screen()
        # Get OCR data with bounding boxes
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        matches = []
        search_lower = search_text.lower()
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text and search_lower in text.lower():
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                # Return center point (best for clicking)
                cx = x + w // 2
                cy = y + h // 2
                matches.append({
                    "text": text,
                    "center_x": cx,
                    "center_y": cy,
                    "bounds": {"x": x, "y": y, "width": w, "height": h},
                    "confidence": data['conf'][i]
                })
        
        # Sort by confidence
        matches.sort(key=lambda m: m.get('confidence', 0), reverse=True)
        
        if matches:
            return {"status": "ok", "result": matches[:10]}  # Top 10 matches
        else:
            return {"status": "ok", "result": [], "message": f"Text '{search_text}' not found on screen"}
    
    except ImportError:
        return {"status": "error", "result": "pytesseract not installed. Install with: pip install pytesseract"}
    except Exception as e:
        return {"status": "error", "result": f"OCR failed: {e}"}


def describe_screen() -> dict:
    """
    Get a text description of what's on screen via OCR.
    Used as fallback when model doesn't support vision.
    """
    try:
        import pytesseract
        
        img = capture_screen()
        text = pytesseract.image_to_string(img)
        # Clean up
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        description = '\n'.join(lines[:50])  # Cap at 50 lines
        
        return {"status": "ok", "result": description}
    
    except ImportError:
        return {"status": "error", "result": "pytesseract not installed"}
    except Exception as e:
        return {"status": "error", "result": f"Screen description failed: {e}"}


def get_screen_elements() -> dict:
    """
    Detect all text elements on screen with their coordinates.
    Returns a structured list of UI elements the AI can reference.
    """
    try:
        import pytesseract
        
        img = capture_screen()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        elements = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
            if text and conf > 40:  # Filter low confidence
                elements.append({
                    "text": text,
                    "x": data['left'][i] + data['width'][i] // 2,
                    "y": data['top'][i] + data['height'][i] // 2,
                })
        
        return {"status": "ok", "result": elements[:100]}  # Cap at 100 elements
    
    except ImportError:
        return {"status": "error", "result": "pytesseract not installed"}
    except Exception as e:
        return {"status": "error", "result": f"Element detection failed: {e}"}
