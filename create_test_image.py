import numpy as np
import cv2
from PIL import Image

def create_test_image():
    """Create a simple test image to verify the model processing"""
    # Create a simple test image (black background with a white rectangle)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add a white rectangle in the center (simulating an object to detect)
    cv2.rectangle(img, (200, 200), (300, 300), (255, 255, 255), -1)
    
    # Add some random shapes
    cv2.circle(img, (100, 100), 30, (128, 128, 128), -1)
    cv2.rectangle(img, (400, 400), (450, 450), (200, 200, 200), -1)
    
    # Save as test image
    cv2.imwrite('test_image.png', img)
    print("Created test_image.png")
    
    # Also create a JPEG version
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img.save('test_image.jpg', 'JPEG')
    print("Created test_image.jpg")

if __name__ == "__main__":
    create_test_image()