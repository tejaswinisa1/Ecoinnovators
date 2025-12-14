import torch
import numpy as np
import cv2
from inference_script import load_model, predict_mask

def test_model_basic():
    """Test the model with basic settings to see what it detects"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = "full_unet_model (1).pth"
    try:
        model = load_model(model_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Print model information
    print(f"Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Note: To actually test with an image, you would need to provide a valid image path
    # For now, we're just verifying the model loads correctly
    print("Model is ready for inference.")

if __name__ == "__main__":
    test_model_basic()