import torch
import numpy as np
import cv2
import os
from inference_script import load_model, predict_mask, IMAGE_SIZE

def diagnose_model():
    """Diagnose potential issues with the model"""
    print("=== Model Diagnosis ===")
    
    # Check if model file exists
    model_path = "full_unet_model (1).pth"
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
        print(f"  File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    else:
        print(f"✗ Model file not found: {model_path}")
        return
    
    # Check device availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✓ CUDA is available")
    else:
        device = torch.device('cpu')
        print("⚠ CUDA is not available, using CPU")
    
    # Load model
    print("\n--- Loading Model ---")
    try:
        model = load_model(model_path, device)
        print("✓ Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Check preprocessing settings
    print("\n--- Preprocessing Settings ---")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # Test with our created image
    test_image = "test_image.png"
    if os.path.exists(test_image):
        print(f"\n--- Testing with {test_image} ---")
        try:
            # Test with different thresholds
            for threshold in [0.1, 0.2, 0.5]:
                print(f"  Testing with threshold {threshold}...")
                mask = predict_mask(model, test_image, device, threshold)
                print(f"    ✓ Prediction successful!")
                print(f"    Mask shape: {mask.shape}")
                print(f"    Mask min/max: {mask.min():.3f}/{mask.max():.3f}")
                print(f"    Positive pixels: {np.sum(mask > 0)} ({100*np.sum(mask > 0)/mask.size:.2f}%)")
        except Exception as e:
            print(f"    ✗ Error during prediction: {e}")
    else:
        print(f"\n--- No test image found ---")
        # List all files in directory to see what images we might test with
        print("\n--- Directory Contents ---")
        files = os.listdir('.')
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            print(f"  Found {len(image_files)} image files:")
            for img in image_files[:5]:  # Show first 5
                print(f"    - {img}")
            if len(image_files) > 5:
                print(f"    ... and {len(image_files) - 5} more")
        else:
            print("  No image files found in directory")
    
    print("\n=== Diagnosis Complete ===")

if __name__ == "__main__":
    diagnose_model()