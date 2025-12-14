import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
import tempfile
# Import the required functions from the inference script
from inference_script import load_model, predict_mask

# --- Application Configuration ---
# Set device for model inference
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("CUDA is available. Using GPU for inference.")
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available. Using CPU for inference.")

# Load the trained model once at application startup
MODEL_PATH = "full_unet_model (1).pth"
print(f"Loading model from {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH, DEVICE)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# --- Helper Functions ---
def create_overlay(original_image_path, mask_array, alpha=0.5):
    """
    Create an overlay of the original image and the predicted mask.
    
    Args:
        original_image_path (str): Path to the original image
        mask_array (np.ndarray): Binary mask array
        alpha (float): Transparency level for the mask
        
    Returns:
        PIL.Image: Overlay image
    """
    # Load original image
    original_image = Image.open(original_image_path).convert("RGBA")
    
    # Resize mask to match original image size if needed
    mask_resized = np.array(Image.fromarray((mask_array * 255).astype(np.uint8)).resize(original_image.size))
    
    # Create colored mask (green for solar panels)
    colored_mask = np.zeros((*original_image.size[::-1], 4), dtype=np.uint8)
    colored_mask[mask_resized > 0] = [0, 255, 0, int(255 * alpha)]  # Green with transparency
    
    # Combine original image and mask
    overlay_image = Image.alpha_composite(original_image, Image.fromarray(colored_mask))
    
    return overlay_image.convert("RGB")

def segment_image(input_image_path):
    """
    Perform segmentation on the input image and return results.
    
    Args:
        input_image_path (str): Path to the input image
        
    Returns:
        tuple: (binary_mask, overlay_image)
    """
    try:
        # Predict mask using the loaded model with fixed parameters
        mask_array = predict_mask(model, input_image_path, DEVICE, 0.1)
        
        # Convert mask to PIL Image for display
        mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
        
        # Create overlay visualization
        overlay_image = create_overlay(input_image_path, mask_array)
        
        return mask_image, overlay_image
    except Exception as e:
        raise gr.Error(f"Error during segmentation: {str(e)}")

# --- Solar Panel Detection Interface ---
with gr.Blocks(title="Solar Panel Detector") as demo:
    gr.Markdown("# ☀️ Solar Panel Detection")
    gr.Markdown("## AI-Powered Segmentation for Solar Panel Identification")
    gr.Markdown("Upload an aerial or satellite image to detect and segment solar panels using your trained U-Net model.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            input_image = gr.Image(type="filepath", label="Aerial/Satellite Image")
            run_button = gr.Button("🔍 Detect Solar Panels", variant="primary")
            gr.Markdown("*Supported formats: JPG, PNG*")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Results")
            with gr.Tabs():
                with gr.TabItem("Binary Mask"):
                    mask_output = gr.Image(label="Detected Solar Panels", type="pil")
                with gr.TabItem("Overlay"):
                    overlay_output = gr.Image(label="Panel Locations", type="pil")
    
    gr.Markdown("---")
    gr.Markdown("### 📋 Instructions")
    gr.Markdown("""
    1. Upload an aerial or satellite image containing solar panels
    2. Click 'Detect Solar Panels' to process the image
    3. View the binary mask showing detected panels and overlay visualization
    4. Green areas in the overlay indicate detected solar panels
    """)
    
    gr.Markdown("### ℹ️ Model Information")
    gr.Markdown("""
    - **Model**: Custom trained U-Net for solar panel detection
    - **Input**: Aerial/Satellite RGB images
    - **Output**: Binary segmentation mask of solar panels
    - **Threshold**: 0.1 (optimized for solar panel detection)
    """)
    
    run_button.click(
        fn=segment_image,
        inputs=input_image,
        outputs=[mask_output, overlay_output],
        api_name="detect_panels"
    )

# --- Launch Application ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7872)
