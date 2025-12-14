
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage

# --- 1. UNet Model Architecture --- #
# These classes must be defined to load the full model object

class DoubleConv(nn.Module):
    """ (convolution => BN => ReLU) * 2 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """ Downscaling with maxpool then double conv """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """ Upscaling then double conv """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# --- 2. Preprocessing Steps --- #
IMAGE_SIZE = 1024  # High resolution for detailed solar panel detection
inference_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),  # Enhance contrast for solar panels
    A.Sharpen(alpha=(0.5, 1.0), lightness=(0.5, 1.0), p=1.0),  # Sharpen edges of solar panels
    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),  # Enhance texture details
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),  # Adjust for different lighting
    ToTensorV2(),
])

# --- 3. Model Loading and Inference Function --- #

def load_model(model_path, device='cpu'):
    """Loads the full UNet model from a .pth file."""
    # Create a context where the UNet class is available in __main__
    import __main__
    __main__.UNet = UNet
    __main__.DoubleConv = DoubleConv
    __main__.Down = Down
    __main__.Up = Up
    __main__.OutConv = OutConv
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval() # Set to evaluation mode
    print(f"Model loaded successfully from {model_path} to {device}.")
    return model

def predict_mask(model, image_path, device='cpu', threshold=0.1):
    """ 
    Performs inference on a single image and returns the predicted binary mask.

    Args:
        model (torch.nn.Module): The loaded UNet model.
        image_path (str): Path to the input image.
        device (str): Device to run inference on ('cpu' or 'cuda').
        threshold (float): Threshold to convert probabilities to binary mask.

    Returns:
        numpy.ndarray: The predicted binary mask (0s and 1s) as a NumPy array.
    """
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply inference transform
    transformed_image = inference_transform(image=image)["image"]
    
    # Add batch dimension and move to device
    input_tensor = transformed_image.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output_logits = model(input_tensor)
    
    # Apply sigmoid and threshold for binary mask
    predicted_probabilities = torch.sigmoid(output_logits)
    predicted_mask = (predicted_probabilities > threshold).float()

    # Convert to numpy array and remove batch/channel dimensions
    predicted_mask_np = predicted_mask.squeeze().cpu().numpy()

    # Post-processing specifically for solar panels
    # Remove small noise (solar panels are typically large structures)
    predicted_mask_np = ndimage.binary_opening(predicted_mask_np, iterations=2)
    # Fill small holes within detected panels
    predicted_mask_np = ndimage.binary_closing(predicted_mask_np, iterations=2)
    # Remove very small isolated regions (likely false positives)
    labeled_array, num_features = ndimage.label(predicted_mask_np)
    if num_features > 1:
        # Calculate sizes and remove components smaller than 1% of image
        sizes = ndimage.sum(predicted_mask_np, labeled_array, range(num_features + 1))
        min_size = predicted_mask_np.size * 0.01  # 1% of image size
        mask_sizes = sizes < min_size
        for i, is_small in enumerate(mask_sizes[1:], 1):  # Skip background (index 0)
            if is_small:
                predicted_mask_np[labeled_array == i] = 0

    return predicted_mask_np

# Example Usage (uncomment to run when you have an image and model file):
# if __name__ == "__main__":
#     MODEL_PATH = "full_unet_model.pth" # Make sure this file is in the same directory
#     SAMPLE_IMAGE_PATH = "path/to/your/sample_image.jpg"
#     
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Load the model
#     model = load_model(MODEL_PATH, device)
#
#     # Get prediction
#     try:
#         mask = predict_mask(model, SAMPLE_IMAGE_PATH, device)
#         print(f"Predicted mask shape: {mask.shape}")
#         # You can save the mask or display it using matplotlib
#         # import matplotlib.pyplot as plt
#         # plt.imshow(mask, cmap='gray')
#         # plt.title('Predicted Mask')
#         # plt.show()
#     except FileNotFoundError as e:
#         print(e)
#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")
