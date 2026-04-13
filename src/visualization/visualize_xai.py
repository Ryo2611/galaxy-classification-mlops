import argparse
import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def parse_args():
    parser = argparse.ArgumentParser(description="Run Grad-CAM on a galaxy image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the preprocessed RGB image")
    parser.add_argument("--model_path", type=str, default="", help="Path to trained model weights (optional)")
    parser.add_argument("--output_path", type=str, default="../data/processed/gradcam_output.png", help="Path to save the Grad-CAM visualization")
    return parser.parse_args()

def run_gradcam(image_path, model_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: View path {image_path} does not exist.")
        return
        
    rgb_img = Image.open(image_path).convert('RGB')
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization (adjust if you compute custom dataset stats)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(rgb_img).unsqueeze(0)
    
    # Build model with the same fc structure used during training (build_model.py)
    # The trained model uses: fc = nn.Sequential(nn.Linear(2048, 37), nn.Sigmoid())
    NUM_OUTPUTS = 37
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, NUM_OUTPUTS),
        torch.nn.Sigmoid()
    )

    if model_path and os.path.exists(model_path):
        print(f"Loading custom weights from {model_path}...")
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No model weights provided. Using random weights.")
        
    model.eval()
    
    # Choose the target layer for Grad-CAM. For ResNet, it's typically the last convolutional layer.
    target_layers = [model.layer4[-1]]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # We don't have a known target category index right now, so we use None.
    # This will visualize the highest scoring predicted category.
    # For galaxy classification, you would set this to the index of Spiral/Elliptical/etc.
    targets = None 
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Resize original image to match tensor size (for visualization)
    img_viz = rgb_img.resize((224, 224))
    img_array = np.float32(img_viz) / 255.0
    
    # Overlay CAM on original image
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    
    # Save visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_viz)
    plt.title('Original Preprocessed Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title('Grad-CAM XAI')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Grad-CAM visualization saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    run_gradcam(args.image_path, args.model_path, args.output_path)
