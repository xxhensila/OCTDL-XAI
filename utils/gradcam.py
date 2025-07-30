import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from modules.builder import generate_model
from omegaconf import OmegaConf

# === Load configuration ===
cfg = OmegaConf.load("configs/OCTDL.yaml")

# === Set model checkpoint path ===
cfg.train.checkpoint = os.path.join(cfg.base.save_path, 'best_validation_weights.pt')

# === Load the model ===
model = generate_model(cfg)
model.eval()

# === Load OCT image ===
img_path = "dataset1/test/RO/001.png"  # <- Update this to your test image path
img = Image.open(img_path).convert("RGB")

# === Preprocess the image ===
transform = transforms.Compose([
    transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0)

# === Initialize Grad-CAM ===
cam_extractor = GradCAM(model, target_layer='layer4')  # For ResNet50

# === Forward pass and extract activation map ===
output = model(input_tensor)
activation_map = cam_extractor(output.squeeze().item(), output)

# === Overlay heatmap on original image ===
result = overlay_mask(img, activation_map[0].resize(img.size), alpha=0.5)

# === Save result to file ===
output_path = "gradcam_result.png"
result.save(output_path)
print(f"Grad-CAM saved to: {output_path}")
