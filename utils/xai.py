import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from omegaconf import OmegaConf
from modules.builder import generate_model
import cv2

# === Load config and model ===
cfg = OmegaConf.load("configs/OCTDL.yaml")
checkpoint_path = os.path.join(cfg.base.save_path, 'best_validation_weights.pt')
cfg.train.checkpoint = checkpoint_path

model = generate_model(cfg)
model.eval()

# === Load image ===
img_path = "dataset_1/test/AMD/amd_1163930_2.jpg"   #change this to your local file used
img = Image.open(img_path).convert("RGB")
input_size = cfg.data.input_size

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0).to(cfg.base.device)

# === Target layer ===
target_layers = [model.blocks[6][0].conv_dw]

# === Normalized RGB image for overlay ===
rgb_img = np.array(img.resize((input_size, input_size))) / 255.0
rgb_img = rgb_img.astype(np.float32)

# === Run CAM ===
cam = EigenGradCAM(model=model, target_layers=target_layers)

if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

outputs = model(input_tensor)
pred_class = outputs.argmax().item()

grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])
grayscale_cam = grayscale_cam[0, :]  # shape: [H, W]

# === Invert and apply colormap ===
cam_uint8 = 255 - np.uint8(255 * grayscale_cam)  # invert grayscale
heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# === Overlay CAM on image ===
overlayed_image = (heatmap * 0.4 + rgb_img * 255 * 0.6).astype(np.uint8)

# === Save result ===
output_path = "eigenplus-result.png"
cv2.imwrite(output_path, overlayed_image)
print(f"Saved XAI image to {output_path}")
