import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

def segment_kidney(unet, img_path, device):
    t = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("L")
    x = t(img).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = unet(x).squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask
