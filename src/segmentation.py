import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class DoubleConv(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, 3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(1, 64)
        self.p1 = torch.nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128)
        self.p2 = torch.nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256)
        self.p3 = torch.nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512)
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.u1 = DoubleConv(512+256, 256)
        self.u2 = DoubleConv(256+128, 128)
        self.u3 = DoubleConv(128+64, 64)
        self.final = torch.nn.Conv2d(64, 1, 1)
    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.p1(c1))
        c3 = self.d3(self.p2(c2))
        c4 = self.d4(self.p3(c3))
        u1 = self.u1(torch.cat([self.up(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up(u1), c2], dim=1))
        u3 = self.u3(torch.cat([self.up(u2), c1], dim=1))
        return torch.sigmoid(self.final(u3))

def load_unet(model_path):
    # Map to CPU for your laptop
    try:
        model = SimpleUNet()
        state = torch.load(str(model_path), map_location="cpu")
        try:
            model.load_state_dict(state)
        except Exception:
            # sometimes state dict nested
            try:
                model.load_state_dict(state["model_state_dict"])
            except Exception:
                pass
        model.eval()
        return model
    except Exception as e:
        return None

def run_unet(model, image_bgr):
    """
    image_bgr: numpy BGR image (H,W,3)
    returns: mask_resized uint8 of shape (H,W) with 0/255
    """
    H, W = image_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("L")
    transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    t = transform(pil).unsqueeze(0)  # (1,1,256,256)
    with torch.no_grad():
        pred = model(t)
    pred = pred.squeeze().cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    if pred_mask.ndim > 2:
        pred_mask = pred_mask[:, :, 0]
    mask_resized = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 127).astype(np.uint8) * 255
    return mask_resized
