import torch
from ultralytics import YOLO

def load_yolo(path, device):
    model = YOLO(path)
    model.model.eval()
    return model

class DoubleConv(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(i, o, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(o, o, 3, padding=1),
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
        x = self.u1(torch.cat([self.up(c4), c3], 1))
        x = self.u2(torch.cat([self.up(x), c2], 1))
        x = self.u3(torch.cat([self.up(x), c1], 1))
        return torch.sigmoid(self.final(x))

def load_unet(path, device):
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
