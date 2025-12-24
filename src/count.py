from google.colab import drive
drive.mount('/content/drive')

ROOT = "/content/drive/MyDrive/KidneyStoneX"
DATA_DIR = f"{ROOT}/data"
PROCESSED = f"{DATA_DIR}/processed"
TRAIN = f"{DATA_DIR}/train"
VAL = f"{DATA_DIR}/val"
TEST = f"{DATA_DIR}/test"
MODELS = f"{ROOT}/models"
RESULTS = f"{ROOT}/results"

import os
for p in [ROOT, DATA_DIR, PROCESSED, TRAIN, VAL, TEST, MODELS, RESULTS]:
    os.makedirs(p, exist_ok=True)

print("Folders ready:", ROOT)
!mkdir -p /content/kidx_data

!unzip "/content/drive/MyDrive/KidneyStoneX/data/Original.zip" -d "/content/kidx_data/"

!unzip "/content/drive/MyDrive/KidneyStoneX/data/Augmented.zip" -d "/content/kidx_data/"
import os, shutil

base = "/content/kidx_data"
merged = "/content/kidx_data/merged"

os.makedirs(f"{merged}/Stone", exist_ok=True)
os.makedirs(f"{merged}/Non_Stone", exist_ok=True)

stone_folders = [
    f"{base}/Original/Stone",
    f"{base}/Augmented/Stone"
]

for folder in stone_folders:
    for f in os.listdir(folder):
        shutil.copy(os.path.join(folder, f), f"{merged}/Stone")

nonstone_folders = [
    f"{base}/Original/Non-Stone",
    f"{base}/Augmented/Non-Stone"
]

for folder in nonstone_folders:
    for f in os.listdir(folder):
        shutil.copy(os.path.join(folder, f), f"{merged}/Non_Stone")

print("✅ Dataset merged successfully")
print("Stone images:", len(os.listdir(f"{merged}/Stone")))
print("Non-Stone images:", len(os.listdir(f"{merged}/Non_Stone")))
import torch
print(torch.cuda.is_available())
import os, random, shutil

merged = "/content/kidx_data/merged"
train_dir = "/content/kidx_data/train"
val_dir = "/content/kidx_data/val"

# make dirs
for d in [train_dir, val_dir]:
    os.makedirs(f"{d}/Stone", exist_ok=True)
    os.makedirs(f"{d}/Non_Stone", exist_ok=True)

# list paths
stone = [os.path.join(merged, "Stone", f) for f in os.listdir(f"{merged}/Stone")]
nonstone = [os.path.join(merged, "Non_Stone", f) for f in os.listdir(f"{merged}/Non_Stone")]

random.seed(42)
random.shuffle(stone)
random.shuffle(nonstone)

# 80% train / 20% val
cut_stone = int(0.8 * len(stone))
cut_nonstone = int(0.8 * len(nonstone))

train_stone = stone[:cut_stone]
val_stone = stone[cut_stone:]

train_nonstone = nonstone[:cut_nonstone]
val_nonstone = nonstone[:cut_nonstone]

# copy
def copy_files(files, dest):
    for f in files:
        shutil.copy(f, dest)

copy_files(train_stone, f"{train_dir}/Stone")
copy_files(val_stone, f"{val_dir}/Stone")

copy_files(train_nonstone, f"{train_dir}/Non_Stone")
copy_files(val_nonstone, f"{val_dir}/Non_Stone")

print("✅ Split complete")
print("Train Stone:", len(os.listdir(f'{train_dir}/Stone')))
print("Val Stone:", len(os.listdir(f'{val_dir}/Stone')))
print("Train Non-Stone:", len(os.listdir(f'{train_dir}/Non_Stone')))
print("Val Non-Stone:", len(os.listdir(f'{val_dir}/Non_Stone')))
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

train_dir = "/content/kidx_data/train"
val_dir = "/content/kidx_data/val"

# transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds   = datasets.ImageFolder(val_dir, transform=transform)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False)

# model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# training loop
for epoch in range(3):  # keep small first to test setup
    model.train()
    running_loss = 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_dl):.4f}")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        output = model(imgs)
        _, preds = torch.max(output, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Validation Accuracy: {100 * correct / total:.2f}%")
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print("YOLO ready ✅")
import os

base = "/content/kidx_data"

for split in ["train", "val"]:
    img_path = f"{base}/{split}"
    label_path = f"{base}/{split}_labels"
    os.makedirs(label_path, exist_ok=True)
    print("Created:", label_path)
import cv2, os, numpy as np

def detect_stone_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return x, y, w, h
def create_yolo_labels(img_folder, label_folder):
    for cls in os.listdir(img_folder):
        class_folder = f"{img_folder}/{cls}"
        for img_name in os.listdir(class_folder):
            img_path = f"{class_folder}/{img_name}"
            img = cv2.imread(img_path)
            if img is None: continue

            bbox = detect_stone_bbox(img)

            h, w = img.shape[:2]
            label_path = f"{label_folder}/{img_name.replace('.png','.txt').replace('.jpg','.txt')}"

            with open(label_path, "w") as f:
                if cls.lower() == "stone" and bbox is not None:
                    x, y, bw, bh = bbox
                    # convert to YOLO format (cx, cy, w, h normalized)
                    cx = (x + bw/2) / w
                    cy = (y + bh/2) / h
                    nw = bw / w
                    nh = bh / h

                    f.write(f"0 {cx} {cy} {nw} {nh}\n")
                else:
                    pass
create_yolo_labels("/content/kidx_data/train", "/content/kidx_data/train_labels")
create_yolo_labels("/content/kidx_data/val", "/content/kidx_data/val_labels")
!rm -rf /content/kidx_data/train/images
!rm -rf /content/kidx_data/train/labels
!rm -rf /content/kidx_data/val/images
!rm -rf /content/kidx_data/val/labels
import os

for split in ["train", "val"]:
    os.makedirs(f"/content/kidx_data/{split}/images", exist_ok=True)
    os.makedirs(f"/content/kidx_data/{split}/labels", exist_ok=True)

print("✅ Fresh YOLO folders created")
import os, shutil

base="/content/kidx_data"

def move_files(img_dir, lbl_dir, split):
    for cls in os.listdir(img_dir):
        class_path = f"{img_dir}/{cls}"

        if not os.path.isdir(class_path) or cls in ["images", "labels"]:
            continue

        for img in os.listdir(class_path):
            if img.startswith("."):
                continue

            src_img = f"{class_path}/{img}"
            dest_img = f"{base}/{split}/images/{img}"

            # only copy if not already copied
            if not os.path.exists(dest_img):
                shutil.copy(src_img, dest_img)

            img_base = os.path.splitext(img)[0]
            label_name = f"{img_base}.txt"

            src_label = f"{lbl_dir}/{label_name}"
            dest_label = f"{base}/{split}/labels/{label_name}"

            if not os.path.exists(dest_label):
                if os.path.exists(src_label):
                    shutil.copy(src_label, dest_label)
                else:
                    open(dest_label, "w").close()

move_files(f"{base}/train", f"{base}/train_labels", "train")
move_files(f"{base}/val", f"{base}/val_labels", "val")

print("✅ Images & labels moved safely without duplicates")
%%writefile kidney_stone.yaml
path: /content/kidx_data

train: train/images
val: val/images

names:
  0: stone
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="kidney_stone.yaml",
    imgsz=512,
    epochs=15,
    batch=16,
    workers=2,
    patience=5,
    device=0
)
from ultralytics import YOLO

model = YOLO("/content/runs/detect/train2/weights/best.pt")
print("✅ Model loaded!")
import os, random

src = "/content/kidx_data/val/images"
tmp = "/content/val_sample"
os.makedirs(tmp, exist_ok=True)

files = random.sample(os.listdir(src), 50)
for f in files:
    os.symlink(os.path.join(src, f), os.path.join(tmp, f))
model.predict(
    source="/content/val_sample",
    save=True,
    conf=0.25,
    project="/content/yolo_predictions",
    name="sample_results"
)
from PIL import Image
import matplotlib.pyplot as plt
import os, random

results_dir = "/content/yolo_predictions/sample_results"
imgs = random.sample([f for f in os.listdir(results_dir) if f.endswith(".jpg")], 6)

plt.figure(figsize=(12,8))
for i, img in enumerate(imgs):
    im = Image.open(f"{results_dir}/{img}")
    plt.subplot(2,3,i+1)
    plt.imshow(im)
    plt.axis('off')
plt.show()
from google.colab import drive
drive.mount('/content/drive')

backup_dir = "/content/drive/MyDrive/kidney_stone_project_backup"
!mkdir -p $backup_dir

# ✅ Backup YOLO training folder (weights, metrics, plots)
!cp -r /content/runs/detect/train2 $backup_dir/

# ✅ Backup classifier weights (if exists)
!cp /content/kidney_stone_classifier.pth $backup_dir/ 2>/dev/null || echo "Classifier file not found, skipped ✅"

# ✅ Backup dataset splits (optional but useful)
!cp -r /content/kidx_data $backup_dir/

# ✅ Backup predictions if folder exists
pred_dir = "/content/yolo_predictions"
!if [ -d $pred_dir ]; then zip -r $backup_dir/stone_predictions.zip $pred_dir; else echo "No predictions folder found ✅"; fi

print("✅ Backup completed! Your models & results are safe in Google Drive ✅")
!unzip "/content/drive/MyDrive/labelled.zip" -d /content/datasets/
!ls /content/datasets/labelled/images | head
yaml_text = """
path: /content/datasets/labelled
train: images
val: images  # same set; small dataset
nc: 1
names: ['stone']
"""
with open("/content/kidney_stone.yaml", "w") as f:
    f.write(yaml_text)
print("✅ YAML ready")
restore_dir = "/content/drive/MyDrive/kidney_stone_project_backup"

!mkdir -p /content/runs/detect/
!cp -r "$restore_dir/train2" /content/runs/detect/

#!cp -r "$restore_dir/kidx_data" /content/

#!cp "$restore_dir/kidney_stone_classifier.pth" /content/ 2>/dev/null

print("✅ Restore complete")
!mkdir -p /content/runs/detect/train/weights
!cp "/content/drive/MyDrive/YOLO_stone_final.pt" "/content/runs/detect/train/weights/best.pt"
print("✅ YOLO model restored successfully and ready for use!")
!yolo detect train \
  data=/content/kidney_stone.yaml \
  model=/content/runs/detect/train2/weights/best.pt \
  epochs=10 \
  imgsz=512 \
  batch=8
!cp /content/runs/detect/train/weights/best.pt "/content/drive/MyDrive/YOLO_stone_final.pt"
print("✅ Model saved to Drive")
from ultralytics import YOLO
model = YOLO("/content/runs/detect/train/weights/best.pt")

test_dir = "/content/kidx_data/Original/Stone"
results = model.predict(source=test_dir, conf=0.3, save=True)
import random
import matplotlib.pyplot as plt
from PIL import Image

pred_dir = "/content/runs/detect/predict5"

files = [f for f in os.listdir(pred_dir) if f.lower().endswith((".jpg", ".png"))]
samples = random.sample(files, min(6, len(files)))

plt.figure(figsize=(12, 8))
for i, f in enumerate(samples):
    plt.subplot(2, 3, i + 1)
    img = Image.open(os.path.join(pred_dir, f))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f)
plt.show()
from ultralytics import YOLO
import os, shutil

model_path = "/content/runs/detect/train/weights/best.pt"
input_dir = "/content/kidx_data/Original/Stone"
output_dir = "/content/auto_labels"

model = YOLO(model_path)

results = model.predict(
    source=input_dir,
    conf=0.3,
    save=True,
    save_txt=True,
    imgsz=512
)

pred_dir = results[0].save_dir if results else "runs/detect/predict"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/labels", exist_ok=True)

if os.path.exists(pred_dir):

    for f in os.listdir(pred_dir):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.copy(os.path.join(pred_dir, f), f"{output_dir}/images/")

    labels_dir = os.path.join(pred_dir, "labels")
    if os.path.exists(labels_dir):
        for f in os.listdir(labels_dir):
            if f.lower().endswith(".txt"):
                shutil.copy(os.path.join(labels_dir, f), f"{output_dir}/labels/")
    print(f"✅ Auto-labels saved in: {output_dir}")
else:
    print("⚠️ Prediction folder not found.")

num_labels = len(os.listdir(f"{output_dir}/labels"))
num_images = len(os.listdir(f"{output_dir}/images"))
print(f"🩻 Auto-labeled {num_images} images with {num_labels} label files")
import shutil, os

manual_path = "/content/datasets/labelled"
auto_path = "/content/auto_labels"
final_path = "/content/datasets/final"

os.makedirs(f"{final_path}/images", exist_ok=True)
os.makedirs(f"{final_path}/labels", exist_ok=True)

for folder in ["images", "labels"]:
    src = f"{manual_path}/{folder}"
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), f"{final_path}/{folder}/")

for folder in ["images", "labels"]:
    src = f"{auto_path}/{folder}"
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), f"{final_path}/{folder}/")

print(f"✅ Final merged dataset ready at: {final_path}")
print(f"Total images: {len(os.listdir(f'{final_path}/images'))}")
print(f"Total labels: {len(os.listdir(f'{final_path}/labels'))}")
yaml_text = """
path: /content/datasets/final
train: images
val: images
nc: 1
names: ['stone']
"""
with open("/content/kidney_stone.yaml", "w") as f:
    f.write(yaml_text)
print("✅ kidney_stone.yaml created")

!yolo detect train \
  data=/content/kidney_stone.yaml \
  model="/content/drive/MyDrive/YOLO_stone_final.pt" \
  epochs=20 \
  imgsz=512 \
  batch=8
!cp /content/runs/detect/train3/weights/best.pt "/content/drive/MyDrive/YOLO_stone_final_v2.pt"
print("✅ Final YOLO stone model (v2) saved to Drive")
import os
img_dir = "/content/datasets/final/images"
lbl_dir = "/content/datasets/final/labels"

labeled = 0
for f in os.listdir(img_dir):
    name = os.path.splitext(f)[0] + ".txt"
    if os.path.exists(os.path.join(lbl_dir, name)):
        labeled += 1
print(f"{labeled} / {len(os.listdir(img_dir))} images have labels")
import shutil, os
src_img = "/content/datasets/final/images"
src_lbl = "/content/datasets/final/labels"
clean_path = "/content/datasets/final_clean"
os.makedirs(f"{clean_path}/images", exist_ok=True)
os.makedirs(f"{clean_path}/labels", exist_ok=True)

for lbl in os.listdir(src_lbl):
    name = os.path.splitext(lbl)[0]
    img_file = None
    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = os.path.join(src_img, name + ext)
        if os.path.exists(candidate):
            img_file = candidate
            break
    if img_file:
        shutil.copy(img_file, f"{clean_path}/images/")
        shutil.copy(os.path.join(src_lbl, lbl), f"{clean_path}/labels/")

print("✅ Clean dataset ready at:", clean_path)
print("Images:", len(os.listdir(f'{clean_path}/images')),\
      "Labels:", len(os.listdir(f'{clean_path}/labels')))
yaml_text = """
path: /content/datasets/final_clean
train: images
val: images
nc: 1
names: ['stone']
"""
with open("/content/kidney_stone.yaml", "w") as f:
    f.write(yaml_text)
!yolo detect train \
  data=/content/kidney_stone.yaml \
  model="/content/drive/MyDrive/YOLO_stone_final.pt" \
  epochs=10 \
  imgsz=512 \
  batch=8
from ultralytics import YOLO
model = YOLO("/content/runs/detect/train4/weights/best.pt")
results = model.predict(source="/content/kidx_data/Original/Stone", conf=0.3, save=True)
import random
import matplotlib.pyplot as plt
from PIL import Image

pred_dir = "/content/runs/detect/predict5"

files = [f for f in os.listdir(pred_dir) if f.lower().endswith((".jpg", ".png"))]
samples = random.sample(files, min(6, len(files)))

plt.figure(figsize=(12, 8))
for i, f in enumerate(samples):
    plt.subplot(2, 3, i + 1)
    img = Image.open(os.path.join(pred_dir, f))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f)
plt.show()
!cp /content/runs/detect/train4/weights/best.pt "/content/drive/MyDrive/YOLO_stone_final_v3.pt"
print("✅ Model saved to Drive (YOLO_stone_final_v3.pt)")
!mkdir -p /content/runs/detect/train4/weights
!cp "/content/drive/MyDrive/YOLO_stone_final_v3.pt" "/content/runs/detect/train4/weights/best.pt"
print("✅ YOLO model restored successfully and ready for use!")
!pip install -q torch torchvision opencv-python-headless matplotlib numpy tqdm albumentations segmentation-models-pytorch

import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
input_dir = "/content/kidx_data/Original/Stone"
mask_dir = "/content/kidney_masks"
os.makedirs(mask_dir, exist_ok=True)

for img_name in tqdm(os.listdir(input_dir)):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img = cv2.imread(os.path.join(input_dir, img_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(img_norm, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imwrite(os.path.join(mask_dir, img_name), opened)

print("✅ Kidney mask generation complete.")
class KidneyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = np.transpose(img, (2, 0, 1)).astype('float32') / 255.0
        mask = np.transpose(mask, (2, 0, 1)).astype('float32') / 255.0

        return torch.tensor(img), torch.tensor(mask)
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
])

train_imgs, val_imgs = train_test_split(os.listdir(input_dir), test_size=0.2, random_state=42)
train_dataset = KidneyDataset(input_dir, mask_dir, transform=transform)
val_dataset = KidneyDataset(input_dir, mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1).to(device)

loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/5] - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "/content/kidney_unet.pt")
print("✅ Kidney U-Net saved!")
model.eval()
with torch.no_grad():
    imgs, masks = next(iter(val_loader))
    preds = model(imgs.to(device))
    preds = (preds.sigmoid().cpu().numpy() > 0.5).astype(np.uint8)

plt.figure(figsize=(12,6))
for i in range(3):
    plt.subplot(3,3,i*3+1); plt.imshow(np.transpose(imgs[i].cpu().numpy(), (1,2,0))); plt.title("Original")
    plt.subplot(3,3,i*3+2); plt.imshow(masks[i][0].cpu(), cmap='gray'); plt.title("Mask")
    plt.subplot(3,3,i*3+3); plt.imshow(preds[i][0], cmap='gray'); plt.title("Predicted")
plt.tight_layout()
plt.show()
!cp "/content/drive/MyDrive/kidney_unet.pt" "/content/kidney_unet.pt"
print("✅ U-Net restored successfully and ready for use!")
import cv2, os, numpy as np
from skimage import measure, morphology

src = "/content/kidx_data/Original/Stone"
balanced_mask_dir = "/content/kidney_masks_balanced"
os.makedirs(balanced_mask_dir, exist_ok=True)

for img_name in os.listdir(src):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = cv2.imread(os.path.join(src, img_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.inRange(img, 55, 185)     # balanced soft-tissue range

    mask = morphology.remove_small_objects(mask > 0, min_size=1200)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    # keep 2 biggest blobs (likely kidneys)
    lbl = measure.label(mask)
    props = measure.regionprops(lbl)
    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)[:2]
    newmask = np.zeros_like(mask)
    for p in props_sorted:
        newmask[lbl == p.label] = 255

    newmask = cv2.morphologyEx(newmask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    cv2.imwrite(os.path.join(balanced_mask_dir, img_name), newmask)

print("✅ Balanced masks saved in:", balanced_mask_dir)
import os, shutil, random

# existing folders
img_sources = [
    "/content/kidx_data/train/Stone",
    "/content/kidx_data/val/Stone",
    "/content/kidx_data/merged/Stone",
    "/content/kidx_data/Original/Stone",
    "/content/kidx_data/Augmented/Stone"
]

mask_src = "/content/kidney_masks_balanced"  # folder where your refined masks are
final_img_dir = "/content/kidx_data/Stone/images"
final_mask_dir = "/content/kidx_data/Stone/masks"

os.makedirs(final_img_dir, exist_ok=True)
os.makedirs(final_mask_dir, exist_ok=True)

# collect all stone images (jpg/png/jpeg)
stone_images = []
for src in img_sources:
    if os.path.exists(src):
        for f in os.listdir(src):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(src, f)
                dst_path = os.path.join(final_img_dir, f)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)
                    stone_images.append(f)

# copy only matching masks
copied_masks = 0
for f in stone_images:
    mask_path = os.path.join(mask_src, f)
    if os.path.exists(mask_path):
        shutil.copy(mask_path, os.path.join(final_mask_dir, f))
        copied_masks += 1

print(f"✅ Copied {len(stone_images)} stone images.")
print(f"✅ Copied {copied_masks} matching masks.")
print("Final dataset ready at /content/kidx_data/Stone/")
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np, os
from tqdm import tqdm

# -------- Dataset --------
class KidneyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                       and os.path.exists(os.path.join(mask_dir, f))]
        self.tfm = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("L")
        msk = Image.open(os.path.join(self.mask_dir, name)).convert("L")
        img = self.tfm(img)
        msk = self.tfm(msk)
        return img, msk

# -------- Model --------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(1, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.u1 = DoubleConv(512+256, 256)
        self.u2 = DoubleConv(256+128, 128)
        self.u3 = DoubleConv(128+64, 64)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        u1 = self.u1(torch.cat([self.up(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up(u1), c2], dim=1))
        u3 = self.u3(torch.cat([self.up(u2), c1], dim=1))
        return torch.sigmoid(self.final(u3))

# -------- Training --------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0004)

dataset = KidneyDataset("/content/kidx_data/Stone/images", "/content/kidx_data/Stone/masks", size=256)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"🧠 Training on {len(dataset)} paired images...")
for epoch in range(6):
    model.train(); total = 0
    for imgs, masks in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1}/6  Loss: {total/len(loader):.4f}")

torch.save(model.state_dict(), "/content/UNet_kidney_v5.pth")
print("✅ U-Net model saved → /content/UNet_kidney_v5.pth")
import matplotlib.pyplot as plt
import random

model.eval()
with torch.no_grad():
    samples = random.sample(os.listdir("/content/kidx_data/Stone/images"), 3)
    for s in samples:
        img_path = os.path.join("/content/kidx_data/Stone/images", s)
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        t = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(t).squeeze().cpu().numpy()
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("CT Image")
        plt.subplot(1,2,2); plt.imshow(pred>0.5, cmap="gray"); plt.title("Predicted Mask")
        plt.show()
import shutil

unet_path = "/content/UNet_kidney_v5.pth"

dst = "/content/drive/MyDrive/UNet_kidney_v5.pth"

shutil.copy(unet_path, dst)
print("✅ U-Net model saved to:", dst)
import torch

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

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(1, 64)
        self.p1 = torch.nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128)
        self.p2 = torch.nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256)
        self.p3 = torch.nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512)
        self.u1 = DoubleConv(512 + 256, 256)
        self.u2 = DoubleConv(256 + 128, 128)
        self.u3 = DoubleConv(128 + 64, 64)
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

yolo_model_path = "/content/runs/detect/train4/weights/best.pt"
unet_model_path = "/content/UNet_kidney_v5.pth"
stone_dir = "/content/kidx_data/Original/Stone"
output_dir = "/content/integrated_results1"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO(yolo_model_path)

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

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(1, 64)
        self.p1 = torch.nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128)
        self.p2 = torch.nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256)
        self.p3 = torch.nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512)
        self.u1 = DoubleConv(512 + 256, 256)
        self.u2 = DoubleConv(256 + 128, 128)
        self.u3 = DoubleConv(128 + 64, 64)
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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

unet = UNet().to(device)
unet.load_state_dict(torch.load(unet_model_path, map_location=device))
unet.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

for filename in os.listdir(stone_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(stone_dir, filename)
    img_pil = Image.open(img_path).convert("L")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = unet(img_tensor).squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    kernel_close = np.ones((5,5), np.uint8)
    kernel_dilate = np.ones((3,3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel_close)
    pred_mask = cv2.dilate(pred_mask, kernel_dilate, iterations=1)

    img_cv = np.array(img_pil)
    mask_resized = cv2.resize(pred_mask, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv2.INTER_NEAREST)

    if len(img_cv.shape) == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

    overlay = img_cv.copy()
    overlay[mask_resized > 127] = [180, 90, 90]
    overlay = cv2.addWeighted(img_cv, 0.8, overlay, 0.2, 0)

    results = yolo_model.predict(img_path, conf=0.3, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(overlay, "stone", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("✅ Integration completed!")
print(f"Results saved to: {output_dir}")
import matplotlib.pyplot as plt
import random
from PIL import Image
import os

result_dir = "/content/integrated_results1"

samples = random.sample(os.listdir(result_dir), min(6, len(os.listdir(result_dir))))

plt.figure(figsize=(15, 10))
for i, filename in enumerate(samples):
    img_path = os.path.join(result_dir, filename)
    img = Image.open(img_path)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(filename)
plt.tight_layout()
plt.show()
from google.colab import drive
drive.mount('/content/drive')

!zip -r /content/integrated_results1.zip /content/integrated_results1
!cp /content/integrated_results1.zip /content/drive/MyDrive/
print("✅ Zipped and saved to Drive as 'integrated_results1.zip'")
import os
import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO

yolo_model_path = "/content/runs/detect/train4/weights/best.pt"
input_dir = "/content/integrated_results1"
csv_output = "/content/stone_metrics.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(yolo_model_path)

data = []

for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    results = model.predict(img_path, conf=0.15, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    stone_count = len(boxes)
    avg_area = 0
    mean_intensity = float(np.mean(img))
    side = "Unknown"

    if stone_count > 0:
        areas = []
        x_positions = []
        for (x1, y1, x2, y2) in boxes:
            w, h = (x2 - x1), (y2 - y1)
            areas.append(w * h)
            x_positions.append((x1 + x2) / 2)

        avg_area = np.mean(areas)
        mean_x = np.mean(x_positions)
        side = "Left" if mean_x < img.shape[1] / 2 else "Right"

    data.append({
        "Image": img_name,
        "Stone_Count": stone_count,
        "Avg_BBox_Area": round(avg_area, 2),
        "Mean_Intensity": round(mean_intensity, 2),
        "Kidney_Side": side
    })

df = pd.DataFrame(data)
df.to_csv(csv_output, index=False)
print(f"✅ Quantification complete! Results saved to {csv_output}")

print(df.head())
import torch
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO

yolo_model = YOLO("/content/runs/detect/train4/weights/best.pt").to("cuda" if torch.cuda.is_available() else "cpu")
yolo_model.model.eval()

image_folder = "/content/kidx_data/Original/Stone"
sample_images = random.sample(
    [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))], 5
)

def show_gradcam(model, image_path, layer_name='model.9.cv2'):
    device = next(model.model.parameters()).device

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640)) / 255.0
    tensor_img = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    tensor_img.requires_grad = True

    fmap, grad = [], []
    def forward_hook(module, input, output): fmap.append(output)
    def backward_hook(module, grad_in, grad_out): grad.append(grad_out[0])

    target_layer = dict(model.model.named_modules())[layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    preds = model.model(tensor_img)
    loss = preds[0].mean()
    loss.backward()

    weights = torch.mean(grad[0], dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * fmap[0], dim=1).squeeze()
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam.detach().cpu().numpy(), (img_rgb.shape[1], img_rgb.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.show()

for path in sample_images[:3]:
    print(f"\n📸 Image: {os.path.basename(path)}")
    show_gradcam(yolo_model, path)
!mkdir -p /content/runs/detect/train4/weights
!cp "/content/drive/MyDrive/YOLO_stone_final_v3.pt" "/content/runs/detect/train4/weights/best.pt"
dataset_yaml = """
path: /content/datasets/labelled
train:
val: images

names:
  0: stone
"""

with open("/content/labelled.yaml", "w") as f:
    f.write(dataset_yaml)

print("✅ YAML created at /content/labelled.yaml")
from ultralytics import YOLO

model_path = "/content/runs/detect/train4/weights/best.pt"

model = YOLO(model_path)

metrics = model.val(data="/content/labelled.yaml", imgsz=512)



this is one part of the code I'll be sending you the other part. and i feel in this code alot of it is useless and repetetive. if I werr to train this on vs code so i would have made a .py file. if at all you can rewrite the code the clean one with no unnessasary steps then do that. 