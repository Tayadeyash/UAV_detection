"""Pascal VOC dataset loader with safe augmentation for UAV detection."""
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class VOCDataset(Dataset):
    def __init__(self, root, train=False, image_size=800):
        self.root = Path(root)
        self.train = train
        self.image_size = image_size
        self.items = sorted(self.root.glob("*.xml"))
        if not self.items:
            raise FileNotFoundError(f"No XML annotations found in {self.root}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        xml_path = self.items[idx]
        root = ET.parse(xml_path).getroot()
        image_name = root.findtext("filename") or (xml_path.stem + ".jpg")
        image_path = xml_path.with_name(image_name)
        if not image_path.exists():
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                p = xml_path.with_suffix(ext)
                if p.exists(): image_path = p; break
        image = Image.open(image_path).convert("RGB")
        boxes = []
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            if b is None: continue
            xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
            xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
            if xmax > xmin and ymax > ymin: boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                w, _ = image.size
                if len(boxes):
                    x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
                    boxes[:, 0], boxes[:, 2] = w - x2, w - x1
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.12: image = image.filter(ImageFilter.GaussianBlur(radius=0.8))

        image = TF.to_tensor(image)
        _, h, w = image.shape
        scale = self.image_size / max(h, w)
        if scale != 1:
            nh, nw = round(h * scale), round(w * scale)
            image = TF.resize(image, [nh, nw])
            if len(boxes): boxes *= scale

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))
