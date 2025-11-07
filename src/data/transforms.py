# src/data/transforms.py
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

class BasicTransform:
    def __init__(self, size=512):
        self.img_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        image = self.img_transform(image)
        mask = mask.astype(np.int32)  # ✅ 转换为 Pillow 可接受的类型
        mask = torch.tensor(np.array(
            Image.fromarray(mask).resize((512, 512), resample=Image.Resampling.NEAREST)
        ), dtype=torch.long)
        sample["image"], sample["mask"] = image, mask
        return sample