# src/data/transforms.py

from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class BasicTransform:
    def __init__(self, size=512):
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.NEAREST)
        ])

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        image = self.image_transform(image)
        mask = Image.fromarray(mask.astype("uint8"))
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        sample["image"] = image
        sample["mask"] = mask
        return sample