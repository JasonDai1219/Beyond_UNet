# src/data/dataset_foodseg.py
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

# src/data/dataset_foodseg.py
class FoodSegDataset(Dataset):
    def __init__(self, split="train", img_transform=None, mask_transform=None,
                 compute_reflect=False, reflect_threshold=220):
        self.dataset = load_dataset("EduardoPacheco/FoodSeg103")[split]
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.compute_reflect = compute_reflect
        self.reflect_threshold = reflect_threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        # # Convert mask to NumPy, using int32 to avoid PIL errors
        mask = np.array(self.dataset[idx]["label"], dtype=np.int32)
        mask = torch.from_numpy(mask).long()  # (H, W)
        # sample = {"image": image, "mask": mask}

        # Optional: Calculate highlight ratio
        if self.compute_reflect:
            gray = np.array(image.convert("L"))
            reflect_ratio = (gray > self.reflect_threshold).sum() / gray.size
            # return reflect_ratio
            # sample["reflect_ratio"] = reflect_ratio

        if self.img_transform:
            image = self.img_transform(image)
        # mask transform (only support resize for now)
        if self.mask_transform:
            for t in self.mask_transform.transforms:
                if isinstance(t, transforms.Resize):
                    size = t.size
                    if isinstance(size, int):
                        size = (size, size)
                    mask = mask.unsqueeze(0).float()  # (1,H,W)
                    mask = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")  # (1,1,H,W)
                    mask = mask.squeeze(0).squeeze(0).long()  # -> (H,W)



        return image, mask