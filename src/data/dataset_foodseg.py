# src/data/dataset_foodseg.py
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch

# src/data/dataset_foodseg.py
class FoodSegDataset(Dataset):
    def __init__(self, split="train", transform=None,
                 compute_reflect=False, reflect_threshold=220):
        self.dataset = load_dataset("EduardoPacheco/FoodSeg103")[split]
        self.transform = transform
        self.compute_reflect = compute_reflect
        self.reflect_threshold = reflect_threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        mask = np.array(self.dataset[idx]["label"], dtype=np.int64)
        sample = {"image": image, "mask": mask}

        # 可选：计算高亮比例
        if self.compute_reflect:
            gray = np.array(image.convert("L"))
            reflect_ratio = (gray > self.reflect_threshold).sum() / gray.size
            sample["reflect_ratio"] = reflect_ratio

        if self.transform:
            sample = self.transform(sample)
        return sample