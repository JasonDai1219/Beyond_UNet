# src/data/dataset_foodseg.py
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random

class FoodSegDataset(Dataset):
    def __init__(self, hf_subset, transform=None, compute_reflect=False, reflect_threshold=220):
        self.dataset = hf_subset
        self.transform = transform
        self.compute_reflect = compute_reflect
        self.reflect_threshold = reflect_threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        mask = np.array(self.dataset[idx]["label"], dtype=np.int64)

        sample = {
            "image": image,
            "mask": mask
        }

        if self.compute_reflect:
            gray = np.array(image.convert("L"))
            reflect_ratio = (gray > self.reflect_threshold).sum() / gray.size
            sample["reflect_ratio"] = reflect_ratio

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_foodseg103_splits(train_ratio=0.8, val_ratio=0.1, seed=42):
    ds = load_dataset("EduardoPacheco/FoodSeg103")["train"]

    # 固定随机性
    random.seed(seed)
    np.random.seed(seed)

    train_val = ds.train_test_split(test_size=(1 - train_ratio), seed=seed)
    train_ds = train_val["train"]
    temp_ds = train_val["test"]

    relative_val = val_ratio / (1 - train_ratio)
    val_test = temp_ds.train_test_split(test_size=(1 - relative_val), seed=seed)
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    return train_ds, val_ds, test_ds