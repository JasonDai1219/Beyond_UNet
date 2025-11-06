# tests/test_dataset_loading.py
from src.data.dataset_foodseg import FoodSegDataset
from src.data.transforms import BasicTransform
from torch.utils.data import DataLoader

def test_dataset():
    ds = FoodSegDataset(split="train", transform=BasicTransform(), compute_reflect=True)
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(dl))
    print(batch["image"].shape, batch["mask"].shape, batch["reflect_ratio"])

if __name__ == "__main__":
    test_dataset()