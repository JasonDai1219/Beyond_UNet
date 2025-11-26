# src/train.py
import os
import yaml
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset_foodseg import FoodSegDataset, load_foodseg103_splits
from src.data.transforms import BasicTransform
from src.models.unet import UNet
from src.models.losses import TotalLoss


# =========================
# 固定随机种子（可复现）
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# =========================
# mIoU 计算
# =========================

def compute_miou(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        intersection = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()

        if union == 0:
            continue
        ious.append(intersection / union)

    return sum(ious) / len(ious) if len(ious) > 0 else 0.0


# =========================
# 实验结果写入 TXT
# =========================

def log_experiment(cfg, best_val_miou, test_miou, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "a") as f:
        f.write("="*60 + "\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Loss mode: {cfg['training']['loss_mode']}\n")
        f.write(f"epochs: {cfg['training']['epochs']}\n")
        f.write(f"lr: {cfg['training']['lr']}\n")
        f.write(f"batch_size: {cfg['dataset']['batch_size']}\n")
        f.write(f"alpha: {cfg['training']['alpha']}\n")
        f.write(f"beta: {cfg['training']['beta']}\n")
        f.write(f"lambda_edge: {cfg['training']['lambda_edge']}\n")
        f.write(f"lambda_reflect: {cfg['training']['lambda_reflect']}\n")
        f.write(f"sigma_edge: {cfg['training']['sigma_edge']}\n")
        f.write(f"BEST Val mIoU: {best_val_miou:.4f}\n")
        f.write(f"TEST mIoU: {test_miou:.4f}\n\n")


# =========================
# 主流程
# =========================

def main(cfg_path="configs/config_foodseg.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ===== Drive 路径（永不丢失）=====
    BASE_DIR = "/content/drive/MyDrive/foodseg_logs"
    LOG_TXT_PATH = os.path.join(BASE_DIR, "experiments_log.txt")
    BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
    TENSORBOARD_DIR = os.path.join(BASE_DIR, "tensorboard")

    os.makedirs(BASE_DIR, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Using device: {device}")

    # Dataset
    train_hf, val_hf, test_hf = load_foodseg103_splits(0.8, 0.1, seed=42)
    transform = BasicTransform(size=cfg["dataset"]["image_size"])

    ds_train = FoodSegDataset(train_hf, transform=transform)
    ds_val = FoodSegDataset(val_hf, transform=transform)
    ds_test = FoodSegDataset(test_hf, transform=transform)

    train_loader = DataLoader(ds_train, batch_size=cfg["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg["dataset"]["batch_size"], shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=cfg["dataset"]["batch_size"], shuffle=False)

    model = UNet(n_classes=104).to(device)
    loss_fn = TotalLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    writer = SummaryWriter(TENSORBOARD_DIR)

    best_val_miou = 0.0

    # ---------------- TRAIN ----------------
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(imgs)
            loss, _ = loss_fn(preds, masks, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_miou = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                preds = model(imgs)
                total_miou += compute_miou(preds, masks, 104)

        val_miou = total_miou / len(val_loader)
        print(f"Epoch {epoch} | Val mIoU: {val_miou:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🌟 Best model updated: {best_val_miou:.4f}")

    # ---------------- TEST ----------------
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    total_test_miou = 0

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(imgs)
            total_test_miou += compute_miou(preds, masks, 104)

    test_miou = total_test_miou / len(test_loader)
    print(f"✅ Final Test mIoU: {test_miou:.4f}")

    # 写入实验记录
    log_experiment(cfg, best_val_miou, test_miou, LOG_TXT_PATH)
    writer.close()


if __name__ == "__main__":
    main()