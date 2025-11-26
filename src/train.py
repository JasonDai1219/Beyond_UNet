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
# Reproducibility
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
# mIoU computation
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

    return sum(ious) / len(ious) if ious else 0.0


# =========================
# Experiment Logger
# =========================
def log_experiment(cfg, best_val_miou, test_miou, save_path="experiments/experiments_log.txt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "a") as f:
        f.write("=" * 60 + "\n")
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
# Train / Val / Test
# =========================
def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch, writer=None):
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        preds = model(images)
        loss, parts = loss_fn(preds, masks, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if writer:
            global_step = epoch * len(loader) + step
            writer.add_scalar("Loss/train_total", loss.item(), global_step)
            for k, v in parts.items():
                writer.add_scalar(f"Loss/train_{k}", v, global_step)

    return total_loss / len(loader)


def validate(model, loader, loss_fn, device, num_classes, epoch, writer=None):
    model.eval()
    total_loss, total_miou = 0, 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            loss, _ = loss_fn(preds, masks, images)

            total_loss += loss.item()
            total_miou += compute_miou(preds, masks, num_classes)

    avg_loss = total_loss / len(loader)
    avg_miou = total_miou / len(loader)

    print(f"Epoch {epoch} avg val loss: {avg_loss:.4f} | mIoU: {avg_miou:.4f}")

    if writer:
        writer.add_scalar("Loss/val", avg_loss, epoch)
        writer.add_scalar("Metric/mIoU_val", avg_miou, epoch)

    return avg_loss, avg_miou


def test(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss, total_miou = 0, 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            loss, _ = loss_fn(preds, masks, images)

            total_loss += loss.item()
            total_miou += compute_miou(preds, masks, num_classes)

    return total_loss / len(loader), total_miou / len(loader)


# =========================
# Main
# =========================
def main(cfg_path="configs/config_foodseg.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Using {device}")

    train_hf, val_hf, test_hf = load_foodseg103_splits(0.8, 0.1, 42)
    transform = BasicTransform(size=cfg["dataset"]["image_size"])

    train_loader = DataLoader(FoodSegDataset(train_hf, transform=transform), batch_size=cfg["dataset"]["batch_size"], shuffle=True, num_workers=cfg["dataset"]["num_workers"])
    val_loader = DataLoader(FoodSegDataset(val_hf, transform=transform), batch_size=cfg["dataset"]["batch_size"], shuffle=False, num_workers=cfg["dataset"]["num_workers"])
    test_loader = DataLoader(FoodSegDataset(test_hf, transform=transform), batch_size=cfg["dataset"]["batch_size"], shuffle=False, num_workers=cfg["dataset"]["num_workers"])

    model = UNet(n_classes=104).to(device)
    loss_fn = TotalLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    writer = SummaryWriter(cfg["logging"]["log_dir"]) if cfg["logging"]["use_tensorboard"] else None

    best_val_miou = 0.0
    best_model_path = os.path.join(cfg["logging"]["checkpoint_dir"], "best_model.pt")

    for epoch in range(cfg["training"]["epochs"]):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, writer)
        _, val_miou = validate(model, val_loader, loss_fn, device, 104, epoch, writer)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 New best model saved (mIoU = {best_val_miou:.4f})")

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_miou = test(model, test_loader, loss_fn, device, 104)

    print(f"✅ Final Test Loss: {test_loss:.6f} | Test mIoU: {test_miou:.4f}")

    log_experiment(cfg, best_val_miou, test_miou)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()