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

from jason_approach.src.data.dataset_foodseg import FoodSegDataset, load_foodseg103_splits
from jason_approach.src.data.transforms import BasicTransform
from jason_approach.src.models.unet import UNet
from jason_approach.src.models.simple_cnn import SimpleSegNet
from jason_approach.src.models.losses import TotalLoss

import argparse


# =========================
# Reproducibility Seed
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

    return 0.0 if len(ious) == 0 else sum(ious) / len(ious)


# =========================
# Train one epoch (AMP)
# =========================

def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch, writer=None):
    model.train()
    total_loss = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            preds = model(images)
            loss, parts = loss_fn(preds, masks, images)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if writer:
            global_step = epoch * len(loader) + step
            writer.add_scalar("Loss/train_total", loss.item(), global_step)
            for k, v in parts.items():
                writer.add_scalar(f"Loss/train_{k}", v, global_step)

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} avg train loss: {avg_loss:.4f}")
    return avg_loss


# =========================
# Validation
# =========================

def validate(model, loader, loss_fn, device, epoch, writer=None, num_classes=104):
    model.eval()
    total_loss = 0
    total_miou = 0

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


# =========================
# Test
# =========================

def test(model, loader, loss_fn, device, num_classes=104):
    model.eval()
    total_loss = 0
    total_miou = 0

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

    print(f"✅ Final Test Loss: {avg_loss:.6f} | Test mIoU: {avg_miou:.4f}")
    return avg_loss, avg_miou


# =========================
# Experiment Logger
# =========================

def log_experiment(cfg, best_val_miou, test_miou, save_path="experiments/experiments_log.txt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "a") as f:
        f.write("="*60 + "\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Model: {cfg['training'].get('model_type', 'unet')}\n")
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
# Main
# =========================

def main(cfg_path="configs/config_foodseg.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Using device: {device}")

    train_hf, val_hf, test_hf = load_foodseg103_splits(0.8, 0.1, seed=42)
    transform = BasicTransform(size=cfg["dataset"]["image_size"])

    ds_train = FoodSegDataset(train_hf, transform=transform)
    ds_val = FoodSegDataset(val_hf, transform=transform)
    ds_test = FoodSegDataset(test_hf, transform=transform)

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"]
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"]
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"]
    )

    model_type = cfg["training"].get("model_type", "unet")

    if model_type == "unet":
        model = UNet(n_classes=104).to(device)
    elif model_type == "simple_cnn":
        model = SimpleSegNet(num_classes=104).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    loss_fn = TotalLoss(
        alpha=cfg["training"]["alpha"],
        beta=cfg["training"]["beta"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    writer = SummaryWriter(cfg["logging"]["log_dir"]) if cfg["logging"]["use_tensorboard"] else None

    best_miou = 0.0
    patience = 10
    no_improve = 0
    best_model_path = os.path.join(cfg["logging"]["checkpoint_dir"], "best_model.pt")

    for epoch in range(cfg["training"]["epochs"]):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, writer)
        val_loss, val_miou = validate(model, val_loader, loss_fn, device, epoch, writer)

        scheduler.step(val_miou)

        if val_miou > best_miou:
            best_miou = val_miou
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 Best model updated (mIoU = {best_miou:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹ Early stopping triggered")
                break

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_miou = test(model, test_loader, loss_fn, device)

    log_experiment(cfg, best_miou, test_miou)

    if writer:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Training Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="jason_approach/configs/config_foodseg.yaml",
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["unet", "simple_cnn"],
        help="Override model type in config",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if args.model is not None:
        cfg["training"]["model_type"] = args.model
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch is not None:
        cfg["dataset"]["batch_size"] = args.batch

    # Write updated config to a temporary dictionary instead of modifying file
    main_cfg_path = "__temp_config_runtime.yaml"
    with open(main_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    # Launch training
    main(main_cfg_path)

    # Clean temp file
    os.remove(main_cfg_path)