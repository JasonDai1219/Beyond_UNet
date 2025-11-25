# src/train.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset_foodseg import FoodSegDataset, load_foodseg103_splits
from src.data.transforms import BasicTransform
from src.models.unet import UNet
from src.models.losses import TotalLoss

# =========================
# Train one epoch
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

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} avg train loss: {avg_loss:.4f}")
    return avg_loss


# =========================
# Validation loop
# =========================

def validate(model, loader, loss_fn, device, epoch, writer=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            loss, _ = loss_fn(preds, masks, images)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} avg val loss: {avg_loss:.4f}")

    if writer:
        writer.add_scalar("Loss/val", avg_loss, epoch)

    return avg_loss


# =========================
# Test loop
# =========================

def test(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            loss, _ = loss_fn(preds, masks, images)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"✅ Final Test Loss: {avg_loss:.6f}")
    return avg_loss


# =========================
# Main training entry
# =========================

def main(cfg_path="configs/config_foodseg.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- Device ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🔥 Using Apple Metal GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("⚡ Using NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("🐢 Using CPU")

    # =========================
    # Dataset split (80/10/10)
    # =========================
    train_hf, val_hf, test_hf = load_foodseg103_splits(
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )

    transform = BasicTransform(size=cfg["dataset"]["image_size"])

    ds_train = FoodSegDataset(
        train_hf,
        transform=transform,
        compute_reflect=cfg["dataset"]["compute_reflect"],
        reflect_threshold=cfg["dataset"]["reflect_threshold"]
    )

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
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"]
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"]
    )

    # =========================
    # Model & Loss
    # =========================
    model = UNet(n_classes=104).to(device)
    loss_fn = TotalLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    # =========================
    # Logging
    # =========================
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)

    writer = None
    if cfg["logging"]["use_tensorboard"]:
        writer = SummaryWriter(cfg["logging"]["log_dir"])

    # =========================
    # Training Loop
    # =========================
    best_val_loss = float("inf")
    best_model_path = os.path.join(cfg["logging"]["checkpoint_dir"], "best_model.pt")

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, writer
        )

        val_loss = validate(
            model, val_loader, loss_fn, device, epoch, writer
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 New best model saved (val loss = {best_val_loss:.4f})")

        # 定期保存 checkpoint
        if (epoch + 1) % cfg["training"]["save_interval"] == 0:
            ckpt_path = os.path.join(
                cfg["logging"]["checkpoint_dir"],
                f"unet_epoch{epoch + 1}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # =========================
    # Final Test Evaluation
    # =========================
    print("🚀 Evaluating on test set...")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss = test(model, test_loader, loss_fn, device)

    if writer:
        writer.add_scalar("Loss/test", test_loss)
        writer.close()


if __name__ == "__main__":
    main()