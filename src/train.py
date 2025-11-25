# src/train.py
import os, yaml, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset_foodseg import FoodSegDataset, load_foodseg103_splits
from src.data.transforms import BasicTransform
from src.models.unet import UNet
from src.models.losses import TotalLoss

def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch, writer=None):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        preds = model(images)
        loss, parts = loss_fn(preds, masks, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if writer:
            writer.add_scalar("Loss/total", loss.item(), epoch * len(loader))
            for k, v in parts.items():
                writer.add_scalar(f"Loss/{k}", v, epoch * len(loader))

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
    return avg_loss

def main(cfg_path="configs/config_foodseg.yaml"):
    # ---- Load config ----
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
        print("🐢 Using CPU (slow)")

    # ---- Dataset & Loader ----
    # ✅ 标准化数据划分（80 / 10 / 10 + 固定 seed）
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

    ds_val = FoodSegDataset(
        val_hf,
        transform=transform
    )

    ds_test = FoodSegDataset(
        test_hf,
        transform=transform
    )

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

    # ---- Model & Loss ----
    model = UNet(n_classes=104).to(device)
    loss_fn = TotalLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"])
    lr = float(cfg["training"]["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Logging ----
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["logging"]["log_dir"]) if cfg["logging"]["use_tensorboard"] else None

    # ---- Training Loop ----
    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, writer)

        # ===== Validation =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                preds = model(images)
                loss, _ = loss_fn(preds, masks, images)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")

        if writer:
            writer.add_scalar("Loss/val", val_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % cfg["training"]["save_interval"] == 0:
            ckpt_path = os.path.join(
                cfg["logging"]["checkpoint_dir"],
                f"unet_epoch{epoch + 1}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    if writer:
        writer.close()

if __name__ == "__main__":
    main()