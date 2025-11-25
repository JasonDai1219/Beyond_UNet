# src/train.py
import os, yaml, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset_foodseg import FoodSegDataset, load_foodseg103_splits
from src.data.transforms import BasicTransform
from src.data.weightmap_utils import build_weight_map
from src.models.unet import UNet
from src.models.losses import TotalLoss


def compute_weight_map_batch(images_cpu, masks_cpu,
                             lambda_edge=1.0,
                             lambda_reflect=1.0,
                             sigma=5.0,
                             num_classes=104):
    """
    images_cpu: (B, 3, H, W) torch tensor on CPU
    masks_cpu:  (B, H, W)   torch tensor on CPU
    返回: (B, H, W) 的 weight_map tensor (CPU)
    """
    weight_maps = []
    B = images_cpu.shape[0]

    for i in range(B):
        img_np = images_cpu[i].permute(1, 2, 0).numpy()      # (H,W,C)
        mask_np = masks_cpu[i].numpy()                       # (H,W)
        w_np = build_weight_map(
            image_np=img_np,
            mask_np=mask_np,
            lambda_edge=lambda_edge,
            lambda_reflect=lambda_reflect,
            sigma=sigma,
            num_classes=num_classes,
        )
        weight_maps.append(torch.from_numpy(w_np).float())

    weight_map = torch.stack(weight_maps, dim=0)  # (B,H,W)
    return weight_map


def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    epoch, writer=None,
                    lambda_edge=1.0,
                    lambda_reflect=1.0,
                    sigma=5.0,
                    num_classes=104):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        # 先在 CPU 上拿到图像 & mask 做 weight map
        images_cpu = batch["image"]          # (B,3,H,W), still on CPU
        masks_cpu = batch["mask"]            # (B,H,W)

        weight_map = compute_weight_map_batch(
            images_cpu, masks_cpu,
            lambda_edge=lambda_edge,
            lambda_reflect=lambda_reflect,
            sigma=sigma,
            num_classes=num_classes,
        )

        # 再搬到 device
        images = images_cpu.to(device)
        masks = masks_cpu.to(device)
        weight_map = weight_map.to(device)

        preds = model(images)
        loss, parts = loss_fn(preds, masks, images, weight_map)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if writer:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar("Loss/total", loss.item(), global_step)
            for k, v in parts.items():
                writer.add_scalar(f"Loss/{k}", v, global_step)

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

<<<<<<< Updated upstream
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
=======
    # ---- Dataset & Loader: 80/10/10 ----
    print("📂 Splitting FoodSeg103 into 80/10/10 ...")
    ds_train_raw, ds_val_raw, ds_test_raw = load_foodseg103_splits()
>>>>>>> Stashed changes

    transform = BasicTransform(size=cfg["dataset"]["image_size"])

    ds_train = FoodSegDataset(
        hf_subset=ds_train_raw,
        transform=transform,
        compute_reflect=cfg["dataset"]["compute_reflect"],
        reflect_threshold=cfg["dataset"]["reflect_threshold"],
    )
    ds_val = FoodSegDataset(
        hf_subset=ds_val_raw,
        transform=transform,
        compute_reflect=False,
    )
    ds_test = FoodSegDataset(
        hf_subset=ds_test_raw,
        transform=transform,
        compute_reflect=False,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
    )

    print(f"Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

    # ---- Hyper-params from config ----
    alpha = float(cfg["training"]["alpha"])
    beta = float(cfg["training"]["beta"])
    lambda_edge = float(cfg["training"].get("lambda_edge", 1.0))
    lambda_reflect = float(cfg["training"].get("lambda_reflect", 1.0))
    sigma_edge = float(cfg["training"].get("sigma_edge", 5.0))

    # ---- Model & Loss & Optimizer ----
    model = UNet(n_classes=104).to(device)
    loss_fn = TotalLoss(alpha=alpha, beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    # ---- Logging ----
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["logging"]["log_dir"]) if cfg["logging"]["use_tensorboard"] else None

    best_val_loss = float("inf")

    # ---- Training Loop ----
    for epoch in range(cfg["training"]["epochs"]):
<<<<<<< Updated upstream
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
=======
        train_loss = train_one_epoch(
            model,
            loader_train,
            loss_fn,
            optimizer,
            device,
            epoch,
            writer,
            lambda_edge=lambda_edge,
            lambda_reflect=lambda_reflect,
            sigma=sigma_edge,
            num_classes=104,
        )

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in loader_val:
                images_cpu = batch["image"]
                masks_cpu = batch["mask"]

                weight_map = compute_weight_map_batch(
                    images_cpu, masks_cpu,
                    lambda_edge=lambda_edge,
                    lambda_reflect=lambda_reflect,
                    sigma=sigma_edge,
                    num_classes=104,
                )

                images = images_cpu.to(device)
                masks = masks_cpu.to(device)
                weight_map = weight_map.to(device)

                preds = model(images)
                loss, _ = loss_fn(preds, masks, images, weight_map)
                val_loss += loss.item()
        val_loss /= len(loader_val)
        print(f"Epoch {epoch} VAL loss: {val_loss:.4f}")

        # ---- Checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Best model saved at epoch {epoch} → {ckpt_path}")

        if (epoch + 1) % cfg["training"]["save_interval"] == 0:
            ckpt_path = os.path.join(
                cfg["logging"]["checkpoint_dir"],
                f"unet_epoch{epoch+1}.pt",
>>>>>>> Stashed changes
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()