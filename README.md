# 🍱 FoodSeg103 Segmentation Project  
UNet Enhancements + Team Baselines

This repository contains our team’s experimental segmentation models on the **FoodSeg103** dataset.  
Each team member contributed a different modeling idea, and this repo serves as a unified benchmark + comparison hub.

The project includes:

- CNN baseline (simple segmentation model)
- Vanilla UNet
- Enhanced UNet with multi-loss supervision (this method)
- **[Placeholder]** Teammate’s Method A  
- **[Placeholder]** Teammate’s Method B  
- Automated hyperparameter search
- Colab + local training options

---

# 📂 Project Structure

UNet_updated_loss_weight/
│
├── src/
│   ├── train.py
│   ├── models/
│   └── data/
│
├── configs/
│   ├── config_cnn.yaml
│   ├── config_unet.yaml
│   └── config_unet_advanced.yaml
│
├── train.ipynb
└── README.md
