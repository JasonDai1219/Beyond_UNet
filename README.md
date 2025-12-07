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
```
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
```
---

# 🚀 Project Overview

This repository contains our experiments on **semantic segmentation for food images** based on the **FoodSeg103** dataset.  
The project explores multiple segmentation models and compares their performance under different training objectives.

Our contributions include:

### 🧩 1. Baseline CNN Segmentation
A lightweight convolutional network used as a sanity check.  
This model provides a minimal baseline for comparison with more advanced architectures.

### 🧩 2. Standard U-Net
A conventional U-Net trained with **Cross-Entropy + Dice loss**, serving as a strong baseline.

### 🧩 3. Enhanced U-Net (My Work)
This repository includes an extended loss function that adds:

- **Edge-Aware Loss** (λ_edge)  
- **Reflection Consistency Loss** (λ_reflect)  
- **Dice Loss Weighting** (α)  
- **Feature Consistency Loss** (β)

We explored these components through:

- grid search over hyperparameters  
- fast prototype runs using a notebook  
- full-scale training with TensorBoard logging

### 🧩 4. Partner’s Method (Placeholder)
This section is reserved for my teammate's alternative segmentation approach  
(e.g., DeepLabV3+, Transformer-based segmentation, or post-processing optimization).  
Details will be added by the teammate.

---

The goal of the project is to evaluate how each model behaves under different training setups and to investigate whether additional inductive biases (edge priors, reflection priors, consistency constraints) improve segmentation performance.

---
# 🛠 Installation

This project requires Python 3.9+ and PyTorch with CUDA support.

### 1. Clone the repository
```bash
git clone https://github.com/<your-repo>/UNet_updated_loss_weight.git
```
### 2. Set up an environment
```
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
---
# ▶️ How to Run (Local Training via CLI)

In UNet_updated_loss_weight, all training is controlled by YAML configuration files located in UNet_updated_loss_weight/configs/
```bash
python3 -m UNet_updated_loss_weight.src.train --config <path-to-config>
```
The baseline CNN:
```bash
python3 -m UNet_updated_loss_weight.src.train \
    --config UNet_updated_loss_weight/configs/config_cnn.yaml
```
The vanilla UNet:
```bash
python3 -m UNet_updated_loss_weight.src.train \
    --config UNet_updated_loss_weight/configs/config_unet.yaml
```
The UNet + Updated Loss & Weight Map:
```bash
python3 -m UNet_updated_loss_weight.src.train \
    --config UNet_updated_loss_weight/configs/config_unet_advanced.yaml
```
---
# 🟦 Running in Google Colab

If you prefer running the project without local GPU setup, we provide a ready-to-use Google Colab notebook:

👉 **Colab Notebook:**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1_6uT4zyGa0F-40jQTX_1mwDqQ4tIhLTc?authuser=1
)

This notebook includes:

- automatic project folder download  
- environment setup  
- dataset preparation  
- training (CNN / UNet / Improved UNet)  
- visualization of predictions  
- TensorBoard integration  

---

## ▶️ Steps to Run in Colab

### **1. Open the notebook**
Just click the link above.

### **2. Enable GPU**
In Colab: Runtime → Change runtime type → GPU

### **3. Run all cells**
