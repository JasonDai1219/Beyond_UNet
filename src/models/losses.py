# src/models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropy(nn.Module):
    """
    像素级加权 Cross-Entropy：
    L_ce = mean_x w(x) * CE(p(x), y(x))
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weight_map=None):
        """
        inputs: (B, C, H, W) logits
        targets: (B, H, W) int64
        weight_map: (B, H, W) or None
        """
        logp = F.log_softmax(inputs, dim=1)              # (B, C, H, W)
        nll = -logp.gather(1, targets.unsqueeze(1))      # (B, 1, H, W)

        if weight_map is not None:
            nll = nll * weight_map.unsqueeze(1)          # 广播到 (B,1,H,W)

        return nll.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: (B, C, H, W) logits
        targets: (B, H, W)
        """
        probs = F.softmax(inputs, dim=1)   # (B, C, H, W)
        num_classes = probs.shape[1]

        one_hot = F.one_hot(targets, num_classes=num_classes)  # (B,H,W,C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()          # (B,C,H,W)

        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class ConsistencyLoss(nn.Module):
    """
    类内亮度一致性：惩罚同一类别内部亮度方差过大
    """
    def __init__(self):
        super().__init__()

    def forward(self, images, preds):
        """
        images: (B, 3, H, W), 已经 normalize 也没关系（线性变换不改变相对方差结构）
        preds: (B, C, H, W) logits
        """
        with torch.no_grad():
            # 简单 RGB → 灰度
            r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:2+1]
            brightness = 0.299 * r + 0.587 * g + 0.114 * b  # (B,1,H,W)

        probs = F.softmax(preds, dim=1)   # (B, C, H, W)

        # 每个类的 brightness 均值 & 方差（soft）
        num = (brightness * probs).sum(dim=(2, 3), keepdim=True)
        den = probs.sum(dim=(2, 3), keepdim=True) + 1e-6
        mean_brightness = num / den                       # (B,C,1,1)

        var = ((brightness - mean_brightness) ** 2 * probs).sum(dim=(2, 3))
        var = var / (probs.sum(dim=(2, 3)) + 1e-6)        # (B,C)

        return var.mean()


class TotalLoss(nn.Module):
    """
    L_total = sum_x w(x)*CE + alpha * Dice + beta * Consistency
    """
    def __init__(self, alpha=0.5, beta=0.2):
        super().__init__()
        self.ce = WeightedCrossEntropy()
        self.dice = DiceLoss()
        self.cons = ConsistencyLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets, images, weight_map=None):
        L_ce = self.ce(preds, targets, weight_map)
        L_dice = self.dice(preds, targets)
        L_cons = self.cons(images, preds)

        total = L_ce + self.alpha * L_dice + self.beta * L_cons
        parts = {
            "CE": L_ce.item(),
            "Dice": L_dice.item(),
            "Cons": L_cons.item()
        }
        return total, parts