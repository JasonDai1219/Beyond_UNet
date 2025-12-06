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
        logp = F.log_softmax(inputs, dim=1)              
        nll = -logp.gather(1, targets.unsqueeze(1))      

        if weight_map is not None:
            nll = nll * weight_map.unsqueeze(1)

        return nll.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        num_classes = probs.shape[1]

        one_hot = F.one_hot(targets, num_classes=num_classes)  
        one_hot = one_hot.permute(0, 3, 1, 2).float()          

        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class ConsistencyLoss(nn.Module):
    """
    类内亮度一致性损失
    """
    def __init__(self):
        super().__init__()

    def forward(self, images, preds):
        with torch.no_grad():
            r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
            brightness = 0.299 * r + 0.587 * g + 0.114 * b

        probs = F.softmax(preds, dim=1)
        num = (brightness * probs).sum(dim=(2, 3), keepdim=True)
        den = probs.sum(dim=(2, 3), keepdim=True) + 1e-6
        mean_brightness = num / den

        var = ((brightness - mean_brightness) ** 2 * probs).sum(dim=(2, 3))
        var = var / (probs.sum(dim=(2, 3)) + 1e-6)

        return var.mean()


# ============================================================
# 🔥 新增 edge loss + reflective loss（简单可扩展）
# ============================================================
def compute_edge_loss(preds, targets, sigma=1.0):
    """
    边缘一致性损失（占位版本，与你 pipeline 兼容）
    如果你之后要加 Canny / Sobel，可在这里扩展。
    """
    preds_argmax = torch.argmax(preds, dim=1)
    edge_pred = preds_argmax.float()
    edge_gt = targets.float()

    return F.l1_loss(edge_pred, edge_gt)


def compute_reflective_loss(preds, targets, images):
    """
    反光区域辅助损失（占位版本）
    """
    brightness = images.mean(dim=1, keepdim=True)   # (B,1,H,W)
    refl_mask = (brightness > 0.85).float()

    preds_argmax = torch.argmax(preds, dim=1).unsqueeze(1).float()

    return F.l1_loss(preds_argmax * refl_mask, targets.unsqueeze(1).float() * refl_mask)


# ============================================================
# 🧩 完整 TotalLoss —— train.py 完全兼容
# ============================================================
class TotalLoss(nn.Module):
    """
    L_total = CE + alpha*Dice + beta*Consistency
              + lambda_edge * L_edge
              + lambda_reflect * L_refl
    """

    def __init__(
        self,
        alpha=0.5,
        beta=0.2,
        lambda_edge=0.0,
        lambda_reflect=0.0,
        sigma_edge=1.0
    ):
        super().__init__()
        self.ce = WeightedCrossEntropy()
        self.dice = DiceLoss()
        self.cons = ConsistencyLoss()

        self.alpha = alpha
        self.beta = beta
        self.lambda_edge = lambda_edge
        self.lambda_reflect = lambda_reflect
        self.sigma_edge = sigma_edge

    def forward(self, preds, targets, images, weight_map=None):
        L_ce = self.ce(preds, targets, weight_map)
        L_dice = self.dice(preds, targets)
        L_cons = self.cons(images, preds)

        # optional losses
        L_edge = compute_edge_loss(preds, targets, sigma=self.sigma_edge) \
                 if self.lambda_edge > 0 else torch.tensor(0.0, device=preds.device)

        L_refl = compute_reflective_loss(preds, targets, images) \
                 if self.lambda_reflect > 0 else torch.tensor(0.0, device=preds.device)

        total = (
            L_ce +
            self.alpha * L_dice +
            self.beta * L_cons +
            self.lambda_edge * L_edge +
            self.lambda_reflect * L_refl
        )

        parts = {
            "CE": L_ce.item(),
            "Dice": L_dice.item(),
            "Cons": L_cons.item(),
            "Edge": L_edge.item(),
            "Reflect": L_refl.item()
        }

        return total, parts