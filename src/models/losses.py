# src/models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropy(nn.Module):
    def __init__(self, weight_map=None):
        super().__init__()
        self.weight_map = weight_map

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, reduction='mean')

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0,3,1,2)
        intersection = (inputs * targets_onehot).sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3)) + self.smooth)
        return 1 - dice.mean()

class ConsistencyLoss(nn.Module):
    """Enforces intra-class brightness consistency"""
    def forward(self, images, preds):
        with torch.no_grad():
            brightness = images[:, 0:1, :, :] * 0.299 + images[:, 1:2, :, :] * 0.587 + images[:, 2:3, :, :] * 0.114
        probs = F.softmax(preds, dim=1)
        mean_brightness = (brightness * probs).sum(dim=(2,3), keepdim=True) / (probs.sum(dim=(2,3), keepdim=True) + 1e-6)
        variance = ((brightness - mean_brightness)**2 * probs).sum(dim=(2,3)) / (probs.sum(dim=(2,3)) + 1e-6)
        return variance.mean()

class TotalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.2):
        super().__init__()
        self.ce = WeightedCrossEntropy()
        self.dice = DiceLoss()
        self.cons = ConsistencyLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets, images):
        L_ce = self.ce(preds, targets)
        L_dice = self.dice(preds, targets)
        L_cons = self.cons(images, preds)
        total = L_ce + self.alpha * L_dice + self.beta * L_cons
        return total, {"CE": L_ce.item(), "Dice": L_dice.item(), "Cons": L_cons.item()}