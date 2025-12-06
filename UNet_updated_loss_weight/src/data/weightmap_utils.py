import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_gradient_magnitude


def compute_class_balance_weight(mask, num_classes=104, eps=1e-6):
    """
    mask: (H, W) with class IDs
    返回 w_c(x) —— class-level balancing weight
    """
    flat = mask.reshape(-1)
    counts = np.bincount(flat, minlength=num_classes)

    freq = counts / (flat.size + eps)
    inv_freq = 1.0 / (freq + eps)

    # normalize to [0.5, 1.5] 区间，避免极端权重
    inv_freq = inv_freq / inv_freq.mean()

    wc = inv_freq[flat].reshape(mask.shape)
    return wc


def compute_edge_distance_weight(mask, sigma=5.0):
    """
    mask: (H, W)
    返回 e^{-d_edge^2 / (2σ^2)} —— 用于强调边界附近的像素
    """
    # 找到边界像素
    edge = np.zeros_like(mask, dtype=np.uint8)
    edge[1:] |= (mask[1:] != mask[:-1])
    edge[:-1] |= (mask[:-1] != mask[1:])
    edge[:, 1:] |= (mask[:, 1:] != mask[:, :-1])
    edge[:, :-1] |= (mask[:, :-1] != mask[:, 1:])

    # 距离变换
    d_edge = distance_transform_edt(edge == 0)

    w_edge = np.exp(-(d_edge ** 2) / (2 * sigma ** 2))
    return w_edge


def compute_reflect_suppression(image_np, threshold=220):
    """
    image_np: RGB numpy (H, W, 3)
    返回 R(x)，亮度梯度
    """

    gray = image_np.mean(axis=2).astype(np.float32)
    R = gaussian_gradient_magnitude(gray, sigma=1.0)

    # normalize to [0,1]
    R = R / (R.max() + 1e-6)
    return R


def build_weight_map(image_np, mask_np,
                     lambda_edge=1.0,
                     lambda_reflect=1.0,
                     sigma=5.0,
                     num_classes=104):
    """
    综合：
    w(x) = w_c(x) + λ_edge * w_edge(x) - λ_reflect * R(x)
    """
    wc = compute_class_balance_weight(mask_np, num_classes=num_classes)
    w_edge = compute_edge_distance_weight(mask_np, sigma=sigma)
    R = compute_reflect_suppression(image_np)

    w = wc + lambda_edge * w_edge - lambda_reflect * R

    w = np.clip(w, 0.05, None)
    return w