import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion


def binarize(prob, threshold=0.5):
    return (prob >= threshold).astype(np.uint8)


def dice_score(pred, target, eps=1e-7):
    pred = pred.astype(np.uint8).ravel()
    target = target.astype(np.uint8).ravel()
    inter = np.sum(pred * target)
    return (2 * inter + eps) / (np.sum(pred) + np.sum(target) + eps)


def iou_score(pred, target, eps=1e-7):
    pred = pred.astype(np.uint8).ravel()
    target = target.astype(np.uint8).ravel()
    inter = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - inter
    return (inter + eps) / (union + eps)


def precision_score(pred, target, eps=1e-7):
    pred = pred.astype(np.uint8).ravel()
    target = target.astype(np.uint8).ravel()
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, target, eps=1e-7):
    pred = pred.astype(np.uint8).ravel()
    target = target.astype(np.uint8).ravel()
    tp = np.sum(pred * target)
    fn = np.sum((1 - pred) * target)
    return (tp + eps) / (tp + fn + eps)


def surface_points(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.empty((0, 2))
    eroded = binary_erosion(mask)
    surface = mask ^ eroded
    return np.argwhere(surface)


def hd95(pred, target):
    p = surface_points(pred)
    t = surface_points(target)
    if len(p) == 0 or len(t) == 0:
        return np.nan
    distances_pt = np.sqrt(((p[:, None, :] - t[None, :, :]) ** 2).sum(axis=2))
    min_p_to_t = distances_pt.min(axis=1)
    min_t_to_p = distances_pt.min(axis=0)
    return np.percentile(np.concatenate([min_p_to_t, min_t_to_p]), 95)


def asd(pred, target):
    p = surface_points(pred)
    t = surface_points(target)
    if len(p) == 0 or len(t) == 0:
        return np.nan
    distances_pt = np.sqrt(((p[:, None, :] - t[None, :, :]) ** 2).sum(axis=2))
    min_p_to_t = distances_pt.min(axis=1)
    min_t_to_p = distances_pt.min(axis=0)
    return np.mean(np.concatenate([min_p_to_t, min_t_to_p]))


def compute_metrics(pred, target):
    return {
        'dice': dice_score(pred, target),
        'iou': iou_score(pred, target),
        'precision': precision_score(pred, target),
        'recall': recall_score(pred, target),
        'hd95': hd95(pred, target),
        'asd': asd(pred, target),
    }
