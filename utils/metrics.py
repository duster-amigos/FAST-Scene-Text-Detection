import numpy as np

def compute_f1(pred, gt, eps=1e-6):
    """Computes precision, recall, and F1 score for binary segmentation masks.
    Args:
        pred: predicted binary mask
        gt: ground truth binary mask
    Returns:
        precision, recall, f1 (all floats)
    """
    try:
        # True positives: predicted and ground truth are both 1
        tp = np.logical_and(pred, gt).sum()
        # False positives: predicted 1, ground truth 0
        fp = np.logical_and(pred, np.logical_not(gt)).sum()
        # False negatives: predicted 0, ground truth 1
        fn = np.logical_and(np.logical_not(pred), gt).sum()
        # Compute precision, recall, F1
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return precision, recall, f1
    except Exception as e:
        print(f"[ERROR] compute_f1 error: {e}")
        raise 