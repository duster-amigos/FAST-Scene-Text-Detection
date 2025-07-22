import numpy as np

def compute_f1(pred, gt, eps=1e-6):
    try:
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, np.logical_not(gt)).sum()
        fn = np.logical_and(np.logical_not(pred), gt).sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return precision, recall, f1
    except Exception as e:
        print(f"[ERROR] compute_f1 error: {e}")
        raise 