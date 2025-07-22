import torch
import torch.nn.functional as F
import numpy as np
import cv2

def fast_postprocess(pred, thresh=0.5, kernel_thresh=0.5, min_area=300):
    try:
        batch_size = pred.shape[0]
        num_kernels = pred.shape[1]
        results = []
        for b in range(batch_size):
            text_score = pred[b, 0].cpu().numpy()
            kernels = pred[b, 1:].cpu().numpy()
            text_mask = (text_score > thresh).astype(np.uint8)
            kernel_masks = [(kernels[i] > kernel_thresh).astype(np.uint8) for i in range(num_kernels-1)]
            # Use the smallest kernel for connected components
            cc, labels = cv2.connectedComponents(kernel_masks[-1], connectivity=4)
            polygons = []
            for i in range(1, cc):
                mask = (labels == i).astype(np.uint8)
                # Dilate to text region
                mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
                mask = mask * text_mask
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < min_area:
                        continue
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    polygons.append(approx.reshape(-1,2))
            results.append(polygons)
        return results
    except Exception as e:
        print(f"[ERROR] fast_postprocess error: {e}")
        raise 