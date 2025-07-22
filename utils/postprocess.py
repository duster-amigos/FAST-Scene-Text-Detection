import torch
import torch.nn.functional as F
import numpy as np
import cv2

def fast_postprocess(pred, thresh=0.5, kernel_thresh=0.5, min_area=300, dilation_size=3):
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
                # Dilation with F.max_pool2d (GPU, as in paper)
                mask_torch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
                mask_torch = F.max_pool2d(mask_torch, dilation_size, 1, dilation_size//2)
                mask = (mask_torch.squeeze().cpu().numpy() > 0).astype(np.uint8)
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