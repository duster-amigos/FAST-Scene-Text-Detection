import torch
import torch.nn.functional as F
import numpy as np
import cv2

def fast_postprocess(pred, thresh=0.5, kernel_thresh=0.5, min_area=300, dilation_size=9):
    """Post-processes model output to extract polygons for detected text regions.
    Uses GPU-parallel dilation and connected components to grow kernels and find text instances.
    Args:
        pred: Model output, shape [B, num_kernels, H, W]
        thresh: Threshold for text region
        kernel_thresh: Threshold for kernel regions
        min_area: Minimum area for valid polygons
        dilation_size: Dilation kernel size for mask growing
    Returns:
        List of list of polygons (per image in batch)
    """
    try:
        batch_size = pred.shape[0]
        num_kernels = pred.shape[1]
        results = []
        for b in range(batch_size):
            # Get text region and kernel region predictions for this image
            text_score = pred[b, 0].cpu().numpy()
            kernels = pred[b, 1:].cpu().numpy()
            # Threshold to get binary masks
            text_mask = (text_score > thresh).astype(np.uint8)
            kernel_masks = [(kernels[i] > kernel_thresh).astype(np.uint8) for i in range(num_kernels-1)]
            # Use the smallest kernel for connected components (instance separation)
            cc, labels = cv2.connectedComponents(kernel_masks[-1], connectivity=4)
            polygons = []
            for i in range(1, cc):
                # Isolate each instance mask
                mask = (labels == i).astype(np.uint8)
                # Dilation with F.max_pool2d (GPU) to grow kernel to text region
                mask_torch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                mask_torch = F.max_pool2d(mask_torch, dilation_size, 1, dilation_size//2)
                mask = (mask_torch.squeeze().cpu().numpy() > 0).astype(np.uint8)
                # Mask with text region to avoid overflow
                mask = mask * text_mask
                # Find contours for each instance
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    # Filter out small regions
                    if cv2.contourArea(cnt) < min_area:
                        continue
                    # Polygon approximation for output
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    polygons.append(approx.reshape(-1,2))
            results.append(polygons)
        return results
    except Exception as e:
        print(f"[ERROR] fast_postprocess error: {e}")
        raise 