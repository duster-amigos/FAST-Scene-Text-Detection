import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(input, target, mask, eps=1e-6):
    try:
        # input: (N, H, W), target: (N, H, W), mask: (N, H, W)
        input = input.contiguous().view(input.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        mask = mask.contiguous().view(mask.size(0), -1)
        input = input * mask
        target = target * mask
        intersection = (input * target).sum(1)
        union = input.sum(1) + target.sum(1) + eps
        loss = 1 - (2 * intersection / union)
        return loss.mean()
    except Exception as e:
        print(f"[ERROR] dice_loss computation error: {e}")
        raise

class FASTLoss(nn.Module):
    def __init__(self, num_kernels=6, alpha=1.0, beta=0.7):
        super().__init__()
        self.num_kernels = num_kernels
        self.alpha = alpha  # weight for text region loss
        self.beta = beta   # weight for kernel loss
    def forward(self, pred, gt_text, gt_kernels, training_mask):
        try:
            # pred: (N, num_kernels, H, W)
            # gt_text: (N, 1, H, W)
            # gt_kernels: (N, num_kernels-1, H, W)
            # training_mask: (N, 1, H, W)
            pred_text = pred[:, 0, :, :]
            pred_kernels = pred[:, 1:, :, :]
            gt_text = gt_text.squeeze(1)
            training_mask = training_mask.squeeze(1)
            # Text region loss (BCE)
            loss_text = F.binary_cross_entropy_with_logits(pred_text, gt_text, reduction='none')
            loss_text = (loss_text * training_mask).sum() / (training_mask.sum() + 1e-6)
            # Kernel loss (Dice)
            loss_kernels = 0
            for i in range(self.num_kernels-1):
                loss_kernels += dice_loss(torch.sigmoid(pred_kernels[:, i, :, :]), gt_kernels[:, i, :, :], training_mask)
            loss_kernels /= (self.num_kernels-1)
            # Total loss
            loss = self.alpha * loss_text + self.beta * loss_kernels
            return loss, {'loss_text': loss_text, 'loss_kernels': loss_kernels}
        except Exception as e:
            print(f"[ERROR] FASTLoss computation error: {e}")
            raise 