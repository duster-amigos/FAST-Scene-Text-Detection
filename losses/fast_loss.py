import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(input, target, mask, eps=1e-6):
    try:
        input = input.contiguous().view(input.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        mask = mask.contiguous().view(mask.size(0), -1)
        input = input * mask
        target = target * mask
        intersection = (input * target).sum(1)
        union = (input ** 2).sum(1) + (target ** 2).sum(1) + eps
        loss = 1 - (2 * intersection / union)
        return loss.mean()
    except Exception as e:
        print(f"[ERROR] dice_loss computation error: {e}")
        raise

def ohem_mask(pred, gt, training_mask, ratio=3):
    # pred: (N, H, W) after sigmoid
    # gt: (N, H, W)
    # training_mask: (N, H, W)
    with torch.no_grad():
        pos = (gt > 0.5) & (training_mask > 0.5)
        neg = (gt <= 0.5) & (training_mask > 0.5)
        pos_num = pos.sum(dim=(1,2))
        neg_num = neg.sum(dim=(1,2))
        mask = torch.zeros_like(gt, dtype=torch.float32)
        for i in range(gt.size(0)):
            pos_idx = pos[i]
            neg_idx = neg[i]
            mask[i][pos_idx] = 1.0
            n_pos = pos_num[i].item()
            n_neg = min(int(ratio * n_pos), int(neg_num[i].item()))
            if n_neg > 0:
                # For negatives, select those with highest prediction (hardest negatives)
                neg_pred = pred[i][neg_idx]
                if neg_pred.numel() > 0:
                    sorted_idx = torch.argsort(neg_pred, descending=True)
                    hard_neg_idx = neg_idx.nonzero(as_tuple=True)
                    selected = tuple(idx[sorted_idx[:n_neg]] for idx in hard_neg_idx)
                    mask[i][neg_idx] = 0.0  # clear all first
                    mask[i][neg_idx][sorted_idx[:n_neg]] = 1.0
        return mask

class FASTLoss(nn.Module):
    def __init__(self, num_kernels=6, alpha=0.5):
        super().__init__()
        self.num_kernels = num_kernels
        self.alpha = alpha  # weight for text region loss (paper uses 0.5)
    def forward(self, pred, gt_text, gt_kernels, training_mask):
        try:
            pred_text = pred[:, 0, :, :]
            pred_kernels = pred[:, 1:, :, :]
            gt_text = gt_text.squeeze(1)
            training_mask = training_mask.squeeze(1)
            # OHEM mask for text region
            with torch.no_grad():
                pred_prob = torch.sigmoid(pred_text)
                ohem = ohem_mask(pred_prob, gt_text, training_mask, ratio=3)
            # Text region loss (Dice with OHEM mask)
            loss_text = dice_loss(pred_prob, gt_text, ohem)
            # Kernel loss (Dice, as in paper)
            loss_kernels = 0
            for i in range(self.num_kernels-1):
                loss_kernels += dice_loss(torch.sigmoid(pred_kernels[:, i, :, :]), gt_kernels[:, i, :, :], training_mask)
            loss_kernels /= (self.num_kernels-1)
            # Total loss (paper: L = Lker + alpha * Ltex)
            loss = loss_kernels + self.alpha * loss_text
            return loss, {'loss_text': loss_text, 'loss_kernels': loss_kernels}
        except Exception as e:
            print(f"[ERROR] FASTLoss computation error: {e}")
            raise 