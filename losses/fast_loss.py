import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Dice Loss ----
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

# ---- OHEM Mask ----
def ohem_mask(pred, gt, training_mask, ratio=3):
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
                neg_pred = pred[i][neg_idx]
                if neg_pred.numel() > 0:
                    sorted_idx = torch.argsort(neg_pred, descending=True)
                    hard_neg_idx = neg_idx.nonzero(as_tuple=True)
                    mask[i][neg_idx] = 0.0
                    mask[i][neg_idx][sorted_idx[:n_neg]] = 1.0
        return mask

# ---- Embedding Loss v1 (for instance separation) ----
class EmbLoss_v1(nn.Module):
    def __init__(self, feature_dim=4, delta_v=0.5, delta_d=1.5, loss_weight=0.25):
        super().__init__()
        self.feature_dim = feature_dim
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.loss_weight = loss_weight
    def forward(self, emb, gt_instance, training_mask):
        try:
            batch_size = emb.size(0)
            loss_var, loss_dist, loss_reg = 0., 0., 0.
            for b in range(batch_size):
                emb_b = emb[b].permute(1,2,0).contiguous().view(-1, self.feature_dim)  # [H*W, C]
                gt_b = gt_instance[b].view(-1)
                mask_b = training_mask[b].view(-1)
                unique_labels = gt_b[mask_b > 0].unique()
                if len(unique_labels) <= 1:
                    continue
                mean_vecs = []
                for label in unique_labels:
                    indices = (gt_b == label) & (mask_b > 0)
                    if indices.sum() == 0:
                        mean_vecs.append(torch.zeros(self.feature_dim, device=emb.device))
                        continue
                    mean_vec = emb_b[indices].mean(0)
                    mean_vecs.append(mean_vec)
                    # Variance loss
                    dist = (emb_b[indices] - mean_vec).norm(p=2, dim=1)
                    loss_var += torch.clamp(dist - self.delta_v, min=0).pow(2).mean() / len(unique_labels)
                mean_vecs = torch.stack(mean_vecs)
                # Distance loss
                for i in range(len(mean_vecs)):
                    for j in range(i+1, len(mean_vecs)):
                        dist = (mean_vecs[i] - mean_vecs[j]).norm(p=2)
                        loss_dist += torch.clamp(2 * self.delta_d - dist, min=0).pow(2) / (len(mean_vecs)*(len(mean_vecs)-1))
                # Regularization loss
                loss_reg += mean_vecs.norm(p=2, dim=1).mean() / batch_size
            loss = loss_var + loss_dist + 0.001 * loss_reg
            return self.loss_weight * loss
        except Exception as e:
            print(f"[ERROR] EmbLoss_v1 computation error: {e}")
            raise

# ---- FAST Loss (Dice + Embedding) ----
class FASTLoss(nn.Module):
    def __init__(self, num_kernels=6, alpha=0.5, dilation_size=9, feature_dim=4):
        super().__init__()
        self.num_kernels = num_kernels
        self.alpha = alpha  # weight for text region loss (paper uses 0.5)
        self.dilation_size = dilation_size
        self.emb_loss = EmbLoss_v1(feature_dim=feature_dim)
    def forward(self, pred, gt_text, gt_kernels, training_mask, gt_instance=None):
        try:
            pred_text = pred[:, 0, :, :]
            pred_kernels = pred[:, 1:self.num_kernels, :, :]
            pred_emb = pred[:, self.num_kernels:, :, :] if pred.size(1) > self.num_kernels else None
            gt_text = gt_text.squeeze(1)
            training_mask = training_mask.squeeze(1)
            # Dilation with F.max_pool2d (as in paper)
            pred_prob = torch.sigmoid(pred_text)
            pred_dilated = F.max_pool2d(pred_prob.unsqueeze(1), self.dilation_size, 1, self.dilation_size//2).squeeze(1)
            # OHEM mask for text region (on dilated region)
            with torch.no_grad():
                ohem = ohem_mask(pred_dilated, gt_text, training_mask, ratio=3)
            # Text region loss (Dice with OHEM mask, on dilated region)
            loss_text = dice_loss(pred_dilated, gt_text, ohem)
            # Kernel loss (Dice, as in paper)
            loss_kernels = 0
            for i in range(self.num_kernels-1):
                loss_kernels += dice_loss(torch.sigmoid(pred_kernels[:, i, :, :]), gt_kernels[:, i, :, :], training_mask)
            loss_kernels /= (self.num_kernels-1)
            # Embedding loss (if embedding prediction and gt_instance are provided)
            loss_emb = 0
            if pred_emb is not None and gt_instance is not None:
                loss_emb = self.emb_loss(pred_emb, gt_instance, training_mask)
            # Total loss (paper: L = Lker + alpha * Ltex + 0.25*Lemb)
            loss = loss_kernels + self.alpha * loss_text + loss_emb
            return loss, {'loss_text': loss_text, 'loss_kernels': loss_kernels, 'loss_emb': loss_emb}
        except Exception as e:
            print(f"[ERROR] FASTLoss computation error: {e}")
            raise 