import torch
import torch.nn as nn
import torch.nn.functional as F
from piqa import SSIM

class CompositeLossScheduler(nn.Module):
    """
    Implementation of the Dynamic Loss Scheduler described in Section 2.2.4.
    Formula: weights = (1 - alpha) * start_weights + alpha * end_weights
    """
    def __init__(self, start_weights, end_weights, total_epochs):
        super().__init__()
        # weights order: [MAPE, SSIM, Segmentation_BCE, Salt_Regularization]
        self.start_weights = torch.tensor(start_weights)
        self.end_weights = torch.tensor(end_weights)
        self.total_epochs = total_epochs
        
        # Initialize SSIM metric
        self.ssim_loss = SSIM(n_channels=1).cuda()

    def get_current_weights(self, current_epoch):
        alpha = min(1.0, current_epoch / self.total_epochs)
        current_weights = (1 - alpha) * self.start_weights + alpha * self.end_weights
        return current_weights

    def forward(self, pred_vp, target_vp, pred_mask, target_mask, epoch):
        w = self.get_current_weights(epoch).to(pred_vp.device)
        
        # 1. MAPE (Mean Absolute Percentage Error)
        loss_mape = torch.mean(torch.abs((target_vp - pred_vp) / (target_vp + 1e-6)))
        
        # 2. SSIM (Structural Similarity)
        # SSIM is usually maximization, so loss is 1 - SSIM
        loss_ssim = 1.0 - self.ssim_loss(pred_vp, target_vp)
        
        # 3. Segmentation Loss (Binary Cross Entropy)
        loss_bce = F.binary_cross_entropy_with_logits(pred_mask, target_mask)
        
        # 4. Salt-Weighted Regularization (Domain Prior)
        # Penalize velocity errors more heavily inside salt regions
        salt_weight_map = 1.0 + (target_mask * 2.0) # 3x weight inside salt
        loss_salt_reg = torch.mean(torch.abs(pred_vp - target_vp) * salt_weight_map)

        # Composite Sum
        total_loss = (w[0] * loss_mape) +                      (w[1] * loss_ssim) +                      (w[2] * loss_bce) +                      (w[3] * loss_salt_reg)
                     
        return total_loss, {
            "mape": loss_mape.item(),
            "ssim": loss_ssim.item(),
            "bce": loss_bce.item(),
            "salt_reg": loss_salt_reg.item()
        }

class PathAwareLoss(nn.Module):
    """
    Paper Section 2.2.4: Enforces local accuracy along vertical borehole paths.
    """
    def __init__(self, weight=0.3, num_paths=3):
        super().__init__()
        self.weight = weight
        self.num_paths = num_paths

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        
        # Simulate vertical boreholes
        path_mask = torch.zeros_like(pred)
        # Select random lateral positions for wells
        cols = torch.randint(0, W, (self.num_paths,))
        for c in cols:
            path_mask[:, :, :, c] = 1.0
            
        # Calculate loss only on paths
        path_error = (pred - target) ** 2
        masked_error = (path_error * path_mask).sum() / (path_mask.sum() + 1e-6)
        
        return self.weight * masked_error