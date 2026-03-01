import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridMatchingLoss(nn.Module):
    def __init__(self, tau=0.5, alpha_mse=0.5, alpha_clip=0.5):
        super().__init__()
        self.tau = tau
        self.alpha_mse = alpha_mse
        self.alpha_clip = alpha_clip

    def forward(self, e, q):
        loss_mse = F.mse_loss(e, q)
        e_norm = F.normalize(e, dim=1)
        q_norm = F.normalize(q, dim=1)
        logits = torch.matmul(e_norm, q_norm.T) / self.tau
        labels = torch.arange(e.size(0), device=e.device)
        loss_clip = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return self.alpha_mse * loss_mse + self.alpha_clip * loss_clip
