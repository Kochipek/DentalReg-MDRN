"""
Loss Functions for Dental Radiograph Residual Refinement
Author: İpek Koçhisarlı (ipekkochisarli@gmail.com)
Project: DentalReg-MDRN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Differentiable Smooth L1 proxy)
    
    L_charb(y_pred, y_true) = mean(sqrt((y_pred - y_true)^2 + eps^2))
    
    Robust against isolated radiographic outliers without over-penalizing high gradients.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        diff = pred - target
        if mask is not None:
            diff = diff * mask
            num_valid = torch.clamp(mask.sum(), min=1.0)
            return torch.sum(torch.sqrt(diff ** 2 + self.eps ** 2) * mask) / num_valid
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))


class SobelEdgeLoss(nn.Module):
    """
    Sobel Gradient Edge Preservation Loss
    
    Extracts directional gradients using fixed 3x3 Sobel kernels and penalizes
    discrepancies along mineralized enamel borders and root margins.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)
        
        diff_x = grad_pred_x - grad_target_x
        diff_y = grad_pred_y - grad_target_y
        
        if mask is not None:
            diff_x = diff_x * mask
            diff_y = diff_y * mask
            num_valid = torch.clamp(mask.sum(), min=1.0)
            return (torch.sum(diff_x ** 2) + torch.sum(diff_y ** 2)) / num_valid
        return torch.mean(diff_x ** 2 + diff_y ** 2)


class CompositeLoss(nn.Module):
    """
    Composite Loss combining Charbonnier photometric loss with Sobel edge regularization.
    
    L_total = L_charb + lambda_edge * L_sobel
    """
    def __init__(self, eps: float = 1e-3, lambda_edge: float = 0.15):
        super().__init__()
        self.charb = CharbonnierLoss(eps=eps)
        self.edge = SobelEdgeLoss()
        self.lambda_edge = lambda_edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        l_charb = self.charb(pred, target, mask)
        l_edge = self.edge(pred, target, mask)
        l_total = l_charb + self.lambda_edge * l_edge
        return l_total, l_charb, l_edge
