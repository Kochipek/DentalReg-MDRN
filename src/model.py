"""
Multi-Scale Dilated Residual Network (MDRN) Architecture
Author: İpek Koçhisarlı (ipekkochisarli@gmail.com)
Project: DentalReg-MDRN
"""

import torch
import torch.nn as nn


class MultiScaleDilatedResNet(nn.Module):
    """
    Multi-Scale Dilated Residual Convolutional Neural Network (MDRN)
    
    Designed specifically for sub-pixel residual intensity refinement in periapical
    dental radiographs. Operates without pooling layers to retain full spatial resolution,
    combining parallel dilated branches (d=1, 2, 4) with a bounded residual mapping head.
    
    Args:
        in_channels (int): Number of input image channels (default: 1 for grayscale).
        num_features (int): Base feature depth across convolutional layers (default: 32).
        alpha_init (float): Initial value for the learnable residual scaling parameter (default: 0.25).
    """
    def __init__(self, in_channels: int = 1, num_features: int = 32, alpha_init: float = 0.25):
        super().__init__()
        
        # Initial feature extraction layer
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Parallel dilated branches capturing multi-scale receptive fields
        # Branch 1 (d=1, RF=3x3): Micro-fine enamel cracks & canal edges
        self.dil1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Branch 2 (d=2, RF=5x5): Dentin-enamel junctions & periodontal spaces
        self.dil2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Branch 3 (d=4, RF=9x9): Macro tooth contours & alveolar bone boundaries
        self.dil3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=4, dilation=4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Multi-scale feature fusion (concatenate 3 branches -> 1x1 projection)
        self.fuse = nn.Sequential(
            nn.Conv2d(num_features * 3, num_features, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Bounded residual prediction head with Tanh activation
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Learnable residual modulation parameter
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input registered radiograph tensor of shape (B, C, H, W) in [0, 1].
            
        Returns:
            out (torch.Tensor): Refined radiograph tensor clamped to [0, 1].
            residual (torch.Tensor): Raw residual correction map prior to scaling.
        """
        feat = self.conv_in(x)
        f1 = self.dil1(feat)
        f2 = self.dil2(feat)
        f3 = self.dil3(feat)
        
        fused = self.fuse(torch.cat([f1, f2, f3], dim=1))
        residual = self.conv_out(fused)
        
        # Residual skip connection with bounded range
        out = torch.clamp(x + self.alpha * residual, 0.0, 1.0)
        return out, residual
