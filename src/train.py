"""
Training Pipeline for Dental Radiograph Residual Refinement via MDRN
Author: İpek Koçhisarlı (ipekkochisarli@gmail.com)
Project: DentalReg-MDRN
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import MultiScaleDilatedResNet
from src.losses import CompositeLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Scale Dilated ResNet (MDRN) for Dental Radiography")
    parser.add_argument("--ref", type=str, default="", help="Path to reference ROI radiograph")
    parser.add_argument("--reg", type=str, default="", help="Path to registered test radiograph")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial learning rate")
    parser.add_argument("--lambda_edge", type=float, default=0.15, help="Weight of Sobel edge regularization")
    parser.add_argument("--output_dir", type=str, default="figures", help="Directory to save output artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def resolve_path(provided_path: str, default_rel_paths: list) -> str:
    """Resolve file path across potential project directory structures."""
    if provided_path and os.path.exists(provided_path):
        return provided_path
    
    # Try finding relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel_path in default_rel_paths:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            return full_path
        if os.path.exists(rel_path):
            return rel_path
    raise FileNotFoundError(f"Could not locate image file among candidates: {default_rel_paths}")


def main():
    args = parse_args()

    # 1. Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. Device selection (CUDA -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using compute device: {device}")

    # 3. Locate and load image data
    ref_path = resolve_path(args.ref, ["data/processed/ref_roi.bmp", "ref_roi.bmp"])
    reg_path = resolve_path(args.reg, ["data/processed/test_registered.bmp", "test_registered.bmp"])

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    reg_img = cv2.imread(reg_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or reg_img is None:
        raise ValueError(f"Failed to load images from {ref_path} and {reg_path}")

    # Normalize to tensors: shape (1, 1, H, W) in [0, 1]
    ref_t = torch.from_numpy(ref_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    reg_t = torch.from_numpy(reg_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # Binary mask excluding zero-fill rotation borders
    mask = (reg_t > 0.02).float().to(device)

    # Baseline metric computation
    baseline_mse = torch.sum(((ref_t - reg_t) * mask) ** 2) / torch.clamp(mask.sum(), min=1.0)
    baseline_psnr = 10.0 * np.log10(1.0 / baseline_mse.item())
    print(f"\n[Baseline] Registered Input PSNR (Tissue-masked): {baseline_psnr:.2f} dB (MSE: {baseline_mse.item():.6f})")

    # 4. Instantiate Model, Loss, Optimizer, and Scheduler
    model = MultiScaleDilatedResNet().to(device)
    criterion = CompositeLoss(eps=1e-3, lambda_edge=args.lambda_edge).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    print(f"\n[Training] Commencing MDRN optimization ({args.epochs} epochs)...")
    print(f"{'Epoch':>8} | {'Total Loss':>12} | {'Charbonnier':>12} | {'Sobel Edge':>12} | {'PSNR (dB)':>10}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred, residual = model(reg_t)
        loss, l_charb, l_edge = criterion(pred, ref_t, mask)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                cur_mse = torch.sum(((ref_t - pred) * mask) ** 2) / torch.clamp(mask.sum(), min=1.0)
                cur_psnr = 10.0 * np.log10(1.0 / cur_mse.item())
                print(f"{epoch:8d} | {loss.item():12.6f} | {l_charb.item():12.6f} | {l_edge.item():12.6f} | {cur_psnr:10.2f}")

    # 5. Final Evaluation and Artifact Export
    model.eval()
    with torch.no_grad():
        final_pred, final_res = model(reg_t)
        final_mse = torch.sum(((ref_t - final_pred) * mask) ** 2) / torch.clamp(mask.sum(), min=1.0)
        final_psnr = 10.0 * np.log10(1.0 / final_mse.item())

    net_gain = final_psnr - baseline_psnr
    print("-" * 64)
    print(f"[Result] Optimization Complete!")
    print(f"  Final Refined PSNR: {final_psnr:.2f} dB")
    print(f"  Net Quality Gain:   +{net_gain:.2f} dB over geometric registration")

    os.makedirs(args.output_dir, exist_ok=True)
    refined_np = (final_pred.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    diff_np = cv2.absdiff(ref_img, refined_np)

    refined_path = os.path.join(args.output_dir, "dl_refined.png")
    diff_path = os.path.join(args.output_dir, "diff_dl.png")
    cv2.imwrite(refined_path, refined_np)
    cv2.imwrite(diff_path, diff_np)
    print(f"[Artifacts] Saved refined radiograph to: {refined_path}")
    print(f"[Artifacts] Saved difference map to:       {diff_path}\n")


if __name__ == "__main__":
    main()
