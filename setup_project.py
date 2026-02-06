import os

# --- Structure Definition ---
dirs = [
    "configs",
    "data",
    "models",
    "utils",
    "scripts",
    "notebooks"
]

files = {}

# ==========================================
# 1. UTILS: LOSSES (From Paper Eqs)
# ==========================================
files["utils/losses.py"] = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from piqa import SSIM

class CompositeLossScheduler(nn.Module):
    \"\"\"
    Implementation of the Dynamic Loss Scheduler described in Section 2.2.4.
    Formula: weights = (1 - alpha) * start_weights + alpha * end_weights
    \"\"\"
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
        total_loss = (w[0] * loss_mape) + \
                     (w[1] * loss_ssim) + \
                     (w[2] * loss_bce) + \
                     (w[3] * loss_salt_reg)
                     
        return total_loss, {
            "mape": loss_mape.item(),
            "ssim": loss_ssim.item(),
            "bce": loss_bce.item(),
            "salt_reg": loss_salt_reg.item()
        }

class PathAwareLoss(nn.Module):
    \"\"\"
    Paper Section 2.2.4: Enforces local accuracy along vertical borehole paths.
    \"\"\"
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
"""

# ==========================================
# 2. MODELS: HybridGeo-UNet
# ==========================================
files["models/hybrid_unet.py"] = """
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    \"\"\"(convolution => [BN] => ReLU) * 2\"\"\"
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    \"\"\"Downscaling with maxpool then double conv\"\"\"
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    \"\"\"Upscaling then double conv\"\"\"
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class HybridGeoUNet(nn.Module):
    \"\"\"
    The GCHIM Model: Standard U-Net Encoder with Dual Decoders.
    1. Velocity Regression Head
    2. Geological Segmentation Head
    \"\"\"
    def __init__(self, n_channels=5, bilinear=True):
        super(HybridGeoUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # --- Dual Heads ---
        self.head_velocity = nn.Conv2d(64, 1, kernel_size=1)
        self.head_segmentation = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_dec = self.up4(x, x1)
        
        # Predictions
        vp_pred = self.head_velocity(x_dec)
        mask_pred = self.head_segmentation(x_dec) # Raw logits, apply sigmoid in loss
        
        return vp_pred, mask_pred
"""

# ==========================================
# 3. DATA: Seimic Dataset
# ==========================================
files["data/seismic_loader.py"] = """
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os

class SeismicVelocityDataset(Dataset):
    \"\"\"
    Paper Section 2.2.1: 5-channel source gathers input, Vp + Mask output.
    Target Shape: (300, 1259) -> Resized to (256, 256) for training.
    \"\"\"
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        # Logic to list files would go here. Assuming standardized naming.
        self.samples = sorted(glob.glob(os.path.join(root_dir, 'inputs', '*.npy')))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Load 5 source gathers
        # Note: Implement logic to find paired 5 sources for the specific sample
        # Placeholder for loading (5, 10001, 31) -> (5, 256, 256)
        
        # 2. Load Targets
        # vp = np.load(...)
        
        # 3. Create Mask (Algorithm 2.2.4: Salt/Fault masks from Vp gradients)
        # mask = compute_mask_from_vp(vp) 
        
        # 4. Normalize (-1 to 1 for gathers, MinMax for Vp)
        
        # return input_tensor, vp_tensor, mask_tensor
        pass 
"""

# ==========================================
# 4. SCRIPT: Main Training Loop
# ==========================================
files["scripts/train_gchim.py"] = """
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_unet import HybridGeoUNet
from utils.losses import CompositeLossScheduler, PathAwareLoss
from data.seismic_loader import SeismicVelocityDataset

# Hyperparams
EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 16

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = HybridGeoUNet(n_channels=5).to(device)
    
    # Loss Setup (Section 2.2.4)
    # Weights: [MAPE, SSIM, BCE, Salt]
    # Start: High Reconstruction focus. End: High Structural/Salt focus.
    loss_scheduler = CompositeLossScheduler(
        start_weights=[1.0, 0.1, 0.5, 0.1],
        end_weights=  [0.5, 1.0, 0.5, 1.0],
        total_epochs=EPOCHS
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # DataLoader (Placeholder)
    # train_loader = DataLoader(SeismicVelocityDataset(...), batch_size=BATCH_SIZE)
    
    print("Starting GCHIM Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(range(100)) # Placeholder for loader
        
        for i in loop:
            # Inputs (Mock data for shape verification)
            x = torch.randn(BATCH_SIZE, 5, 256, 256).to(device)
            target_vp = torch.randn(BATCH_SIZE, 1, 256, 256).to(device)
            target_mask = torch.randint(0, 2, (BATCH_SIZE, 1, 256, 256)).float().to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_vp, pred_mask = model(x)
            
            # Hybrid Loss
            loss, details = loss_scheduler(pred_vp, target_vp, pred_mask, target_mask, epoch)
            
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch {epoch}/{EPOCHS}")
            loop.set_postfix(loss=loss.item(), ssim_w=loss_scheduler.get_current_weights(epoch)[1].item())

        # Validation logic here...

if __name__ == "__main__":
    train()
"""

# ==========================================
# Execution
# ==========================================
def create_project():
    print(f"Creating GCHIM Project Structure...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"   [DIR] {d}/")
    
    for path, content in files.items():
        with open(path, 'w') as f:
            f.write(content.strip())
        print(f"   [FILE] {path}")
        
    print("\\nDone! Repo configured for AGU submission.")

if __name__ == "__main__":
    create_project()