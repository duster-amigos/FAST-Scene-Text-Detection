import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Custom Blocks for TextNetSmall ----
class Conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Conv1x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Conv3x1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=(3,1), stride=stride, padding=(1,0), bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Identity(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.need_proj = (in_planes != out_planes) or (stride != 1)
        if self.need_proj:
            self.proj = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        if self.need_proj:
            return F.relu(self.bn(self.proj(x)))
        return x

# ---- TextNetSmall Backbone (from fast_small.config) ----
# Block types: 0=Conv3x3, 1=Conv1x3, 2=Conv3x1, 3=Identity
# These are the block types for each stage, from the official config
stage_blocks = [
    # Stage 1 (C=64)
    [0, 0, 0],
    # Stage 2 (C=128)
    [0, 1, 2, 0, 1, 2],
    # Stage 3 (C=256)
    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    # Stage 4 (C=512)
    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
]
block_map = [Conv3x3, Conv1x3, Conv3x1, Identity]
channels = [64, 128, 256, 512]

class TextNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stages = nn.ModuleList()
        in_planes = 64
        for stage_idx, blocks in enumerate(stage_blocks):
            out_planes = channels[stage_idx]
            stage = []
            for i, block_type in enumerate(blocks):
                stride = 2 if (i == 0) else 1  # Downsample at first block of each stage
                block = block_map[block_type](in_planes, out_planes, stride)
                stage.append(block)
                in_planes = out_planes
            self.stages.append(nn.Sequential(*stage))
    def forward(self, x):
        try:
            x = self.stem(x)
            feats = []
            for stage in self.stages:
                x = stage(x)
                feats.append(x)
            return feats  # [f1, f2, f3, f4]
        except Exception as e:
            print(f"[ERROR] TextNetSmall forward error: {e}")
            raise

# ---- MKR Head (unchanged, as before) ----
class MKRHead(nn.Module):
    def __init__(self, in_channels=[64,128,256,512], out_channels=6):
        super().__init__()
        self.conv_f4 = nn.Conv2d(in_channels[3], 128, 3, padding=1)
        self.conv_f3 = nn.Conv2d(in_channels[2], 128, 3, padding=1)
        self.conv_f2 = nn.Conv2d(in_channels[1], 128, 3, padding=1)
        self.conv_f1 = nn.Conv2d(in_channels[0], 128, 3, padding=1)
        self.head_conv1 = nn.Conv2d(128*4, 256, 3, padding=1)
        self.head_conv2 = nn.Conv2d(256, out_channels, 1)
    def forward(self, features):
        try:
            f1, f2, f3, f4 = features
            h, w = f1.shape[2:]
            f4 = F.interpolate(self.conv_f4(f4), size=(h, w), mode='bilinear', align_corners=False)
            f3 = F.interpolate(self.conv_f3(f3), size=(h, w), mode='bilinear', align_corners=False)
            f2 = F.interpolate(self.conv_f2(f2), size=(h, w), mode='bilinear', align_corners=False)
            f1 = self.conv_f1(f1)
            fuse = torch.cat([f1, f2, f3, f4], dim=1)
            out = F.relu(self.head_conv1(fuse))
            out = self.head_conv2(out)
            return out
        except Exception as e:
            print(f"[ERROR] MKRHead forward error: {e}")
            raise

# ---- FAST Model ----
class FAST(nn.Module):
    def __init__(self, num_kernels=6):
        super().__init__()
        self.backbone = TextNetSmall()
        self.head = MKRHead(out_channels=num_kernels)
    def forward(self, x):
        try:
            features = self.backbone(x)
            out = self.head(features)
            return out
        except Exception as e:
            print(f"[ERROR] FAST model forward error: {e}")
            raise 