import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- RepConvLayer (Conv+BN+ReLU) ----
class RepConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size[0]//2, kernel_size[1]//2), dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=0.001)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

# ---- TextNetSmall Backbone (from fast_small.config) ----
class TextNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1, eps=0.001),
            nn.ReLU(inplace=True)
        )
        # Stage 1
        self.stage1 = nn.Sequential(
            RepConvLayer(64, 64, (3,3), stride=1),
            RepConvLayer(64, 64, (3,3), stride=2)
        )
        # Stage 2
        self.stage2 = nn.Sequential(
            RepConvLayer(64, 128, (3,3), stride=2),
            RepConvLayer(128, 128, (1,3), stride=1),
            RepConvLayer(128, 128, (3,3), stride=1),
            RepConvLayer(128, 128, (3,1), stride=1),
            RepConvLayer(128, 128, (3,3), stride=1),
            RepConvLayer(128, 128, (3,1), stride=1),
            RepConvLayer(128, 128, (1,3), stride=1),
            RepConvLayer(128, 128, (3,3), stride=1)
        )
        # Stage 3
        self.stage3 = nn.Sequential(
            RepConvLayer(128, 256, (3,3), stride=2),
            RepConvLayer(256, 256, (3,3), stride=1),
            RepConvLayer(256, 256, (1,3), stride=1),
            RepConvLayer(256, 256, (3,1), stride=1),
            RepConvLayer(256, 256, (3,3), stride=1),
            RepConvLayer(256, 256, (1,3), stride=1),
            RepConvLayer(256, 256, (3,1), stride=1),
            RepConvLayer(256, 256, (3,3), stride=1)
        )
        # Stage 4
        self.stage4 = nn.Sequential(
            RepConvLayer(256, 512, (3,3), stride=2),
            RepConvLayer(512, 512, (3,1), stride=1),
            RepConvLayer(512, 512, (1,3), stride=1),
            RepConvLayer(512, 512, (1,3), stride=1),
            RepConvLayer(512, 512, (3,1), stride=1)
        )
    def forward(self, x):
        try:
            x = self.stem(x)
            f1 = self.stage1(x)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
            return [f1, f2, f3, f4]
        except Exception as e:
            print(f"[ERROR] TextNetSmall forward error: {e}")
            raise

# ---- Neck: Reduce and Fuse Features ----
class FASTNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce1 = RepConvLayer(64, 128, (3,3), stride=1)
        self.reduce2 = RepConvLayer(128, 128, (3,3), stride=1)
        self.reduce3 = RepConvLayer(256, 128, (3,3), stride=1)
        self.reduce4 = RepConvLayer(512, 128, (3,3), stride=1)
    def forward(self, features):
        try:
            r1 = self.reduce1(features[0])
            r2 = self.reduce2(features[1])
            r3 = self.reduce3(features[2])
            r4 = self.reduce4(features[3])
            h, w = r1.shape[2:]
            up_feats = [
                r1,
                F.interpolate(r2, size=(h, w), mode='bilinear', align_corners=False),
                F.interpolate(r3, size=(h, w), mode='bilinear', align_corners=False),
                F.interpolate(r4, size=(h, w), mode='bilinear', align_corners=False)
            ]
            fuse = torch.cat(up_feats, dim=1)  # [B, 512, h, w]
            return fuse
        except Exception as e:
            print(f"[ERROR] FASTNeck forward error: {e}")
            raise

# ---- Head (from config) ----
class FASTHead(nn.Module):
    def __init__(self, out_channels=5):
        super().__init__()
        self.conv = RepConvLayer(512, 128, (3,3), stride=1)
        self.final = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        try:
            x = self.conv(x)
            x = self.final(x)
            return x
        except Exception as e:
            print(f"[ERROR] FASTHead forward error: {e}")
            raise

# ---- FAST Model ----
class FAST(nn.Module):
    def __init__(self, num_kernels=5):
        super().__init__()
        self.backbone = TextNetSmall()
        self.neck = FASTNeck()
        self.head = FASTHead(out_channels=num_kernels)
    def forward(self, x):
        try:
            features = self.backbone(x)  # [f1, f2, f3, f4]
            fuse = self.neck(features)   # [B, 512, h, w]
            out = self.head(fuse)        # [B, 5, h, w]
            return out
        except Exception as e:
            print(f"[ERROR] FAST model forward error: {e}")
            raise 