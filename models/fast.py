import torch
import torch.nn as nn
import torch.nn.functional as F

class RepConvLayer(nn.Module):
    """Basic convolutional block: Conv2d + BatchNorm + ReLU. Used throughout the model."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__()
        # Standard 2D convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size[0]//2, kernel_size[1]//2), dilation=dilation, groups=groups, bias=False)
        # Batch normalization for stable training
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=0.001)
    def forward(self, x):
        # Apply conv, BN, and ReLU
        try:
            return F.relu(self.bn(self.conv(x)))
        except Exception as e:
            print(f"[ERROR] RepConvLayer forward error: {e}")
            raise

class TextNetSmall(nn.Module):
    """Backbone network for FAST. Extracts multi-scale features from the input image."""
    def __init__(self):
        super().__init__()
        # Initial convolutional stem: reduces spatial size and increases channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1, eps=0.001),
            nn.ReLU(inplace=True)
        )
        # Four stages of convolutional blocks, each downsampling and increasing channels
        self.stage1 = nn.Sequential(
            RepConvLayer(64, 64, (3,3), stride=1),
            RepConvLayer(64, 64, (3,3), stride=2)
        )
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
        self.stage4 = nn.Sequential(
            RepConvLayer(256, 512, (3,3), stride=2),
            RepConvLayer(512, 512, (3,1), stride=1),
            RepConvLayer(512, 512, (1,3), stride=1),
            RepConvLayer(512, 512, (1,3), stride=1),
            RepConvLayer(512, 512, (3,1), stride=1)
        )
    def forward(self, x):
        # Pass input through stem and all stages, collecting feature maps
        try:
            x = self.stem(x)      # [B, 64, H/2, W/2]
            f1 = self.stage1(x)   # [B, 64, H/4, W/4]
            f2 = self.stage2(f1)  # [B, 128, H/8, W/8]
            f3 = self.stage3(f2)  # [B, 256, H/16, W/16]
            f4 = self.stage4(f3)  # [B, 512, H/32, W/32]
            # Return all feature maps for neck
            return [f1, f2, f3, f4]
        except Exception as e:
            print(f"[ERROR] TextNetSmall forward error: {e}")
            raise

class FASTNeck(nn.Module):
    """Feature pyramid neck. Reduces and fuses backbone features to a common scale."""
    def __init__(self):
        super().__init__()
        # Reduce all backbone features to 128 channels
        self.reduce1 = RepConvLayer(64, 128, (3,3), stride=1)
        self.reduce2 = RepConvLayer(128, 128, (3,3), stride=1)
        self.reduce3 = RepConvLayer(256, 128, (3,3), stride=1)
        self.reduce4 = RepConvLayer(512, 128, (3,3), stride=1)
    def forward(self, features):
        try:
            # Reduce channels for each feature map
            r1 = self.reduce1(features[0])
            r2 = self.reduce2(features[1])
            r3 = self.reduce3(features[2])
            r4 = self.reduce4(features[3])
            # Upsample all features to the highest resolution (r1)
            h, w = r1.shape[2:]
            up_feats = [
                r1,
                F.interpolate(r2, size=(h, w), mode='bilinear', align_corners=False),
                F.interpolate(r3, size=(h, w), mode='bilinear', align_corners=False),
                F.interpolate(r4, size=(h, w), mode='bilinear', align_corners=False)
            ]
            # Concatenate along channel dimension: [B, 512, H/4, W/4]
            fuse = torch.cat(up_feats, dim=1)
            return fuse
        except Exception as e:
            print(f"[ERROR] FASTNeck forward error: {e}")
            raise

class FASTHead(nn.Module):
    """Head for predicting text and kernel segmentation maps from fused features."""
    def __init__(self, out_channels=5):
        super().__init__()
        # 3x3 conv to reduce channels
        self.conv = RepConvLayer(512, 128, (3,3), stride=1)
        # 1x1 conv to output segmentation maps (text + kernels)
        self.final = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        try:
            # Predict segmentation maps for text and kernels
            x = self.conv(x)
            x = self.final(x)
            return x
        except Exception as e:
            print(f"[ERROR] FASTHead forward error: {e}")
            raise

class FAST(nn.Module):
    """Full FAST model: backbone, neck, and head. Outputs text and kernel maps."""
    def __init__(self, num_kernels=5):
        super().__init__()
        # Build backbone, neck, and head
        self.backbone = TextNetSmall()
        self.neck = FASTNeck()
        self.head = FASTHead(out_channels=num_kernels)
    def forward(self, x):
        try:
            # Data flow: image -> backbone -> neck -> head -> output maps
            features = self.backbone(x)  # [f1, f2, f3, f4]
            fuse = self.neck(features)   # [B, 512, h, w]
            out = self.head(fuse)        # [B, num_kernels, h, w]
            return out
        except Exception as e:
            print(f"[ERROR] FAST model forward error: {e}")
            raise 