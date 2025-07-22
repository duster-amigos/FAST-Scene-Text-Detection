import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- TextNet Backbone (ResNet-18 like, as in paper, but pure PyTorch) ----
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        try:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        except Exception as e:
            print(f"[ERROR] BasicBlock forward error: {e}")
            raise

class TextNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2]):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        try:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 3, 2, 1)
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            return [f1, f2, f3, f4]
        except Exception as e:
            print(f"[ERROR] TextNet forward error: {e}")
            raise

# ---- MKR Head (Multi-scale Kernel Representation) ----
class MKRHead(nn.Module):
    def __init__(self, in_channels=[64,128,256,512], out_channels=6):
        super().__init__()
        # FPN-like upsampling and fusion
        self.conv_f4 = nn.Conv2d(in_channels[3], 128, 1)
        self.conv_f3 = nn.Conv2d(in_channels[2], 128, 1)
        self.conv_f2 = nn.Conv2d(in_channels[1], 128, 1)
        self.conv_f1 = nn.Conv2d(in_channels[0], 128, 1)
        self.out_conv = nn.Conv2d(128, out_channels, 1)
    def forward(self, features):
        try:
            f1, f2, f3, f4 = features
            h, w = f1.shape[2:]
            f4 = F.interpolate(self.conv_f4(f4), size=(h, w), mode='bilinear', align_corners=False)
            f3 = F.interpolate(self.conv_f3(f3), size=(h, w), mode='bilinear', align_corners=False)
            f2 = F.interpolate(self.conv_f2(f2), size=(h, w), mode='bilinear', align_corners=False)
            f1 = self.conv_f1(f1)
            fuse = f1 + f2 + f3 + f4
            out = self.out_conv(fuse)
            return out
        except Exception as e:
            print(f"[ERROR] MKRHead forward error: {e}")
            raise

# ---- FAST Model ----
class FAST(nn.Module):
    def __init__(self, num_kernels=6):
        super().__init__()
        self.backbone = TextNet()
        self.head = MKRHead(out_channels=num_kernels)
    def forward(self, x):
        try:
            features = self.backbone(x)
            out = self.head(features)
            return out
        except Exception as e:
            print(f"[ERROR] FAST model forward error: {e}")
            raise 