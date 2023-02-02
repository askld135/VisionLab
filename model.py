from torch import nn, cat
import torch.nn.functional as F
from torch.nn.functional import affine_grid, grid_sample
from torchvision.utils import save_image

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        
        self.pad = nn.ReflectionPad2d(padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels,3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3)
        
    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        res = self.pad(x)
        res = self.conv2(res)
        res = self.leaky_relu(res)
        
        res = self.pad(res)
        res = self.conv3(res)
        res = self.leaky_relu(res)
        
        out = res + x
        
        return out

class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResUNet, self).__init__()
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        self.down1 = ResBlock(in_channels,32)
        self.down2 = ResBlock(32, 64)
        self.down3 = ResBlock(64, 128)
        self.down4 = ResBlock(128, 256)
        
        self.inter_conv = ResBlock(256, 512)
        
        self.up4 = ResBlock(512 + 256, 256)
        self.up3 = ResBlock(256 + 128, 128)
        self.up2 = ResBlock(128 + 64, 64)
        self.up1 = ResBlock(64 + 32, 32)
        
        self.last_conv = nn.Sequential(nn.ReflectionPad2d(padding=1),
                                       nn.Conv2d(32, out_channels, 3))

    def forward(self, x):
        
        conv1 = self.down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.down4(x)
        x = self.maxpool(conv4)
        
        low_features = self.inter_conv(x)
        
        x = F.interpolate(low_features, (conv4.size(-2), conv4.size(-1)), mode='bilinear')
        
        x = cat([x, conv4], dim=1)
        x = self.up4(x)
        
        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)
        
        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)
        
        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)
        
        out = self.last_conv(x)
        
        return out
    
    
        
        