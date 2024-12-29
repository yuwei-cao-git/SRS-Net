import torch.nn as nn
import torch.nn.functional as F
from .blocks import Stem, ResidualBlock, ConvBlock, UpSampleConcat


# ResidualUNet model
class ResUnet(nn.Module):
    def __init__(self, n_channels=52, n_classes=9):
        super(ResUnet, self).__init__()
        # f = [16, 32, 64, 128, 256]  # Filter sizes
        # f = [64, 128, 256, 512, 1024]
        f = [32, 64, 128, 256, 512]
        # Encoder
        self.stem = Stem(n_channels, f[0])
        self.residual_block1 = ResidualBlock(f[0], f[1], stride=2)
        self.residual_block2 = ResidualBlock(f[1], f[2], stride=2)
        self.residual_block3 = ResidualBlock(f[2], f[3], stride=2)
        self.residual_block4 = ResidualBlock(f[3], f[4], stride=2)

        # Bridge
        self.conv_block1 = ConvBlock(f[4], f[4])
        self.conv_block2 = ConvBlock(f[4], f[4])

        # Decoder
        self.upsample_concat = UpSampleConcat()
        self.residual_block_d1 = ResidualBlock(
            in_channels=f[4] + f[3], out_channels=f[3]
        )
        self.residual_block_d2 = ResidualBlock(
            in_channels=f[3] + f[2], out_channels=f[2]
        )
        self.residual_block_d3 = ResidualBlock(
            in_channels=f[2] + f[1], out_channels=f[1]
        )
        self.residual_block_d4 = ResidualBlock(
            in_channels=f[1] + f[0], out_channels=f[0]
        )

        # Output layer
        self.out_conv = nn.Conv2d(f[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.stem(x)  # f[0] channels
        e2 = self.residual_block1(e1)  # f[1] channels
        e3 = self.residual_block2(e2)  # f[2] channels
        e4 = self.residual_block3(e3)  # f[3] channels
        e5 = self.residual_block4(e4)  # f[4] channels

        # Bridge
        b0 = self.conv_block1(e5)  # f[4] channels
        b1 = self.conv_block2(b0)  # f[4] channels

        # Decoder
        u1 = self.upsample_concat(b1, e4)  # f[4]+f[3] channels
        d1 = self.residual_block_d1(u1)  # f[3] channels

        u2 = self.upsample_concat(d1, e3)  # f[3]+f[2] channels
        d2 = self.residual_block_d2(u2)  # f[2] channels

        u3 = self.upsample_concat(d2, e2)  # f[2]+f[1] channels
        d3 = self.residual_block_d3(u3)  # f[1] channels

        u4 = self.upsample_concat(d3, e1)  # f[1]+f[0] channels
        d4 = self.residual_block_d4(u4)  # f[0] channels

        logits = self.out_conv(d4)  # n_classes channels
        preds = F.softmax(logits, dim=1)
        return preds, b1
