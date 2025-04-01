import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange

# -----------------------------------------------------------------------------------
# Parts of the season fusion module -- Sptial Channel Extraction & Fusion
# -----------------------------------------------------------------------------------


# fusion s2 data
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=2):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(
            x.size(0), -1, x.size(3), x.size(4)
        )  # Resulting shape: (B, in_chns * D, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


# ref: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/se.py
class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): No of input channels
            reduction_ratio (int): By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        Args:
            num_channels (int): No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None):
        """
        Args:
            weights (torch.Tensor): weights for few shot learning
            x: X, shape = (batch_size, num_channels, D, H, W)

        Returns:
            (torch.Tensor): output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = x.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(x, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
    3D extension of concurrent spatial and channel squeeze & excitation:
    *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): No of input channels
            reduction_ratio (int): By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        output_tensor = output_tensor.view(
            output_tensor.size(0), -1, output_tensor.size(3), output_tensor.size(4)
        )
        return output_tensor


class MF(nn.Module):  # Multi-Feature (MF) module for seasonal attention-based fusion
    def __init__(
        self, channels=12, seasons=4, rest_channel=1, spatial_att=False
    ):  # Each season has 13 channels
        super(MF, self).__init__()
        # Channel attention for each season (spring, summer, autumn, winter)
        self.channels = channels
        self.rest_channel = rest_channel
        self.seasons = seasons
        self.spatial_attention = spatial_att

        self.mask_map_summer = nn.Conv2d(self.channels, 1, 1, 1, 0, bias=True)
        self.mask_map_autumn = nn.Conv2d(self.channels, 1, 1, 1, 0, bias=True)
        self.bottleneck_summer = nn.Conv2d(self.channels, 16, 3, 1, 1, bias=False)
        self.bottleneck_autumn = nn.Conv2d(self.channels, 16, 3, 1, 1, bias=False)

        if self.seasons >= 4:
            self.mask_map_spring = nn.Conv2d(self.channels, 1, 1, 1, 0, bias=True)
            self.mask_map_winter = nn.Conv2d(self.channels, 1, 1, 1, 0, bias=True)
            self.bottleneck_spring = nn.Conv2d(self.channels, 16, 3, 1, 1, bias=False)
            self.bottleneck_winter = nn.Conv2d(self.channels, 16, 3, 1, 1, bias=False)
            if self.seasons >= 5:
                self.mask_map_rest = nn.Conv2d(self.rest_channel, 1, 1, 1, 0, bias=True)
                self.bottleneck_rest = nn.Conv2d(
                    self.rest_channel, 16, 3, 1, 1, bias=False
                )
        total_channels = 16 * self.seasons

        # Final SE Block for channel attention across all seasons
        if self.spatial_attention:
            self.se = ChannelSpatialSELayer3D(16, reduction_ratio=2)
        else:
            self.se = SE_Block(total_channels, reduction=16)

    def forward(self, x):  # x is a list of 4 inputs (spring, summer, autumn, winter)
        if self.seasons >= 4:
            if self.seasons == 4:
                spring, summer, autumn, winter = x
            elif self.seasons == 5:
                spring, summer, autumn, winter, rest = x
                rest_mask = torch.mul(
                    self.mask_map_rest(rest).repeat(1, self.rest_channel, 1, 1), rest
                )
            else:
                spring, summer, autumn, winter, phenology, climate, dem = x
                rest = torch.cat((phenology, climate, dem), dim=1)
                rest_mask = torch.mul(
                    self.mask_map_rest(rest).repeat(1, self.rest_channel, 1, 1), rest
                )
            spring_mask = torch.mul(
                self.mask_map_spring(spring).repeat(1, self.channels, 1, 1), spring
            )
            winter_mask = torch.mul(
                self.mask_map_winter(winter).repeat(1, self.channels, 1, 1), winter
            )
            spring_features = self.bottleneck_spring(spring_mask)
            winter_features = self.bottleneck_winter(winter_mask)
            if self.seasons >= 5:
                rest_feature = self.bottleneck_rest(rest_mask)
        else:
            summer, autumn = x

        # Apply attention maps
        summer_mask = torch.mul(
            self.mask_map_summer(summer).repeat(1, self.channels, 1, 1), summer
        )
        autumn_mask = torch.mul(
            self.mask_map_autumn(autumn).repeat(1, self.channels, 1, 1), autumn
        )

        # Apply bottleneck layers
        summer_features = self.bottleneck_summer(summer_mask)
        autumn_features = self.bottleneck_autumn(autumn_mask)

        # Concatenate along a new depth dimension (D)
        if self.seasons == 4:
            combined_features = torch.stack(
                [spring_features, summer_features, autumn_features, winter_features],
                dim=2,
            )  # Shape: (B, 16, D=4, H, W)
        elif self.seasons >= 5:
            combined_features = torch.stack(
                [
                    spring_features,
                    summer_features,
                    autumn_features,
                    winter_features,
                    rest_feature,
                ],
                dim=2,
            )
        else:
            combined_features = torch.stack(
                [summer_features, autumn_features], dim=2
            )  # Shape: (B, 16, D=2, H, W)

        # Apply SE Block for channel-wise attention
        out = self.se(combined_features)  # SE_Block takes 4D input

        return out


# -----------------------------------------------------------------------------------
# Parts of the U-Net model
# -----------------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# -----------------------------------------------------------------------------------
# Parts of the ResU-Net model
# -----------------------------------------------------------------------------------


class BNAct(nn.Module):
    """Batch Normalization followed by an optional ReLU activation."""

    def __init__(self, num_features, act=True):
        super(BNAct, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.act = act
        if self.act:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.activation(x)
        return x


# ConvBlock module
class ConvBlock(nn.Module):
    """Convolution Block with BN and Activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.bn_act = BNAct(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.bn_act(x)
        x = self.conv(x)
        return x


# Stem module
class Stem(nn.Module):
    """Initial convolution block with residual connection."""

    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_block = ConvBlock(out_channels, out_channels)
        self.shortcut_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0
        )
        self.bn_act = BNAct(out_channels, act=False)

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.conv_block(conv)
        shortcut = self.shortcut_conv(x)
        shortcut = self.bn_act(shortcut)
        output = conv + shortcut
        return output


# ResidualBlock module
class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and a shortcut connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv_block2 = ConvBlock(out_channels, out_channels)
        self.shortcut_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride
        )
        self.bn_act = BNAct(out_channels, act=False)

    def forward(self, x):
        res = self.conv_block1(x)
        res = self.conv_block2(res)
        shortcut = self.shortcut_conv(x)
        shortcut = self.bn_act(shortcut)
        output = res + shortcut
        return output


# UpSampleConcat module
class UpSampleConcat(nn.Module):
    """Upsamples the input and concatenates with the skip connection."""

    def __init__(self):
        super(UpSampleConcat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # else:
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x, xskip):
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up(x)
        x = torch.cat([x, xskip], dim=1)
        return x


# -----------------------------------------------------------------------------------
# MiniMamba blocks
# -----------------------------------------------------------------------------------


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class MambaLayer(nn.Module):
    def __init__(
        self,
        in_img_chs,  # Input channels for image
        dim=16,
        d_state=16,
        d_conv=4,
        expand=2,
        last_feat_size=128,
    ):
        super().__init__()

        # Pooling scales for the pooling layers
        pool_scales = self.generate_arithmetic_sequence(
            1, last_feat_size, last_feat_size // 4
        )
        self.pool_len = len(pool_scales)

        # Initialize pooling layers
        self.pool_layers = nn.ModuleList()

        # First pooling layer with 1x1 convolution and adaptive average pool
        self.pool_layers.append(
            nn.Sequential(
                ConvBNReLU(in_img_chs, dim, kernel_size=1), nn.AdaptiveAvgPool2d(1)
            )
        )

        # Add the rest of the pooling layers based on the pooling scales
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvBNReLU(in_img_chs, dim, kernel_size=1),
                )
            )

        # Mamba module
        self.mamba = Mamba(
            d_model=dim * self.pool_len
            + in_img_chs,  # Model dimension, to be set dynamically in forward
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        # Pool over points (max pooling over point cloud features)
        B, C, H, W = x.shape
        res = x
        ppm_out = [res]
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(
                pool_out, (H, W), mode="bilinear", align_corners=False
            )
            ppm_out.append(pool_out)
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        x = rearrange(x, "b c h w -> b (h w) c", b=B, c=chs, h=H, w=W)
        x = self.mamba(x)
        x = x.transpose(2, 1).view(B, chs, H, W)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence  # 1, 3, 5, 7


class MiniMambaBlock(nn.Module):
    def __init__(
        self,
        in_img_chs,  # Input channels for image
        dim=64,
        d_state=8,
        d_conv=4,
        expand=2,
        last_feat_size=128,
    ):
        super(MiniMambaBlock, self).__init__()
        self.mamba = MambaLayer(
            in_img_chs,
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            last_feat_size=last_feat_size,
        )

    def forward(self, img_emb, pc_emb):
        x = self.mamba(img_emb, pc_emb)
        return x
