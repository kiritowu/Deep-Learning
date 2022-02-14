"""
GAN Resnet
Reference : https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/models/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, n_classes, num_features):
        super().__init__()
        self.num_features = num_features
        # Disable Affine Transformation and replace with learned embedding with class information
        self.bn = nn.BatchNorm2d(
            num_features=num_features, eps=1e-4, momentum=0.1, affine=False
        )

        self.embed0 = nn.Embedding(num_embeddings=n_classes, embedding_dim=num_features)
        self.embed1 = nn.Embedding(num_embeddings=n_classes, embedding_dim=num_features)

    def forward(self, x, y):
        gain = (1 + self.embed0(y)).view(-1, self.num_features, 1, 1)
        bias = self.embed1(y).view(-1, self.num_features, 1, 1)
        out = self.bn(x)
        return out * gain + bias


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super(GenBlock, self).__init__()

        # Conditional BN to Retain Class Information
        self.cbn1 = ConditionalBatchNorm2d(n_classes, in_channels)
        self.cbn2 = ConditionalBatchNorm2d(n_classes, out_channels)

        self.activation = nn.LeakyReLU(0.2)
        # 1x1 Conv for Channel Expansion
        self.conv2d0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # 3x3 Conv for Channel Expansion
        self.conv2d1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # 3x3 Conv with same Channel Dimension
        self.conv2d2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, label):
        # Skip Connection
        identity = x

        x = self.cbn1(x, label)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # Res x2
        x = self.conv2d1(x)
        x = self.cbn2(x, label)
        x = self.activation(x)
        x = self.conv2d2(x)

        identity = F.interpolate(identity, scale_factor=2, mode="nearest")  # Res x2
        identity = self.conv2d0(identity)  # Channel Expansion
        out = x + identity
        return out


class Generator(nn.Module):
    def __init__(self, latent_dim, image_size, n_classes, g_conv_dim=64, **kwargs):
        super(Generator, self).__init__()
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [
                g_conv_dim * 16,
                g_conv_dim * 16,
                g_conv_dim * 8,
                g_conv_dim * 4,
                g_conv_dim * 2,
            ],
            "256": [
                g_conv_dim * 16,
                g_conv_dim * 16,
                g_conv_dim * 8,
                g_conv_dim * 8,
                g_conv_dim * 4,
                g_conv_dim * 2,
            ],
            "512": [
                g_conv_dim * 16,
                g_conv_dim * 16,
                g_conv_dim * 8,
                g_conv_dim * 8,
                g_conv_dim * 4,
                g_conv_dim * 2,
                g_conv_dim,
            ],
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [
                g_conv_dim * 16,
                g_conv_dim * 8,
                g_conv_dim * 4,
                g_conv_dim * 2,
                g_conv_dim,
            ],
            "256": [
                g_conv_dim * 16,
                g_conv_dim * 8,
                g_conv_dim * 8,
                g_conv_dim * 4,
                g_conv_dim * 2,
                g_conv_dim,
            ],
            "512": [
                g_conv_dim * 16,
                g_conv_dim * 8,
                g_conv_dim * 8,
                g_conv_dim * 4,
                g_conv_dim * 2,
                g_conv_dim,
                g_conv_dim,
            ],
        }

        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.in_dims = g_in_dims_collection[str(image_size)]
        self.out_dims = g_out_dims_collection[str(image_size)]
        self.bottom = bottom_collection[str(image_size)]
        self.num_blocks = len(self.in_dims)

        self.linear0 = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.in_dims[0] * self.bottom * self.bottom,
            bias=True,
        )

        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [
                GenBlock(
                    in_channels=self.in_dims[index],
                    out_channels=self.out_dims[index],
                    n_classes=self.n_classes,
                )
            ]

        self.blocks = nn.ModuleList(self.blocks)

        self.bn4 = nn.BatchNorm2d(num_features=self.out_dims[-1])
        self.activation = nn.LeakyReLU(0.2)
        self.conv2d5 = nn.Conv2d(
            in_channels=self.out_dims[-1],
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.tanh = nn.Tanh()

    def forward(self, z, label):
        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
        for index, block in enumerate(self.blocks):
            act = block(act, label)

        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()

        # 1x1 Conv for Channel Expansion
        self.conv2d0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # 3x3 Conv for Channel Expansion
        self.conv2d1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # 3x3 Conv with same Channel Dimension
        self.conv2d2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bn0 = nn.BatchNorm2d(num_features=in_channels)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.activation = nn.LeakyReLU(0.2)

        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        identity = x

        x = self.conv2d1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2d2(x)
        x = self.average_pooling(x)  # Res x1/2

        identity = self.average_pooling(identity)  # Res x1/2
        identity = self.bn0(identity)
        identity = self.conv2d0(identity)
        out = x + identity
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DiscBlock, self).__init__()
        self.downsample = downsample

        self.activation = nn.LeakyReLU(0.2)

        # Define if in_channels != out_channels for downsampling
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if self.ch_mismatch or downsample:
            self.conv2d0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.bn0 = nn.BatchNorm2d(num_features=in_channels)

        self.conv2d1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2d2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)

        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            identity = self.bn0(identity)
            identity = self.conv2d0(identity)
            if self.downsample:
                identity = self.average_pooling(identity)
        out = x + identity
        return out

class Discriminator(nn.Module):
    def __init__(self, image_size, n_classes, d_conv_dim=64, **kwargs):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {
            "32": [3] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [3] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8],
            "128": [3] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "256": [3] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16],
            "512": [3] + [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16]
        }

        d_out_dims_collection = {
            "32": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "128": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "256": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "512":
            [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16]
        }

        d_down = {
            "32": [True, True, False, False],
            "64": [True, True, True, True, False],
            "128": [True, True, True, True, True, False],
            "256": [True, True, True, True, True, True, False],
            "512": [True, True, True, True, True, True, True, False]
        }

        self.n_classes = n_classes
        self.in_dims = d_in_dims_collection[str(image_size)]
        self.out_dims = d_out_dims_collection[str(image_size)]
        down = d_down[str(image_size)]

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [
                    DiscOptBlock(in_channels=self.in_dims[index], out_channels=self.out_dims[index])
                ]
            else:
                self.blocks += [
                    DiscBlock(in_channels=self.in_dims[index],
                              out_channels=self.out_dims[index],
                              downsample=down[index])
                ]

        self.blocks = nn.ModuleList(self.blocks)

        self.activation = nn.LeakyReLU(0.2)

        # linear layer for adversarial training
        self.linear1 = nn.Linear(in_features=self.out_dims[-1], out_features=1, bias=True)

        # linear and embedding layers for discriminator conditioning
        self.linear2 = nn.Linear(in_features=self.out_dims[-1], out_features=n_classes, bias=False)

    def forward(self, x):
        h = x
        for index, block in enumerate(self.blocks):
            h = block(h)
        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])

        # adversarial training
        adv_output = torch.squeeze(self.linear1(h))

        # class conditioning
        cls_output = self.linear2(h)

        return adv_output, cls_output