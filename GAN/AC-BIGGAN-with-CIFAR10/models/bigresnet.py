"""
BIGGAN-Resnet
Original Paper: https://arxiv.org/pdf/1809.11096.pdf
Code Reference: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/models/big_resnet.py
"""

from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigGANConditionalBatchNorm2d(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.bn = nn.BatchNorm2d(out_features, eps=1e-4, momentum=0.1, affine=False)

        self.gain = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.bias = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hier_z_dim, apply_g_sn):
        super(GenBlock, self).__init__()

        # Conditional BN to Retain Class Information
        self.cbn1 = BigGANConditionalBatchNorm2d(hier_z_dim, in_channels)
        self.cbn2 = BigGANConditionalBatchNorm2d(hier_z_dim, out_channels)

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

        # Apply Spectral Norm
        if apply_g_sn:
            for m in [
                self.conv2d0,
                self.conv2d1,
                self.conv2d2
                ]:
                m = nn.utils.spectral_norm(m)

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
    def __init__(self, latent_dim, shared_embedding_dim, image_size, n_classes, g_conv_dim=96, apply_g_sn=False, **kwargs):
        super(Generator, self).__init__()
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.latent_dim = latent_dim
        self.shared_embedding_dim = shared_embedding_dim
        self.n_classes = n_classes
        self.in_dims = g_in_dims_collection[str(image_size)]
        self.out_dims = g_out_dims_collection[str(image_size)]
        self.bottom = bottom_collection[str(image_size)]
        self.num_blocks = len(self.in_dims)
        # Dividing latent dim for each resnet block
        self.latent_chunk_size = self.latent_dim // (self.num_blocks + 1)
        self.hier_z_dim = self.latent_chunk_size + self.shared_embedding_dim

        self.linear0 = nn.Linear(
            in_features=self.latent_chunk_size,
            out_features=self.in_dims[0] * self.bottom * self.bottom,
            bias=True,
        )

        self.shared_embedding = nn.Embedding(num_embeddings=self.n_classes, embedding_dim=self.shared_embedding_dim)

        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [
                GenBlock(
                    in_channels=self.in_dims[index],
                    out_channels=self.out_dims[index],
                    hier_z_dim=self.hier_z_dim,
                    apply_g_sn=apply_g_sn
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

        ### Apply Spectral Norm
        if apply_g_sn:
            self.linear0 = nn.utils.spectral_norm(self.linear0) 
            self.conv2d5 = nn.utils.spectral_norm(self.conv2d5)

    def forward(self, z, label):
        z_chunks = torch.split(z, self.latent_chunk_size, dim=1)
        z = z_chunks[0]

        shared_label_embedding = self.shared_embedding(label)
        labels = [torch.cat([shared_label_embedding, z_chunk], 1) for z_chunk in z_chunks]

        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)

        for index, block in enumerate(self.blocks):
            act = block(act, labels[index])

        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn):
        super(DiscOptBlock, self).__init__()
        # Spectral Norm for Discriminator
        self.apply_d_sn = apply_d_sn

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

        # BatchNorm only if not using Spectral Norm
        if not apply_d_sn:
            self.bn0 = nn.BatchNorm2d(num_features=in_channels)
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.activation = nn.LeakyReLU(0.2)

        self.average_pooling = nn.AvgPool2d(2)

        # Apply Spectral Norm
        if apply_d_sn:
            for m in [self.conv2d0, self.conv2d1, self.conv2d2]:
                m = nn.utils.spectral_norm(m)

    def forward(self, x):
        identity = x

        x = self.conv2d1(x)
        x = self.bn1(x) if not self.apply_d_sn else x
        x = self.activation(x)

        x = self.conv2d2(x)
        x = self.average_pooling(x)  # Res x1/2

        identity = self.average_pooling(identity)  # Res x1/2
        identity = self.bn0(identity) if not self.apply_d_sn else identity
        identity = self.conv2d0(identity)
        out = x + identity
        return out

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, downsample=True):
        super(DiscBlock, self).__init__()
        self.apply_d_sn = apply_d_sn
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
            if not apply_d_sn:
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

        if not apply_d_sn:
            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        identity = x

        x = self.bn1(x) if not self.apply_d_sn else x
        x = self.activation(x)
        x = self.conv2d1(x)

        x = self.bn2(x) if not self.apply_d_sn else x
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            identity = self.bn0(identity) if not self.apply_d_sn else identity
            identity = self.conv2d0(identity)
            if self.downsample:
                identity = self.average_pooling(identity)
        out = x + identity
        return out

class Discriminator(nn.Module):
    def __init__(self, image_size, n_classes, d_cond_mtd="AC", d_conv_dim=96, d_embed_dim=512, normalize_d_embed=False, apply_d_sn=False, **kwargs):
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

        self.d_cond_mtd = d_cond_mtd # Discriminator Conditional Method (AC, D2DCE)
        self.d_embed_dim = d_embed_dim # Embedding Dimension for Discriminator (Conditional Information)
        self.n_classes = n_classes
        self.normalize_d_embed = normalize_d_embed
        self.in_dims = d_in_dims_collection[str(image_size)]
        self.out_dims = d_out_dims_collection[str(image_size)]
        down = d_down[str(image_size)]

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [
                    DiscOptBlock(in_channels=self.in_dims[index],
                                 out_channels=self.out_dims[index],
                                 apply_d_sn=apply_d_sn
                                 )
                ]
            else:
                self.blocks += [
                    DiscBlock(in_channels=self.in_dims[index],
                              out_channels=self.out_dims[index],
                              apply_d_sn=apply_d_sn,
                              downsample=down[index])
                ]

        self.blocks = nn.ModuleList(self.blocks)

        self.activation = nn.LeakyReLU(0.2)

        # linear layer for adversarial training
        self.linear1 = nn.Linear(in_features=self.out_dims[-1], out_features=1, bias=True)

        # linear and embedding layers for discriminator conditioning
        if self.d_cond_mtd == "AC": # Auxillary Classification
            self.linear2 = nn.Linear(in_features=self.out_dims[-1], out_features=n_classes, bias=False)
        elif self.d_cond_mtd == "D2DCE": # Data-2-Data Cross Entropy
            self.linear_mi = nn.Linear(in_features=self.out_dims[-1], out_features=self.d_embed_dim, bias=True)
            self.embedding_mi = nn.Embedding(self.n_classes, self.d_embed_dim)
        else:
            raise NotImplementedError

    def forward(self, x, label=None):
        h = x
        for index, block in enumerate(self.blocks):
            h = block(h)
        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])

        # adversarial training
        adv_output = torch.squeeze(self.linear1(h))

        
        # class conditioning
        if self.d_cond_mtd == "AC":
            cls_output = self.linear2(h)
            classifier_output = [cls_output]
        elif self.d_cond_mtd == "D2DCE":
            mi_embed = self.linear_mi(h)
            mi_proxy = self.embedding_mi(label)
            if self.normalize_d_embed:
                mi_embed = F.normalize(mi_embed, dim=1)
                mi_proxy = F.normalize(mi_proxy, dim=1)
            classifier_output = [mi_embed, mi_proxy]
        else:
            raise NotImplementedError
        
        return [adv_output] + classifier_output