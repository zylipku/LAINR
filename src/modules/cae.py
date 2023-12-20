'''
trivial CNN implementation
custom mixed padding
'''
from typing import Tuple
from collections import OrderedDict

import torch
from torch import nn

from .utils import Act, Conv2dCustomBD, ConvTranspose2dCustomBD


class CNNblock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 transpose: bool = False) -> None:

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.transpose = transpose

        if self.transpose:

            self.conv1 = Conv2dCustomBD(self.in_channels,
                                        self.out_channels,
                                        kernel_size=self.kernel_size,
                                        padding_type=padding_type)

            self.conv2 = ConvTranspose2dCustomBD(self.out_channels,
                                                 self.out_channels,
                                                 kernel_size=self.kernel_size,
                                                 padding_type=padding_type,
                                                 stride=2)  # upsample
        else:
            self.conv1 = Conv2dCustomBD(self.in_channels,
                                        self.in_channels,
                                        kernel_size=self.kernel_size,
                                        padding_type=padding_type,
                                        stride=2)  # downsample

            self.conv2 = Conv2dCustomBD(self.in_channels,
                                        self.out_channels,
                                        kernel_size=self.kernel_size,
                                        padding_type=padding_type)

        self.act1 = Act('leaky')
        self.act2 = Act('leaky')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        return x


class CNNencoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 16,
                 latent_channels: int = 8,
                 kernel_size: int = 5,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 depth: int = 3,
                 act_name='leaky',
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.kernel_size = kernel_size
        self.depth = depth

        self.act = Act(act_name)

        self.compblock_seq = nn.Sequential(
            OrderedDict(
                [
                    ('compblock1',
                     CNNblock(in_channels=self.in_channels,
                              out_channels=self.hidden_channels,
                              kernel_size=kernel_size,
                              padding_type=padding_type)
                     )
                ]
                +
                [
                    (f'compblock{i+1}',
                     CNNblock(in_channels=self.hidden_channels,
                              out_channels=self.hidden_channels,
                              kernel_size=kernel_size,
                              padding_type=padding_type)
                     ) for i in range(1, self.depth - 1)
                ]
                +
                [
                    (f'compblock{self.depth}',
                     CNNblock(in_channels=self.hidden_channels,
                              out_channels=self.latent_channels,
                              kernel_size=kernel_size,
                              padding_type=padding_type)
                     )
                ]
            )
        )
        # (-1, 2, 128, 64)
        # (-1, 16, 64, 32)
        # (-1, 16, 32, 16)
        # (-1, 8, 16, 8) latent_dim=1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compblock_seq(x)
        return x


class CNNdecoder(nn.Module):

    def __init__(self,
                 out_channels: int,
                 hidden_channels: int = 16,
                 latent_channels: int = 8,
                 kernel_size: int = 5,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 depth: int = 3,
                 act_name='leaky',
                 ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.kernel_size = kernel_size
        self.depth = depth

        self.act = Act(act_name)

        self.decompblock_seq = nn.Sequential(
            OrderedDict(
                [
                    ('decompblock1',
                     CNNblock(in_channels=self.latent_channels,
                              out_channels=self.hidden_channels,
                              kernel_size=kernel_size,
                              padding_type=padding_type,
                              transpose=True)
                     )
                ]
                +
                [
                    (f'decompblock{i+1}',
                     CNNblock(in_channels=self.hidden_channels,
                              out_channels=self.hidden_channels,
                              kernel_size=kernel_size,
                              padding_type=padding_type,
                              transpose=True)
                     ) for i in range(1, self.depth - 1)
                ]
                +
                [
                    (f'decompblock{self.depth}',
                     CNNblock(in_channels=self.hidden_channels,
                              out_channels=self.out_channels,
                              kernel_size=kernel_size,
                              padding_type=padding_type,
                              transpose=True)
                     )
                ]
            )
        )
        # (-1, 2, 128, 64)
        # (-1, 16, 64, 32)
        # (-1, 16, 32, 16)
        # (-1, 8, 16, 8) latent_dim=1024 (reversed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decompblock_seq(x)
        return x


class CAE(nn.Module):

    def __init__(self,
                 state_channels: int,
                 hidden_channels: int = 16,
                 latent_channels: int = 8,
                 kernel_size: int = 5,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 depth: int = 3,
                 act_name='leaky',
                 **kwargs) -> None:
        super().__init__()

        self.encoders = CNNencoder(in_channels=state_channels,
                                   hidden_channels=hidden_channels,
                                   latent_channels=latent_channels,
                                   kernel_size=kernel_size,
                                   padding_type=padding_type,
                                   depth=depth,
                                   act_name=act_name)

        self.decoders = CNNdecoder(out_channels=state_channels,
                                   hidden_channels=hidden_channels,
                                   latent_channels=latent_channels,
                                   kernel_size=kernel_size,
                                   padding_type=padding_type,
                                   depth=depth,
                                   act_name=act_name)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoders(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoders(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        y = self.decode(z)
        return y
