'''
adapted from
https://github.com/NREL/AEflow/blob/master/AEflow.py
for 2d flow with PyTorch
'''
from typing import Tuple
from functools import partial
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .utils import Act


class AEflowRawResBlock(nn.Module):

    def __init__(self, hidden_channels: int, kernel_size: int,
                 padding_type: str | Tuple[str, str]) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_channels=self.hidden_channels,
                               out_channels=self.hidden_channels,
                               kernel_size=self.kernel_size,
                               padding=self.kernel_size // 2,
                               padding_mode='zeros')

        self.act1 = Act('relu')

        self.conv2 = nn.Conv2d(in_channels=self.hidden_channels,
                               out_channels=self.hidden_channels,
                               kernel_size=self.kernel_size,
                               padding=self.kernel_size // 2,
                               padding_mode='zeros')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)
        y = y + x  # residual connection

        return y


class AEflowRawCompBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int,
                 padding_type: str | Tuple[str, str],
                 transpose: bool = False) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.transpose = transpose

        if self.transpose:

            self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.kernel_size // 2,
                                   padding_mode='zeros')

            self.conv2 = nn.ConvTranspose2d(in_channels=self.out_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=self.kernel_size,
                                            stride=2,  # upsample
                                            padding=self.kernel_size // 2,
                                            output_padding=1,
                                            padding_mode='zeros')
        else:
            self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.in_channels,
                                   kernel_size=self.kernel_size,
                                   stride=2,  # downsample
                                   padding=self.kernel_size // 2,
                                   padding_mode='zeros')

            self.conv2 = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.kernel_size // 2,
                                   padding_mode='zeros')

        self.act1 = Act('leaky')
        self.act2 = Act('leaky')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        return x


class AEflowRawEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 4,
                 latent_channels: int = 8,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 nresblocks: int = 12,
                 ncompblocks: int = 3,
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.nresblocks = nresblocks
        self.ncompblocks = ncompblocks

        self.first_conv = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.hidden_channels,
                                    kernel_size=5,
                                    padding=5 // 2,
                                    padding_mode='zeros')

        self.first_act = Act('leaky')

        self.resblock_seq = nn.Sequential(
            OrderedDict(
                [
                    (f'resblock{i+1}',
                     AEflowRawResBlock(self.hidden_channels,
                                       kernel_size=5,
                                       padding_type=padding_type))
                    for i in range(self.nresblocks)
                ]
            )
        )

        self.mid_conv = nn.Conv2d(in_channels=self.hidden_channels,
                                  out_channels=self.hidden_channels,
                                  kernel_size=5,
                                  padding=5 // 2,
                                  padding_mode='zeros')

        # (-1, 2, 128, 64)

        self.compblock_seq = nn.Sequential(
            OrderedDict(
                [
                    (f'compblock{i+1}',
                     AEflowRawCompBlock(self.hidden_channels * (2**i),
                                        self.hidden_channels * (2**(i + 1)),
                                        kernel_size=5,
                                        padding_type=padding_type))
                    for i in range(self.ncompblocks - 1)
                ]
                +
                [
                    (f'compblock{self.ncompblocks}',
                     AEflowRawCompBlock(self.hidden_channels * (2**(self.ncompblocks - 1)),
                                        self.hidden_channels * (2**(self.ncompblocks - 1)),
                                        kernel_size=5,
                                        padding_type=padding_type))
                ]
            )
        )

        # (-1, 16, 16, 8)

        self.last_conv = nn.Conv2d(in_channels=self.hidden_channels * (2**(self.ncompblocks - 1)),
                                   out_channels=self.latent_channels,
                                   kernel_size=5,
                                   padding=5 // 2,
                                   padding_mode='zeros')

        # (-1, 8, 16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.first_conv(x)
        x = self.first_act(x)

        y = self.resblock_seq(x)
        y = self.mid_conv(y)
        y = y + x  # residual connection for all resblocks

        y = self.compblock_seq(y)
        y = self.last_conv(y)

        return y


class AEflowRawDecoder(nn.Module):

    def __init__(self,
                 out_channels: int,
                 hidden_channels: int = 4,
                 latent_channels: int = 8,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 nresblocks: int = 12,
                 ncompblocks: int = 3,
                 ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.nresblocks = nresblocks
        self.ncompblocks = ncompblocks

        self.first_conv = nn.Conv2d(in_channels=self.latent_channels,
                                    out_channels=self.hidden_channels * (2**(self.ncompblocks - 1)),
                                    kernel_size=5,
                                    padding=5 // 2,
                                    padding_mode='zeros')

        self.compblock_seq = nn.Sequential(
            OrderedDict(
                [
                    (f'compblock{1}',
                     AEflowRawCompBlock(self.hidden_channels * (2**(self.ncompblocks - 1)),
                                        self.hidden_channels * (2**(self.ncompblocks - 1)),
                                        kernel_size=5,
                                        padding_type=padding_type,
                                        transpose=True))
                ]
                +
                [
                    (f'compblock{i+1}',
                     AEflowRawCompBlock(self.hidden_channels * (2**(self.ncompblocks - i)),
                                        self.hidden_channels * (2**(self.ncompblocks - i - 1)),
                                        kernel_size=5,
                                        padding_type=padding_type,
                                        transpose=True))
                    for i in range(1, self.ncompblocks)
                ]
            )
        )

        self.resblock_seq = nn.Sequential(
            OrderedDict(
                [
                    (f'resblock{i+1}',
                     AEflowRawResBlock(self.hidden_channels,
                                       kernel_size=5,
                                       padding_type=padding_type))
                    for i in range(self.nresblocks)
                ]
            )
        )

        self.mid_conv = nn.Conv2d(in_channels=self.hidden_channels,
                                  out_channels=self.hidden_channels,
                                  kernel_size=5,
                                  padding=5 // 2,
                                  padding_mode='zeros')

        self.last_conv = nn.Conv2d(in_channels=self.hidden_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=5,
                                   padding=5 // 2,
                                   padding_mode='zeros')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.first_conv(x)
        x = self.compblock_seq(x)

        y = self.resblock_seq(x)
        y = self.mid_conv(y)
        y = y + x  # residual connection for all resblocks

        y = self.last_conv(y)

        return y


class AEflowRaw(nn.Module):

    def __init__(self,
                 state_channels: int,
                 hidden_channels: int = 4,
                 latent_channels: int = 8,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 nresblocks: int = 12,
                 ncompblocks: int = 3,
                 **kwargs,
                 ) -> None:
        super().__init__()

        self.encoders = AEflowRawEncoder(in_channels=state_channels,
                                         hidden_channels=hidden_channels,
                                         latent_channels=latent_channels,
                                         padding_type=padding_type,
                                         nresblocks=nresblocks,
                                         ncompblocks=ncompblocks)

        self.decoders = AEflowRawDecoder(out_channels=state_channels,
                                         hidden_channels=hidden_channels,
                                         latent_channels=latent_channels,
                                         padding_type=padding_type,
                                         nresblocks=nresblocks,
                                         ncompblocks=ncompblocks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoders(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoders(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        y = self.decode(z)
        return y
