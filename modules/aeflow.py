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


class Conv2dCustomBD(nn.Module):

    '''
    padding_type: 'circular', 'reflect' or 'replicate'
    the output size is the same as the input size
    the padding is done in the forward pass, with padding size
    default as kernel_size // 2 for odd kernel size
    ** no activation is applied **
    '''

    VAILD_PADDING_TYPES = ['circular', 'reflect', 'replicate']

    def __init__(self,
                 in_channels: int, out_channels: int, kernel_size=5,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 padding: int = None,
                 stride: int = 1,
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(padding_type, str):
            padding_type = (padding_type, padding_type)

        assert all(p in self.VAILD_PADDING_TYPES for p in padding_type)

        self.kernel_size = kernel_size

        # load padding
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding

        self.stride = stride

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=0,  # custom padding
                              stride=self.stride)

        self.pad0 = partial(
            F.pad,
            pad=(0, 0, self.padding, self.padding),
            mode=padding_type[0]
        )
        self.pad1 = partial(
            F.pad,
            pad=(self.padding, self.padding, 0, 0),
            mode=padding_type[1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pad0(x)
        x = self.pad1(x)
        x = self.conv(x)

        return x


class ConvTranspose2dCustomBD(nn.Module):

    '''
    padding_type: 'circular', 'reflect' or 'replicate'
    the output size is the same as the input size
    the padding is done in the forward pass, with padding size
    default as kernel_size // 2 for odd kernel size
    ** no activation is applied **
    '''

    VAILD_PADDING_TYPES = ['circular', 'reflect', 'replicate']

    def __init__(self,
                 in_channels: int, out_channels: int, kernel_size=5,
                 padding_type: str | Tuple[str, str] = 'periodic',
                 padding: int = None,
                 stride: int = 1,
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(padding_type, str):
            padding_type = (padding_type, padding_type)

        assert all(p in self.VAILD_PADDING_TYPES for p in padding_type)

        self.kernel_size = kernel_size

        # load padding
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding

        self.stride = stride

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=0,  # custom padding
                              stride=1)

        self.pad0 = partial(
            F.pad,
            pad=(0, 0, self.padding // stride, self.padding // stride),
            mode=padding_type[0]
        )
        self.pad1 = partial(
            F.pad,
            pad=(self.padding // stride, self.padding // stride, 0, 0),
            mode=padding_type[1]
        )

    def dilation(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        '''
        [[1, 2, 3,],
         [4, 5, 6,],
         [7, 8, 9,]]
        to
        [[1, 0, 2, 0, 3, 0,],
         [0, 0, 0, 0, 0, 0,],
         [4, 0, 5, 0, 6, 0,],
         [0, 0, 0, 0, 0, 0,],
         [7, 0, 8, 0, 9, 0,],
         [0, 0, 0, 0, 0, 0,]]
        with dilation=2
        using F.convTranspose2d
        '''
        pass
        kernel_size = dilation * 2 + dilation - 2
        bs, c, h, w = x.shape
        dilation_kernel = torch.zeros(c, c, kernel_size, kernel_size).to(x)
        dilation_kernel[:, :, dilation - 1, dilation - 1] = torch.eye(c).to(x)
        # dilation_kernel[0, 0, dilation_factor - 1, dilation_factor - 1] = 1 # for c=3
        # dilation_kernel[1, 1, dilation_factor - 1, dilation_factor - 1] = 1 # for c=3
        # dilation_kernel[2, 2, dilation_factor - 1, dilation_factor - 1] = 1 # for c=3

        # Apply dilation
        x = F.conv_transpose2d(x, dilation_kernel, stride=dilation, padding=dilation - 1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pad0(x)
        x = self.pad1(x)
        x = self.dilation(x, dilation=self.stride)
        x = self.conv(x)

        return x


class AEflowResBlock(nn.Module):

    def __init__(self, hidden_channels: int, kernel_size: int,
                 padding_type: str | Tuple[str, str]) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.conv1 = Conv2dCustomBD(self.hidden_channels,
                                    self.hidden_channels,
                                    kernel_size=self.kernel_size,
                                    padding_type=padding_type)
        self.act1 = Act('relu')
        self.conv2 = Conv2dCustomBD(self.hidden_channels,
                                    self.hidden_channels,
                                    kernel_size=self.kernel_size,
                                    padding_type=padding_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)
        y = y + x  # residual connection

        return y


class AEflowCompBlock(nn.Module):

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
            self.conv1 = ConvTranspose2dCustomBD(self.in_channels,
                                                 self.in_channels,
                                                 kernel_size=self.kernel_size,
                                                 padding_type=padding_type,
                                                 stride=2)  # upsample
        else:
            self.conv1 = Conv2dCustomBD(self.in_channels,
                                        self.in_channels,
                                        kernel_size=self.kernel_size,
                                        padding_type=padding_type,
                                        stride=2)  # downsample
        self.act1 = Act('leaky')
        self.conv2 = Conv2dCustomBD(self.in_channels,
                                    self.out_channels,
                                    kernel_size=self.kernel_size,
                                    padding_type=padding_type)
        self.act2 = Act('leaky')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        return x


class AEflowEncoder(nn.Module):

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

        self.first_conv = Conv2dCustomBD(self.in_channels,
                                         self.hidden_channels,
                                         kernel_size=5,
                                         padding_type=padding_type)
        self.first_act = Act('leaky')

        self.resblock_seq = nn.Sequential(
            OrderedDict(
                [
                    (f'resblock{i+1}',
                     AEflowResBlock(self.hidden_channels,
                                    kernel_size=5,
                                    padding_type=padding_type))
                    for i in range(self.nresblocks)
                ]
            )
        )

        self.mid_conv = Conv2dCustomBD(self.hidden_channels,
                                       self.hidden_channels,
                                       kernel_size=5,
                                       padding_type=padding_type)

        self.compblock_seq = nn.Sequential(
            OrderedDict(
                [
                    (f'compblock{i+1}',
                     AEflowCompBlock(self.hidden_channels * (2**i),
                                     self.hidden_channels * (2**(i + 1)),
                                     kernel_size=5,
                                     padding_type=padding_type))
                    for i in range(self.ncompblocks - 1)
                ]
                +
                [
                    (f'compblock{self.ncompblocks}',
                     AEflowCompBlock(self.hidden_channels * (2**(self.ncompblocks - 1)),
                                     self.hidden_channels * (2**(self.ncompblocks - 1)),
                                     kernel_size=5,
                                     padding_type=padding_type))
                ]
            )
        )

        self.last_conv = Conv2dCustomBD(self.hidden_channels * (2**(self.ncompblocks - 1)),
                                        self.latent_channels,
                                        kernel_size=5,
                                        padding_type=padding_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.first_conv(x)
        x = self.first_act(x)

        y = self.resblock_seq(x)
        y = self.mid_conv(y)
        y = y + x  # residual connection for all resblocks

        y = self.compblock_seq(y)
        y = self.last_conv(y)

        return y


class AEflowDecoder(nn.Module):

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

        self.first_conv = Conv2dCustomBD(self.latent_channels,
                                         self.hidden_channels * (2**(self.ncompblocks - 1)),
                                         kernel_size=5,
                                         padding_type=padding_type)

        self.compblock_seq = nn.Sequential(
            OrderedDict(
                [
                    (f'compblock{1}',
                     AEflowCompBlock(self.hidden_channels * (2**(self.ncompblocks - 1)),
                                     self.hidden_channels * (2**(self.ncompblocks - 1)),
                                     kernel_size=5,
                                     padding_type=padding_type,
                                     transpose=True))
                ]
                +
                [
                    (f'compblock{i+1}',
                     AEflowCompBlock(self.hidden_channels * (2**(self.ncompblocks - i)),
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
                     AEflowResBlock(self.hidden_channels,
                                    kernel_size=5,
                                    padding_type=padding_type))
                    for i in range(self.nresblocks)
                ]
            )
        )

        self.mid_conv = Conv2dCustomBD(self.hidden_channels,
                                       self.hidden_channels,
                                       kernel_size=5,
                                       padding_type=padding_type)
        self.last_conv = Conv2dCustomBD(self.hidden_channels,
                                        self.out_channels,
                                        kernel_size=5,
                                        padding_type=padding_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.first_conv(x)
        x = self.compblock_seq(x)

        y = self.resblock_seq(x)
        y = self.mid_conv(y)
        y = y + x  # residual connection for all resblocks

        y = self.last_conv(y)

        return y


class AEflow(nn.Module):

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

        self.encoders = AEflowEncoder(in_channels=state_channels,
                                      hidden_channels=hidden_channels,
                                      latent_channels=latent_channels,
                                      padding_type=padding_type,
                                      nresblocks=nresblocks,
                                      ncompblocks=ncompblocks)

        self.decoders = AEflowDecoder(out_channels=state_channels,
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
