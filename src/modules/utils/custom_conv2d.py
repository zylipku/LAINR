from typing import Tuple

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


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
                 padding_type: str | Tuple[str, str] = 'circular',
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
                 padding_type: str | Tuple[str, str] = 'circular',
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
            pad=(0, 0, self.padding, self.padding),
            mode=padding_type[0]
        )
        self.pad1 = partial(
            F.pad,
            pad=(self.padding, self.padding, 0, 0),
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

        x = self.dilation(x, dilation=self.stride)
        x = self.pad0(x)
        x = self.pad1(x)
        x = self.conv(x)

        return x

#! deprecated !!!


# class ConvTranspose2dCustomBD2(nn.Module):

#     '''
#     padding_type: 'circular', 'reflect' or 'replicate'
#     the output size is the same as the input size
#     the padding is done in the forward pass, with padding size
#     default as kernel_size // 2 for odd kernel size
#     ** no activation is applied **
#     '''

#     VAILD_PADDING_TYPES = ['circular', 'reflect', 'replicate']

#     def __init__(self,
#                  in_channels: int, out_channels: int, kernel_size=5,
#                  padding_type: str | Tuple[str, str] = 'circular',
#                  padding: int = None,
#                  stride: int = 1,
#                  ) -> None:
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         if isinstance(padding_type, str):
#             padding_type = (padding_type, padding_type)

#         assert all(p in self.VAILD_PADDING_TYPES for p in padding_type)

#         self.kernel_size = kernel_size

#         # load padding
#         if padding is None:
#             self.padding = kernel_size // 2
#         else:
#             self.padding = padding

#         self.stride = stride

#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size=kernel_size,
#                               padding=0,  # custom padding
#                               stride=1)

#         self.pad0 = partial(
#             F.pad,
#             pad=(0, 0, self.padding // stride, self.padding // stride),
#             mode=padding_type[0]
#         )
#         self.pad1 = partial(
#             F.pad,
#             pad=(self.padding // stride, self.padding // stride, 0, 0),
#             mode=padding_type[1]
#         )

#     def dilation(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
#         '''
#         [[1, 2, 3,],
#          [4, 5, 6,],
#          [7, 8, 9,]]
#         to
#         [[1, 0, 2, 0, 3, 0,],
#          [0, 0, 0, 0, 0, 0,],
#          [4, 0, 5, 0, 6, 0,],
#          [0, 0, 0, 0, 0, 0,],
#          [7, 0, 8, 0, 9, 0,],
#          [0, 0, 0, 0, 0, 0,]]
#         with dilation=2
#         using F.convTranspose2d
#         '''
#         pass
#         kernel_size = dilation * 2 + dilation - 2
#         bs, c, h, w = x.shape
#         dilation_kernel = torch.zeros(c, c, kernel_size, kernel_size).to(x)
#         dilation_kernel[:, :, dilation - 1, dilation - 1] = torch.eye(c).to(x)
#         # dilation_kernel[0, 0, dilation_factor - 1, dilation_factor - 1] = 1 # for c=3
#         # dilation_kernel[1, 1, dilation_factor - 1, dilation_factor - 1] = 1 # for c=3
#         # dilation_kernel[2, 2, dilation_factor - 1, dilation_factor - 1] = 1 # for c=3

#         # Apply dilation
#         x = F.conv_transpose2d(x, dilation_kernel, stride=dilation, padding=dilation - 1)

#         return x

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         x = self.pad0(x)
#         x = self.pad1(x)
#         x = self.dilation(x, dilation=self.stride)
#         x = self.conv(x)

#         return x


# for debugging
if __name__ == '__main__':
    conv = ConvTranspose2dCustomBD(4, 6, kernel_size=7, stride=2)
    x = torch.randn(16, 4, 16, 8)
    torch.set_printoptions(linewidth=1000)
    print(conv.dilation(torch.randn(1, 1, 6, 6), dilation=2))
    print(f'{conv(x).shape=}')
