from typing import Callable, Tuple
from functools import partial, reduce
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .utils import MLP, Act


class AutoEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, latent_dim: int,
                 encoder_hidden_dims: list,
                 decoder_hidden_dims: list) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims

        self._encoder = nn.Sequential(
            MLP([in_dim] + encoder_hidden_dims + [latent_dim], 'smooth_leaky'),
            # nn.Tanh()
        )
        self._decoder = nn.Sequential(
            MLP([latent_dim] + decoder_hidden_dims + [out_dim], 'smooth_leaky')
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder(x)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        rec_x = self.decode(z)
        return rec_x


class Conv2dBlock(nn.Module):

    '''
    padding_type: 'circular', 'reflect' or 'replicate'
    '''

    VAILD_PADDING_TYPES = ['circular', 'reflect', 'replicate']

    def __init__(self,
                 in_channels: int, out_channels: int,
                 act_name='swish', kernel_size=3,
                 padding_type: str | Tuple[str, str] = 'periodic'
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_name = act_name
        self.act = Act(act_name)

        if isinstance(padding_type, str):
            padding_type = (padding_type, padding_type)

        assert all(p in self.VAILD_PADDING_TYPES for p in padding_type)

        self.kernel_size = kernel_size
        self.pad_width = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=0)

        self.pad0 = partial(F.pad, pad=(0, 0, self.pad_width, self.pad_width), mode=padding_type[0])
        self.pad1 = partial(F.pad, pad=(self.pad_width, self.pad_width, 0, 0), mode=padding_type[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pad0(x)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.pad0(x)
        x = self.pad1(x)
        x = self.conv2(x)
        x = self.act(x)

        return x


class AutoEncoder2d(nn.Module):

    def __init__(self,
                 state_dims: Tuple[int, int],
                 state_channels: int,
                 latent_channels: int, kernel_size: int,
                 encoder_hidden_channels: list,
                 decoder_hidden_channels: list = None,
                 padding_type: str | Tuple[str, str] = 'periodic') -> None:
        super().__init__()

        self.state_dims = state_dims
        self.state_channels = state_channels
        self.latent_channels = latent_channels

        self.kernel_size = kernel_size

        self.encoder_hidden_channels = encoder_hidden_channels
        if decoder_hidden_channels is None:
            decoder_hidden_channels = encoder_hidden_channels[::-1]
        self.decoder_hidden_channels = decoder_hidden_channels

        self.encoder_depth = len(encoder_hidden_channels) + 1
        self.decoder_depth = len(decoder_hidden_channels) + 1

        self.poolingclass = nn.AvgPool2d

        # define encoder layers
        encoder_named_layers = []
        encoder_named_layers.append(
            (
                f'encoder_0',
                Conv2dBlock(state_channels,
                            encoder_hidden_channels[0],
                            kernel_size=kernel_size,
                            padding_type=padding_type)
            )
        )
        encoder_named_layers.append(
            (
                f'maxpool_0',
                self.poolingclass(kernel_size=2, stride=2)
            )
        )
        for k in range(self.encoder_depth - 2):
            encoder_named_layers.append(
                (
                    f'encoder_{k+1}',
                    Conv2dBlock(encoder_hidden_channels[k],
                                encoder_hidden_channels[k + 1],
                                kernel_size=kernel_size,
                                padding_type=padding_type)
                )
            )
            encoder_named_layers.append(
                (
                    f'maxpool_{k+1}',
                    self.poolingclass(kernel_size=2, stride=2)
                )
            )
        encoder_named_layers.append(
            (
                f'encoder_{self.encoder_depth-1}',
                Conv2dBlock(encoder_hidden_channels[-1],
                            latent_channels,
                            kernel_size=kernel_size,
                            padding_type=padding_type)
            )
        )

        # define decoder layers
        decoder_named_layers = []
        decoder_named_layers.append(
            (

                f'decoder_0',
                Conv2dBlock(latent_channels,
                            decoder_hidden_channels[0],
                            kernel_size=kernel_size,
                            padding_type=padding_type)
            )
        )
        for k in range(self.decoder_depth - 2):
            decoder_named_layers.append(
                (
                    f'upsample_{k+1}',
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
            )
            decoder_named_layers.append(
                (
                    f'decoder_{k+1}',
                    Conv2dBlock(decoder_hidden_channels[k],
                                decoder_hidden_channels[k + 1],
                                kernel_size=kernel_size,
                                padding_type=padding_type)
                )
            )
        decoder_named_layers.append(
            (
                f'upsample_{self.decoder_depth-1}',
                nn.Upsample(scale_factor=2, mode='nearest')
            )
        )
        decoder_named_layers.append(
            (

                f'decoder_{self.decoder_depth-1}',
                Conv2dBlock(decoder_hidden_channels[-1],
                            state_channels,
                            kernel_size=kernel_size,
                            padding_type=padding_type)
            )
        )

        self.encoders = nn.Sequential(OrderedDict(encoder_named_layers))
        self.decoders = nn.Sequential(OrderedDict(decoder_named_layers))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoders(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoders(x)
        return x

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        rec_x = self.decode(z)
        return rec_x
