
import torch
from torch import nn


class SmoothLeaky(nn.Module):

    def __init__(self, negative_slope=0.2) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        # self.log_sigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor):

        scale = 1. - (1. - self.negative_slope) / (1. + torch.exp(x))
        return x * scale

        return self.negative_slope * x - (1 - self.negative_slope) * self.log_sigmoid(-x)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * nn.functional.softplus(self.beta))).div_(1.1)


class Act(nn.Module):

    def __init__(self, act_name='relu') -> None:
        super().__init__()

        if act_name == 'relu':
            self.act = nn.ReLU()

        elif act_name == 'elu':
            self.act = nn.ELU()

        elif act_name == 'tanh':
            self.act = nn.Tanh()

        elif act_name == 'sigmoid':
            self.act = nn.Sigmoid()

        elif act_name == 'sin':
            self.act = torch.sin

        elif act_name == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=0.2)

        elif act_name == 'smooth_leaky':
            self.act = SmoothLeaky()

        elif act_name == 'swish':
            self.act = Swish()

        elif act_name == 'identity':
            self.act = nn.Identity()

        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.act(x)
