from typing import List

import torch
from torch import nn


class AbstractINR(nn.Module):

    filters: List[nn.Module]  # of length n_hidden_linears-1

    def __init__(self,
                 in_dim=2,
                 out_dim=1,
                 hidden_dim=32,
                 latent_dim=64,
                 n_hidden_linears=3,
                 ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_hidden_linears = n_hidden_linears

        self.linears = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim)
                                      for _ in range(self.n_hidden_linears)])

        self.modulations = nn.ModuleList([nn.Linear(self.latent_dim, self.hidden_dim)
                                          for _ in range(self.n_hidden_linears)])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''in_dim -> hidden_dim (first layer)'''
        raise NotImplementedError

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        '''hidden_dim -> out_dim (last layer)'''
        raise NotImplementedError

    def forward(self, x_coords: torch.Tensor, latent_code: torch.Tensor) -> torch.Tensor:

        y = self.encode(x_coords)

        for layer_idx in range(self.n_hidden_linears - 1):
            y = self.linears[layer_idx](y)  # linear transformation (mlp)
            y = y + self.modulations[layer_idx](latent_code)  # modulation
            y = y * self.filters[layer_idx](x_coords)  # filter

        y = self.linears[-1](y)
        y = y + self.modulations[-1](latent_code)

        y = self.decode(y)

        return y


class FourierINR(AbstractINR):

    class FourierLayer(nn.Module):

        def __init__(self, in_dim, out_dim, weight_scale=1.) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.linear.weight.data *= weight_scale
            self.linear.bias.data.uniform_(-torch.pi, torch.pi)

        def forward(self, x: torch.Tensor):
            return torch.sin(self.linear(x))

    def __init__(self, in_dim=2, out_dim=1, hidden_dim=32, latent_dim=64, n_hidden_linears=3) -> None:
        super().__init__(in_dim, out_dim, hidden_dim, latent_dim, n_hidden_linears)

        self.encoder = self.FourierLayer(self.in_dim, self.hidden_dim)
        self.filters = nn.ModuleList([self.FourierLayer(self.in_dim, self.hidden_dim)
                                      for _ in range(self.n_hidden_linears - 1)])
        self.decoder = nn.Linear(self.hidden_dim, self.out_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
