import torch


class SphereLoss:

    def __init__(self, phi_theta: torch.Tensor = None, root=False):
        '''
        initialized with latlon_mesh

        #phi: [0, 2pi)
        #theta: (pi, 0)

        Args:
            phi_theta (torch.Tensor): 2d tensor of shape (nlons, nlats, 2)
        '''

        self.nlons, self.nlats, _ = phi_theta.shape
        self.phi_grid = phi_theta[..., 0]
        self.theta_grid = phi_theta[..., 1]
        self.thetas = self.theta_grid[0]
        self.phis = self.phi_grid[:, 0]

        self.root = root

        self._weights = None

    def get_weights(self) -> None:
        raise NotImplementedError

    @property
    def weights(self) -> torch.Tensor:
        if self._weights is None:
            self._weights = self.get_weights()
        return self._weights

    def __call__(self, x: torch.Tensor, y: torch.Tensor,
                 start_dim=-3, feature_sep=False,
                 l1: float = None, l2: float = 1.) -> torch.Tensor:
        '''
        compute loss for x and y

        Args:
            x (torch.Tensor): shape: (..., lat_dim, lon_dim, ...)
            y (torch.Tensor): shape: (..., lat_dim, lon_dim, ...)
            start_dim (int, optional): dim index for lat_dim. Defaults to 2.
            root (bool, optional): whether to take the root of the loss. Defaults to False.

        Returns:
            torch.Tensor: loss = mean(weights * (x - y)**2)
            or rooted: loss = mean((weights * (x - y)**2)**0.5)
        '''
        diff1 = torch.abs(x - y)
        diff2 = (x - y) ** 2

        weights = self.weights.to(diff2)
        slices = [None,] * diff2.ndim
        slices[start_dim] = slice(None)
        slices[start_dim + 1] = slice(None)
        weights_unsqueezed = weights[tuple(slices)]
        weighted_diff1 = diff1 * weights_unsqueezed
        weighted_sum1 = torch.sum(weighted_diff1, dim=(start_dim, start_dim + 1), keepdim=False)
        weighted_diff2 = diff2 * weights_unsqueezed
        weighted_sum2 = torch.sum(weighted_diff2, dim=(start_dim, start_dim + 1), keepdim=False)

        if self.root:
            weighted_sum2 = torch.sqrt(weighted_sum2)

        if l1 is None:
            weighted_sum = l2 * weighted_sum2
        else:
            weighted_sum = l1 * weighted_sum1 + l2 * weighted_sum2

        if feature_sep:
            loss = torch.mean(weighted_sum.view(-1, weighted_sum.shape[-1]), dim=0)
        else:
            loss = torch.mean(weighted_sum)

        return loss
