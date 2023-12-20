import torch


class MSELoss:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, x: torch.Tensor, y: torch.Tensor, start_dim=2, root=False) -> torch.Tensor:
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
        diff2 = (x - y) ** 2
        weighted_sum = torch.mean(diff2, dim=(start_dim, start_dim + 1), keepdim=False)

        if root:
            weighted_sum = torch.sqrt(weighted_sum)

        loss = torch.mean(weighted_sum)

        return loss
