import torch

from .sphere_loss import SphereLoss


class WeightedLatLonLoss(SphereLoss):

    def __init__(self,
                 phi_theta: torch.Tensor, mask: torch.Tensor = None,
                 **kwargs):
        super().__init__(phi_theta, **kwargs)

        self.mask = mask

    def get_weights(self) -> None:
        weights = torch.sin(self.theta_grid)
        if self.mask is not None:
            weights = weights * self.mask.to(weights)
        weights = weights / weights.sum()
        return weights
