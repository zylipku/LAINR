import torch

from .sphere_loss import SphereLoss


class WeightedLatLonLoss(SphereLoss):

    def __init__(self, phi_theta: torch.Tensor):
        super().__init__(phi_theta)

    def get_weights(self) -> None:
        weights = torch.sin(self.theta_grid)
        weights = weights / weights.sum()
        return weights
