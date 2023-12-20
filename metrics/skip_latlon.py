import numpy as np
import torch

from .sphere_loss import SphereLoss


class SkipLatLonLoss(SphereLoss):

    def __init__(self, phi_theta: torch.Tensor, base_jump=0):
        super().__init__(phi_theta)

        self.base_jump = base_jump

    def get_weights(self) -> None:

        mask_list = []
        for theta in self.thetas:

            ratio = torch.sin(theta)
            if ratio > 2 / 5:  # fully observed
                mask = torch.ones(self.nlons)
            else:
                mask = torch.zeros(self.nlons)
                try:
                    skip = 2**(int(np.ceil(np.log(ratio) / np.log(2 / 5))) - 1 + self.base_jump)
                    mask[::skip] = 1
                except:  # ratio=0
                    pass
            mask_list.append(mask)

        weights = torch.stack(mask_list, dim=-1)

        return weights / weights.sum()
