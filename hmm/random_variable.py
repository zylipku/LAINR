import torch


class RandomVariable:

    def __init__(self, **kwargs) -> None:

        for key, value in kwargs.items():
            setattr(self, key, value)

    def sample(self, bs: int) -> torch.Tensor:
        '''
        sample from the random variable with batch size `bs`

        Args:
            bs (int): batch size

        Returns:
            torch.Tensor: shape=(bs, out_shape)
        '''
        raise NotImplementedError


class Gaussian(RandomVariable):

    mean: torch.Tensor
    sigma: float

    def __init__(self,
                 mean: torch.Tensor,
                 sigma: float) -> None:
        super().__init__(mean=mean,
                         sigma=sigma)

    def sample(self, bs: int) -> torch.Tensor:

        return self.mean + self.sigma * torch.randn(bs, *self.mean.shape)
