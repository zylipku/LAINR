import torch

from modules import AEflowV2 as AEflow
x = torch.randn(3, 2, 128, 64)
model = AEflow(state_channels=2, kernel_size=3)
y = model(x)
print(f'{y.shape=}')
torch.nn.functional.conv_transpose2d()
