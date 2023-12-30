import os

import numpy as np

import torch

from metrics import get_metrics
from datasets import MetaData

root_path = '/home/lizhuoyuan/datasets/shallow_water'

metadata_tr_path = os.path.join(root_path, 'cached_meta_tr.pt')
metadata_va_path = os.path.join(root_path, 'cached_meta_va.pt')

metadata_tr: MetaData = torch.load(metadata_tr_path)
metadata_va: MetaData = torch.load(metadata_va_path)

trajs_tr = metadata_tr.trajs  # (ntrajs, Nsteps, *state_size, state_channels)
trajs_va = metadata_va.trajs  # (ntrajs, Nsteps, *state_size, state_channels)

ntrajs_tr, Nsteps_tr, *state_size_tr, state_channels_tr = trajs_tr.shape
ntrajs_va, Nsteps_va, *state_size_va, state_channels_va = trajs_va.shape

X_tr = trajs_tr.reshape(trajs_tr.shape[0] * trajs_tr.shape[1], -1)  # (N, n)
X_va = trajs_va.reshape(trajs_va.shape[0] * trajs_va.shape[1], -1)  # (N, n)

X_tr = np.float64(X_tr.numpy())  # (N, n)
X_va = np.float64(X_va.numpy())  # (N, n)

res = np.linalg.svd(X_tr.T, full_matrices=False)  # (n, n)
print(res.U.shape)  # (16384, 3840) = (128x64x2, 16x240)
print(res.S.shape)  # (3840,)
print(res.Vh.shape)  # (3840, 3840)

X_va_emb = np.matmul(X_va, res.U[:, :1024])  # (N, n) x (n, 1024) = (N, 1024)
X_va_rec = np.matmul(X_va_emb, res.U[:, :1024].T)  # (N, 1024) x (1024, n) = (N, n)

loss_fn = get_metrics(name='weighted', phi_theta=metadata_va.coords['coord_latlon'])
print(loss_fn(trajs_va, X_va_rec.reshape(trajs_va.shape)))

from matplotlib import pyplot as plt

plt.plot(res.S)
plt.yscale('log')
plt.show()
plt.savefig('svd.png')
