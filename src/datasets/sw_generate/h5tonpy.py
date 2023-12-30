import os
import h5py
import numpy as np

raw_path = '/home/lizhuoyuan/datasets/shallow_water/raw'

for seedno in range(1, 21):

    print(f'{seedno=}')

    f = h5py.File(os.path.join(raw_path, f'seedno={seedno}/seedno={seedno}_s1.h5'), mode='r')
    height = f['tasks/height'][:, :, :]
    vorticity = f['tasks/vorticity'][:, :, :]
    phi = f['tasks/vorticity'].dims[1][0][:].ravel()[:]
    theta = f['tasks/vorticity'].dims[2][0][:].ravel()[:]

    h5file = h5py.File(os.path.join(raw_path, f'traj_{seedno}.h5'), 'w')
    h5file.create_dataset('height', data=height)
    h5file.create_dataset('vorticity', data=vorticity)
    h5file.create_dataset('phi', data=phi)
    h5file.create_dataset('theta', data=theta)
    h5file.close()
