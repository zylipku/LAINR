import h5py
import numpy as np

for seedno in range(0, 21):

    f = h5py.File(f'seedno={seedno}/seedno={seedno}_s1.h5', mode='r')
    height = f['tasks/height'][:, ::2, ::2]
    vorticity = f['tasks/vorticity'][:, ::2, ::2]
    phi = f['tasks/vorticity'].dims[1][0][:].ravel()[::2]
    theta = f['tasks/vorticity'].dims[2][0][:].ravel()[::2]

    np.save(f'../../sw/traj_{seedno}_height.npy', height)
    np.save(f'../../sw/traj_{seedno}_vorticity.npy', vorticity)
    np.save(f'../../sw/traj_{seedno}_phi.npy', phi)
    np.save(f'../../sw/traj_{seedno}_theta.npy', theta)

for seedno in range(19, 21):

    f = h5py.File(f'seedno={seedno}/seedno={seedno}_s1.h5', mode='r')
    height = f['tasks/height'][:, 1::2, 1::2]
    vorticity = f['tasks/vorticity'][:, 1::2, 1::2]
    phi = f['tasks/vorticity'].dims[1][0][:].ravel()[1::2]
    theta = f['tasks/vorticity'].dims[2][0][:].ravel()[1::2]

    np.save(f'../../sw/traj_{seedno}_height_offgrid.npy', height)
    np.save(f'../../sw/traj_{seedno}_vorticity_offgrid.npy', vorticity)
    np.save(f'../../sw/traj_{seedno}_phi_offgrid.npy', phi)
    np.save(f'../../sw/traj_{seedno}_theta_offgrid.npy', theta)
