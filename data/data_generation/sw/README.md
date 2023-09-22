Current folder provides data generation for shallow water datasets.

# environment
- python 3.10.10
- dedalus 3.0.0a0
- h5py 3.8.0
- numpy 1.24.2
- scipy 1.10.1
- xarray 2023.4.1

# usage
1. run the following:
```
for s in {1..20}; do
    mpiexec -n 24 python3 shallow_water.py -s "$s"
done
```
where `s` is the seed for the random number generator.

2. Then transform the `.h5py` files to regular `.npy` files
```
python h5tonpy.py
```