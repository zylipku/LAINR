# Current directory describes how to generate the dataset for the shallow-water dataset
1. `cd` to the current directory
2. Prepare for the environment
See `./requirements.conda` for the conda environment.
3. Generate the dataset (using 24 processes as an example). You may modify Line 119 in `shallow_water.py` to change the save path.
```
for s in {1..20}; do
    mpiexec -n 24 python3 shallow_water.py -s "$s"
done
```
