#!/bin/bash

cur_dir="/home/lizhuoyuan/MyProject/LAINR-simplified/src/plot/offgrid"
cd "$cur_dir"
drts=("rand_nobs=10" "rand_nobs=20" "rand_nobs=50" "rand_ratio=0.01" "rand_ratio=0.02")
dest_dir="/home/lizhuoyuan/MyProject/LAINR-paper-revised/figs/"

for dir in "${drts[@]}"; do

    echo "$dir"

    if [ -d "$dir" ]; then
        cd "$dir"
        echo "cd $dir"
        for file in mask*; do
            if [ -f "$file" ]; then
                cp "$file" "$cur_dir/${dir}_$file"
            fi
        done
        for file in SINR*; do
            if [ -f "$file" ]; then
                cp "$file" "$cur_dir/${dir}_$file"
            fi
        done
        cd ..
    fi
done