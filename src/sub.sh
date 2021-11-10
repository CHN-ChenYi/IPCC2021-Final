#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 2
#SBATCH -n 128
# mpirun ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72  24 12 3 9
# mpirun ./main 0.005  ../data/ipcc_gauge_32_64  32 32 32 64  32 32 16 1
mpirun ./main 0.005  ../data/ipcc_gauge_48_96  48 48 48 96  48 24 3 24
