#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 2
#SBATCH -n 128
# mpirun ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72  6 12 12 9
# mpirun ./main 0.005  ../data/ipcc_gauge_32_64  32 32 32  64  8 16 16 8
mpirun ./main 0.005  ../data/ipcc_gauge_48_96  48 48 48 96  12 24 24 12
