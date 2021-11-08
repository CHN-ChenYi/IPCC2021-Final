#!/bin/bash
#SBATCH -p amd_256
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
# mpirun ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72  12 24 24 36
mpirun --report-bindings -rf rankfile --bind-to none ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72  12 24 24 36
