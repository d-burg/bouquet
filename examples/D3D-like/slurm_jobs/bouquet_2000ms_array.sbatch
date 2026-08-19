#!/bin/bash
#SBATCH --job-name=bouquet_2000ms
#SBATCH --array=0-31
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m bouquet.parallel shard bouquet_2000ms_bundle.json $SLURM_ARRAY_TASK_ID
