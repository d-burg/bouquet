#!/bin/bash
#SBATCH --job-name=bouquet_2000ms
#SBATCH --array=0-31
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
# compute-node environment (adjust for your cluster), e.g.:
# module load conda && conda activate bouquet
# export OFT_PYTHONPATH=/path/to/OpenFUSIONToolkit/build_release/python
# NOTE threads_per_worker > 1: BLAS/OpenMP reductions are
# no longer bit-reproducible run-to-run, and DLSODE has
# been observed to hang under heavy oversubscription --
# keep cpus-per-task == threads_per_worker (see
# bouquet.parallel._warn_multithreaded).
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m bouquet.parallel shard bouquet_2000ms_bundle.json $SLURM_ARRAY_TASK_ID
