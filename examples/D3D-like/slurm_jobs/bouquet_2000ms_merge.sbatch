#!/bin/bash
#SBATCH --job-name=bouquet_2000ms_merge
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
python -m bouquet.parallel merge bouquet_2000ms_bundle.json
