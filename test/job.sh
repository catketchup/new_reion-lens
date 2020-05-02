#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=3
#SBATCH --time=3

#module load python
source ~/.bashrc.ext
srun -n 4 -c 2 python test.py
