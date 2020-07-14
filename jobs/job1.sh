#!/bin/bash -l

#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --license=SCRATCH

cd /global/cscratch1/sd/hongbo/new_reion-lens

srun -n 1 python new_sims.py --experiment 'CMB_S4' --nlev_t 1 --beam_arcmin 3 --ellmin 30 --ellmax 3000 --delta_L 150

srun -n 1 python new_sims.py --experiment 'CMB_S4' --nlev_t 1 --beam_arcmin 3 --ellmin 30 --ellmax 4000 --delta_L 200
