#!/bin/bash
OMP_NUM_THREADS=3 python new_sims.py --experiment 'Planck_SMICA' --nlev_t 45 --beam_arcmin 5 --ellmin 30 --ellmax 3000 --delta_L 150 & sleep 1
OMP_NUM_THREADS=3 python new_sims.py --experiment 'Planck_SMICA' --nlev_t 45 --beam_arcmin 5 --ellmin 30 --ellmax 4000 --delta_L 200 & sleep 1
OMP_NUM_THREADS=3 python new_sims.py --experiment 'CMB_S3' --nlev_t 7 --beam_arcmin 1.4 --ellmin 30 --ellmax 3000 --delta_L 150 & sleep 1
OMP_NUM_THREADS=3 python new_sims.py --experiment 'CMB_S3' --nlev_t 7 --beam_arcmin 1.4 --ellmin 30 --ellmax 4000 --delta_L 200 & sleep 1
OMP_NUM_THREADS=3 python new_sims.py --experiment 'CMB_S4' --nlev_t 1 --beam_arcmin 3 --ellmin 30 --ellmax 3000 --delta_L 150 & sleep 1
OMP_NUM_THREADS=3 python new_sims.py --experiment 'CMB_S4' --nlev_t 1 --beam_arcmin 3 --ellmin 30 --ellmax 4000 --delta_L 200 & sleep 1
wait
