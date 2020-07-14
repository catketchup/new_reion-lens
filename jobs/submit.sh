#!/bin/bash -l

#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --license=SCRATCH

# export RUNDIR=$SCRATCH/run-$SLURM_JOBID
# mkdir -p $RUNDIR
# cd $RUNDIR

# cd /global/homes/h/hongbo/test
cd /global/cscratch1/sd/hongbo/new_reion-lens
#srun -n 4 bash -c 'echo "Hello, world, from node $(hostname)"'
#srun -n 2 bash -c 'echo "Morning, world, from node $(hostname)"'
#srun -n 4 python ./py_test.py
#srun -n 4 python -c "import socket; print(socket.gethostname())"

# srun -n 1 python ./my_func.py -b 1 -e 2
srun -n 1 python new_sims.py --experiment 'CMB_S4' --nlev_t 1 --beam_arcmin 3 --ellmin 30 --ellmax 4000 --delta_L 150
