#!/bin/env python

import sys, os, time
import params as m

# output directory
outdir = '/global/cscratch1/sd/hongbo/new_reion-lens/'
if not(os.path.isdir):os.makedirs(outdir)

f = open('%s/submit_jobs.sh' %(outdir), 'w')
f.write('#!/bin/bash\n')
f.write('#SBATCH -N 1\n')
f.write('#SBATCH -t %s\n' %(m.runtime))
f.write( '#SBATCH --ntasks-per-node=12\n')
f.write('#SBATCH --license=SCRATCH\n')

for experiment_name, values  in m.experiments.items():
    for groups, moment in m.moments.items():
        nlev_t = values['nlev_t']
        beam_arcmin = values['beam_arcmin']
        ellmin = moment['ellmin']
        ellmax = moment['ellmax']
        delta_L = moment['delta_L']
        f.write('OMP_NUM_THREADS=3 python new_sims.py --experiment \'%s\' --nlev_t %s --beam_arcmin %s --ellmin %s --ellmax %s --delta_L %s & sleep 1\n' %(experiment_name, nlev_t, beam_arcmin, ellmin, ellmax, delta_L))


f.write('wait\n')
f.close()
os.system("sbatch -C $CRAY_CPU_TARGET %s/submit_jobs.sh\n" % (outdir))
