from orphics import io, maps, lensing, cosmology, stats
from pixell import enmap, curvedsky
import numpy as np
import os, sys
import healpy as hp
import matplotlib.pylab as plt
import symlens as s
from symlens import utils
import importlib
from mpi4py import MPI
import pandas as pd
import tools
import ipdb
import argparse
import params as m
# Simulate bias of lensing reconstruction from non-Gaussian kSZ
# experiments configurations

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='experiment name')
parser.add_argument('--nlev_t',
                    type=float,
                    help='noise level of temperature field')
parser.add_argument('--beam_arcmin', type=float, help='beam_arcmin')
parser.add_argument('--ellmin', type=int, help='ellmin of CMB')
parser.add_argument('--ellmax', type=int, help='ellmax of CMB')
parser.add_argument('--delta_L', type=int, help='delta_L of Kappa')

args = parser.parse_args()
experiment = args.experiment
nlev_t = args.nlev_t
beam_arcmin = args.beam_arcmin
ellmin = args.ellmin
ellmax = args.ellmax
delta_L = args.delta_L
Lmin, Lmax = ellmin, ellmax

ksz_g_realizations = m.ksz_g_realizations
map_source = m.map_source
ksz_type = m.ksz_type
decmax = m.decmax
width_deg = m.width_deg
px_arcmin = m.px_arcmin

cutouts = m.cutouts

print('cutouts_num:', cutouts)
print('%s' %(experiment))
print('ellmin = %s, ellmax = %s, delta_L = %s' %(ellmin, ellmax, delta_L))

# Use maps provided by websky or Colin
map_path = m.map_path
# Path of output data
data_path = m.data_path

# print('bin_width=%s' % (delta_L))
# Let's define a cut-sky cylindrical geometry with 1 arcminute pixel width
# and a maximum declination extent of +- 45 deg (see below for reason)
# band width in deg

# shape and wcs  of the band
band_shape, band_wcs = enmap.band_geometry(dec_cut=np.deg2rad(decmax),
                                           res=np.deg2rad(px_arcmin / 60.))
band_modlmap = enmap.modlmap(band_shape, band_wcs)

npix = int(width_deg * 60 / px_arcmin)
ntiles = int(np.prod(band_shape) / npix**2)
num_x = int(360 / width_deg)

ells = np.arange(0, ellmax+1, 1)


# noise power spectrum
Cl_noise_TT = (nlev_t * np.pi / 180. / 60.)**2 * np.ones(ells.shape)
# noise band map
noise = curvedsky.rand_map(band_shape, band_wcs, Cl_noise_TT)

# Read in cmb_alms
print('reading in CMB map')
cmb_alms_file = m.cmb_alms_file
cmb_alms = hp.read_alm(cmb_alms_file)

# Read in ksz_alms
ksz_alms_file = m.ksz_alms_file
ksz_alms = hp.read_alm(ksz_alms_file)




cmb = curvedsky.alm2map(cmb_alms, enmap.empty(band_shape, band_wcs))

use_cmb = m.use_cmb

# generate ksz and ksz_g, g for Gaussian
ksz = curvedsky.alm2map(ksz_alms, enmap.empty(band_shape, band_wcs))

# cmb_t, t for total
if use_cmb:
    cmb_t = cmb + ksz
else:
    cmb_t = ksz

# Read in input kappa map for cross correlation check
print('reading in kappa map')
kap_alms_file = m.kap_alms_file
kap_alms = hp.read_alm(kap_alms_file)
kap = curvedsky.alm2map(kap_alms, enmap.empty(band_shape, band_wcs))

# kap_cls_file = m.kap_cls_file
# kap_cls = pd.read_csv(kap_cls_file)['cls']

# Read in ksz_cls
smooth_ksz_cls_file = m.smooth_ksz_cls_file
smooth_ksz_cls = pd.read_csv(smooth_ksz_cls_file)['cls']

## Non-Gaussian parts
st_tg = stats.Stats()
print('Begin to get reckap_x_reckap_tg_ave')
Data_dict = {}
for r in range(ksz_g_realizations):
    ksz_g_alms = hp.synalm(smooth_ksz_cls)
    ksz_g = curvedsky.alm2map(ksz_g_alms, enmap.empty(band_shape, band_wcs))
    noise = curvedsky.rand_map(band_shape, band_wcs, Cl_noise_TT)

    if use_cmb:
        cmb_tg = cmb + ksz_g
    else:
        cmb_tg = ksz_g

    print('ksz_g realization:', r+1)

    iy, ix = 0, 0
    for itile in range(ntiles):
        # Get bottom-right pixel corner
        ex = ix + npix
        ey = iy + npix

        # Slice cmb_tg
        cut_cmb_tg = cmb_tg[iy:ey, ix:ex]
        # Reconstruction
        results_tg = tools.Rec(ellmin,
                               ellmax,
                               Lmin,
                               Lmax,
                               delta_L,
                               nlev_t,
                               beam_arcmin,
                               enmap1=cut_cmb_tg,
                               enmap2=cut_cmb_tg,
                               ksz_cls=smooth_ksz_cls)

        # Stride across the map, horizontally first and
        # increment vertically when at the end of a row

        if (itile + 1) % num_x != 0:
            ix = ix + npix
        else:
            ix = 0
            iy = iy + npix
        st_tg.add_to_stats('reckap_x_reckap', results_tg['reckap_x_reckap'])

st_tg.get_stats()


## Gaussian parts
st_t = stats.Stats()
iy, ix = 0, 0
print('Begin to get bias for each tile')

for itile in range(ntiles):
    # ipdb.set_trace()
    # Get bottom-right pixel corner
    ex = ix + npix
    ey = iy + npix

    # Slice cmb_t
    cut_cmb_t = cmb_t[iy:ey, ix:ex]
    # Slice input kappa
    cut_inkap = kap[iy:ey, ix:ex]
    # Get inkap_x_inkap
    inkap_x_inkap = tools.powspec(cut_inkap,
                                  lmin=Lmin,
                                  lmax=Lmax,
                                  delta_l=delta_L)[1]

    # Get reconstruction results
    results_t = tools.Rec(ellmin,
                          ellmax,
                          Lmin,
                          Lmax,
                          delta_L,
                          nlev_t,
                          beam_arcmin,
                          enmap1=cut_cmb_t,
                          enmap2=cut_cmb_t,
                          ksz_cls=smooth_ksz_cls)

    # Get inkap_x_reckap
    inkap_x_reckap = tools.powspec(cut_inkap,
                                   enmap2=results_t['reckap'],
                                   taper_order=4,
                                   lmin=Lmin,
                                   lmax=Lmax,
                                   delta_l=delta_L)[1]
    # Stride across the map, horizontally first and
    # increment vertically when at the end of a row
    if (itile + 1) % num_x != 0:
        ix = ix + npix
    else:
        ix = 0
        iy = iy + npix

    st_t.add_to_stats('inkap_x_inkap', inkap_x_inkap)
    st_t.add_to_stats('reckap_x_reckap', results_t['reckap_x_reckap'])
    st_t.add_to_stats('inkap_x_reckap', inkap_x_reckap)
    print('tile %s completed, %s tiles in total' % (itile + 1, ntiles))

st_t.get_stats()

# correct bias factor
if use_cmb:
    factor = st_t.stats['inkap_x_inkap']['mean']/st_t.stats['inkap_x_reckap']['mean']
else:
    factor = 1


# Get bias
bias = factor**2*(st_t.stats['reckap_x_reckap']['mean'] - st_tg.stats['reckap_x_reckap']['mean']) /st_t.stats['inkap_x_inkap']['mean']

st_t.add_to_stats('bias', bias)

# Get bias' error
bias_err = factor**4*(st_t.stats['reckap_x_reckap']['errmean']**2 + st_tg.stats['reckap_x_reckap']['errmean']**2)/(st_t.stats['inkap_x_inkap']['mean'])**2

# Store data in a dictionary
Data_dict['Ls'] = results_t['Ls']
Data_dict['reckap_x_reckap'] = st_t.stats['reckap_x_reckap']['mean']
Data_dict['reckap_x_reckap_err'] = st_t.stats['reckap_x_reckap']['errmean']

Data_dict['bias'] = bias
Data_dict['bias_err'] = bias_err
Data_dict['inkap_x_inkap'] = st_t.stats['inkap_x_inkap']['mean']
Data_dict['inkap_x_reckap'] = st_t.stats['inkap_x_reckap']['mean']


Data_df = pd.DataFrame(Data_dict)

Data_df.to_csv(data_path + map_source + '_' + ksz_type + '_%s_%s_%s.csv' %
               (experiment, ellmin, ellmax),
               index=False)
