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

map_source = m.map_source
ksz_type = m.ksz_type
decmax = m.decmax
width_deg = m.width_deg
px_arcmin = m.px_arcmin

cutouts = m.cutouts

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
noise_map = curvedsky.rand_map(band_shape, band_wcs, Cl_noise_TT)

# Read in cmb_alm
print('reading in CMB map')
cmb_alm = hp.read_alm(map_path + 'lensed_cmb_alm.fits', hdu=1)

# Read in ksz_alm
ksz_alm = hp.read_alm(map_path + f'ksz_{ksz_type}_alm.fits')
# beamed map
cmb_t_alm = hp.smoothalm(cmb_alm[0:len(ksz_alm)]+ksz_alm, fwhm=beam_arcmin)
# Generate beamed map
beamed_t_map = curvedsky.alm2map(cmb_t_alm, enmap.empty(band_shape, band_wcs))
# cmb_t, t for total
cmb_t = beamed_t_map + noise_map


# Read in ksz_g_alm, g for gaussian
ksz_g_alm = hp.read_alm(map_path + f'ksz_{ksz_type}_g_alm_6000.fits')
# beamed map
cmb_tg_alm = hp.smoothalm(cmb_alm[0:len(ksz_g_alm)]+ksz_g_alm, fwhm=beam_arcmin)
# Generate beamed map
beamed_tg_map = curvedsky.alm2map(cmb_tg_alm, enmap.empty(band_shape, band_wcs))
# cmb_tg, t for total, g for Guassian
cmb_tg = beamed_tg_map + noise_map

# Read in input kappa map for cross correlation check
print('reading in kappa map')
kap_alm = hp.read_alm(map_path + 'kappa_alm.fits')
kap_band = curvedsky.alm2map(kap_alm, enmap.empty(band_shape, band_wcs))

ksz_cl = pd.read_csv('maps/Colin/smooth_ksz_cl.csv')['Cl']

st_tg = stats.Stats()
iy, ix = 0, 0
print('Begin to get <cl_kappa_tg_ave>')
Data_dict = {}
for itile in range(ntiles):
    # Get bottom-right pixel corner
    ex = ix + npix
    ey = iy + npix

    # Slice cmb_tg
    cut_cmb_tg = cmb_tg[iy:ey, ix:ex]
    #
    results_tg = tools.Rec(ellmin,
                           ellmax,
                           Lmin,
                           Lmax,
                           delta_L,
                           nlev_t,
                           beam_arcmin,
                           enmap1=cut_cmb_tg,
                           enmap2=cut_cmb_tg,
                           ksz_cl=ksz_cl)

    # Stride across the map, horizontally first and
    # increment vertically when at the end of a row

    if (itile + 1) % num_x != 0:
        ix = ix + npix
    else:
        ix = 0
        iy = iy + npix
    st_tg.add_to_stats('reckap_x_reckap', results_tg['reckap_x_reckap'])
    st_tg.add_to_stats('d_auto_cl', results_tg['d_auto_cl'])
    Data_dict['cutout %s g' % (itile)] = results_tg['reckap_x_reckap']
st_tg.get_stats()
cl_kappa_tg_ave = st_tg.stats['reckap_x_reckap']['mean']
cl_kappa_tg_ave_err = st_tg.stats['reckap_x_reckap']['errmean']


st_t = stats.Stats()
iy, ix = 0, 0
print('Begin to get bias for each tile')

# use a dictionary to record ps on each cutout
for itile in range(ntiles):
    # ipdb.set_trace()
    # Get bottom-right pixel corner
    ex = ix + npix
    ey = iy + npix

    # Slice cmb_t
    cut_cmb_t = cmb_t[iy:ey, ix:ex]
    # Slice input kappa
    cut_inkap = kap_band[iy:ey, ix:ex]
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
                          ksz_cl=ksz_cl)
    # Get bias
    bias = (results_t['reckap_x_reckap'] - cl_kappa_tg_ave) / inkap_x_inkap
    # bias = (results_t['reckap_x_reckap'] - cl_kappa_tg_ave) / (results_t['reckap_x_reckap'] - results_t['noise_cl'])
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
    st_t.add_to_stats('ureckap_x_ureckap', results_t['ureckap_x_ureckap'])
    st_t.add_to_stats('norm_x_norm', results_t['norm_x_norm'])
    st_t.add_to_stats('bias', bias)
    st_t.add_to_stats('inkap_x_reckap', inkap_x_reckap)
    Data_dict['cutout %s' % (itile)] = results_t['reckap_x_reckap']
    print('tile %s completed, %s tiles in total' % (itile + 1, ntiles))
st_t.get_stats()

# Store bias in a dictionary

Data_dict['Ls'] = results_t['Ls']
Data_dict['cl_kappa_tg_ave'] = cl_kappa_tg_ave
Data_dict['cl_kappa_tg_ave_err'] = cl_kappa_tg_ave_err
Data_dict['reckap_x_reckap'] = st_t.stats['reckap_x_reckap']['mean']
Data_dict['reckap_x_reckap_err'] = st_t.stats['reckap_x_reckap']['errmean']
Data_dict['ureckap_x_ureckap'] = st_t.stats['ureckap_x_ureckap']['mean']
Data_dict['ureckap_x_ureckap_err'] = st_t.stats['ureckap_x_ureckap']['errmean']
Data_dict['norm_x_norm'] = st_t.stats['norm_x_norm']['mean']
Data_dict['norm_x_norm_err'] = st_t.stats['norm_x_norm']['errmean']
Data_dict['bias'] = st_t.stats['bias']['mean']
Data_dict['bias_err'] = st_t.stats['bias']['errmean']
Data_dict['d_auto_cl'] = st_tg.stats['d_auto_cl']['mean']
Data_dict['norm'] = results_t['norm']
Data_dict['noise'] = results_t['noise_cl']
Data_dict['inkap_x_inkap'] = st_t.stats['inkap_x_inkap']['mean']
Data_dict['inkap_x_inkap_err'] = st_t.stats['inkap_x_inkap']['errmean']
Data_dict['inkap_x_reckap'] = st_t.stats['inkap_x_reckap']['mean']
Data_dict['inkap_x_reckap_err'] = st_t.stats['inkap_x_reckap']['errmean']

Data_df = pd.DataFrame(Data_dict)
Data_df.to_csv(data_path + map_source + '_' + ksz_type + '_%s_%s_%s.csv' %
               (experiment, ellmin, ellmax),
               index=False)
