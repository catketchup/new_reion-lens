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
import new_tools

# Simulate bias of lensing reconstruction from non-Gaussian kSZ

# map source, 'Colin' or 'websky'
map_source = 'websky'
# 'lt' for late-time kSZ or 'ri' for reionization kSZ
ksz_type = 'ri'

# experiment configuration, name:[nlev_t,beam_arcmin]
# experiments = {'reference':[0,0]}
experiments = {'Planck_SMICA': [45, 5], 'CMB_S3': [7, 1.4], 'CMB_S4': [1, 3]}

# Use maps provided by websky
map_path = 'maps/' + map_source + '/'
# Path of output data
data_path = 'data/'

# lmin, lmax for cmb maps
ellmin = 30
# ellmaxs = [4000]
# ellmaxs = [3000, 4000, 4500]
ellmaxs = 4000
# bin width for reconstructed kappa powerspectrum
delta_L = 200

# pixel size in arcmin
px_arcmin = 1.
# size of cutout square
width_deg = 30

print('bin_width=%s' % (delta_L))
# Let's define a cut-sky cylindrical geometry with 1 arcminute pixel width
# and a maximum declination extent of +- 45 deg (see below for reason)
# band width in deg
decmax = 15
# shape and wcs  of the band
band_shape, band_wcs = enmap.band_geometry(dec_cut=np.deg2rad(decmax),
                                           res=np.deg2rad(px_arcmin / 60.))

band_modlmap = enmap.modlmap(band_shape, band_wcs)

# Read in cmb_alm
print('reading in CMB map')
cmb_alm = hp.read_alm(map_path + 'lensed_cmb_alm.fits', hdu=1)
# Get cmb band map
cmb_band = curvedsky.alm2map(cmb_alm, enmap.empty(band_shape, band_wcs))

# Read in ksz_alm and get ksz band map
print('reading in %s %s kSZ map' % (map_source, ksz_type))
ksz_alm = hp.read_alm(map_path + f'ksz_{ksz_type}_alm.fits')
ksz_band = curvedsky.alm2map(ksz_alm, enmap.empty(band_shape, band_wcs))

# Read in ksz_g_alm  and get ksz_g_band map, 'g' is for gaussian
print('reading in ksz_g map')
ksz_g_alm = hp.read_alm(map_path + f'ksz_{ksz_type}_g_alm_6000.fits')
ksz_g_band = curvedsky.alm2map(ksz_g_alm, enmap.empty(band_shape, band_wcs))

# Read in input kappa map for cross correlation check
print('reading in kappa map')
kap_alm = hp.read_alm(map_path + 'kappa_alm.fits')
kap_band = curvedsky.alm2map(kap_alm, enmap.empty(band_shape, band_wcs))

npix = int(width_deg * 60 / px_arcmin)
ntiles = int(np.prod(shape) / npix**2)
num_x = int(360 / width_deg)

# Getting <CL_kk_tg>
print('Getting <CL_KK_tg>')
for experiment_name, value in experiments.items():
    for ellmax in ellmaxs:
        print('%s, ellmax=%s' % (experiment_name, ellmax))
        nlev_t = value[0]
        beam_arcmin = value[1]
        ells = np.arange(0, ellmax, 1)
        # lmin, lmax for reconstructed kappa map
        Lmin, Lmax = 40, ellmax
        # noise power spectrum
        Cl_noise_TT = (nlev_t * np.pi / 180. / 60.)**2 * np.ones(ells.shape)
        # deconvolved noise power spectrum
        Cl_noise_TT = Cl_noise_TT / utils.gauss_beam(ells, beam_arcmin)**2
        # deconvolved noise band map
        noise_band = curvedsky.rand_map(band_shape, band_wcs, Cl_noise_TT)

        # cmb_tg
        cmb_tg = cmb_band + ksz_g_band + noise_band
        st_tg = stats.Stats()
        iy, ix = 0, 0
        print('Begin to get <cl_kappa_tg_ave>')
        for itile in range(ntiles):
            # Get bottom-right pixel corner
            ex = ix + npix
            ey = iy + npix

            # Slice cmb_tg
            cut_cmb_tg = cmb_tg[iy:ey, ix:ex]
            #
            results_tg = new_tools.rec(ellmin, ellmax, Lmin, Lmax, delta_L,
                                       nlev_t, beam_arcmin, cut_cmb_tg,
                                       cut_cmb_tg)
            st_tg.add_to_stats('reckap_x_reckap',
                               results_tg['reckap_x_reckap'])

        st_tg.get_stats()
        cl_kappa_tg_ave = st_tg.stats['reckap_x_reckap']['mean']

        # cmb_t
        cmb_t = cmb_band + ksz_band + noise_band
        st_t = stats.Stats()
        iy, ix = 0, 0
        print('Begin to get bias for each tile')
        for itile in range(ntiles):
            # Get bottom-right pixel corner
            ex = ix + npix
            ey = iy + npix

            # Slice cmb_t
            cut_cmb_t = cmb_t[iy:ey, ix:ex]
            #
            results_t = new_tools.rec(ellmin, ellmax, Lmin, Lmax, delta_L,
                                      nlev_t, beam_arcmin, cut_cmb_t,
                                      cut_cmb_t)
            bias = (results_t['reckap_x_reckap'] -
                    cl_kappa_tg_ave) / results_t['reckap_x_reckap']

            st_t.add_to_stats('bias', bias)
            print('tile %s completed, %s tiles in total' %
                  (itile + 1, self.ntiles))
        st_t.get_stats()

        # Store bias in a dictionary
        Data_dict = {
            'Ls': results_t['Ls']['mean'],
            'bias': results_t['bias']['mean'],
            'bias_err': results_t['bias']['err']
        }

        Data_df = pd.DataFrame(Data_dict)
        Data_df.to_csv(data_path + map_source + '_' + ksz_type,
                       '_%s_%s_%s.csv' % (experiment_name, ellmin, ellmax),
                       index=False)
