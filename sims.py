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

# Simulate bias of lensing reconstruction from non-Gaussian kSZ

# Use maps provided by websky
map_path = 'maps/websky/'
# Path of output data
data_path = 'temp/'

# lmin, lmax for cmb maps
ellmin = 100
# lmin, lmax for reconstructed kappa map
ellmaxs = [3000, 4000]
# bin width for reconstructed kappa powerspectrum
delta_L = 40

# noise level for temperature and polarization maps, here we only investiget temperature maps
nlev_ts = [1, 5, 10]
nlev_p = np.sqrt(2) * nlev_t
# beam size
beam_arcmin = 10
# pixel size in arcmin
px_arcmin = 1.
# size of cutout square
width_deg = 30

# Let's define a cut-sky cylindrical geometry with 1 arcminute pixel width
# and a maximum declination extent of +- 15 deg (see below for reason)
# band width in deg
decmax = 30
# shape and wcs  of the band
band_shape, band_wcs = enmap.band_geometry(dec_cut=np.deg2rad(decmax),
                                           res=np.deg2rad(px_arcmin / 60.))

band_modlmap = enmap.modlmap(band_shape, band_wcs)

# Read in cmb_alm
cmb_alm = hp.read_alm(map_path + 'lensed_alm.fits', hdu=1)
# Get cmb band map
cmb_band = curvedsky.alm2map(cmb_alm, enmap.empty(band_shape, band_wcs))

# Read in ksz_alm and get ksz band map
ksz_alm = hp.read_alm(map_path + 'ksz_alm_lmax_6000.fits')
ksz_band = curvedsky.alm2map(ksz_alm, enmap.empty(band_shape, band_wcs))

# Read in ksz_g_alm  and get ksz_g_band map, 'g' is for gaussian
ksz_g_alm = hp.read_alm(map_path + 'ksz_g_alm_lmax_6000.fits')
ksz_g_band = curvedsky.alm2map(ksz_g_alm, enmap.empty(band_shape, band_wcs))

ells = np.arange(0, ellmax, 1)
# noise power spectrum
Cl_noise_TT = (nlev_t * np.pi / 180. / 60.)**2 * np.ones(ells.shape)
# deconvolved noise power spectrum
Cl_noise_TT = Cl_noise_TT / utils.gauss_beam(ells, beam_arcmin)**2
# deconvolved noise band map
noise_band = curvedsky.rand_map(band_shape, band_wcs, Cl_noise_TT)

# Add CMB map, ksz map and deconvolved noise map
cmb_wksz_band = cmb_band + ksz_band + noise_band

# Add CMB map, ksz gaussian map and deconvolved noise map
cmb_wksz_g_band = cmb_band + ksz_g_band + noise_band

# Read in input kappa map for cross correlation check
kap_alm = hp.read_alm(map_path + 'kap_alm_lmax_6000.fits')
kap_band = curvedsky.alm2map(kap_alm, enmap.empty(band_shape, band_wcs))

# Calculate the lensing reconstruction auto bias of the two cases above
Bias = tools.lens_bias(ellmin,
                       ellmax,
                       nlev_t,
                       beam_arcmin,
                       px_arcmin,
                       width_deg,
                       cmb1=cmb_wksz_g_band,
                       cmb2=cmb_wksz_band,
                       inkap=kap_band)

Ls, Auto = Bias.auto(Lmin, Lmax, delta_L)

# loop for different ellmax and nlev_t
for ellmax in ellmaxs():
    for nlev_t in nlev_ts():
        # Calculate the lensing reconstruction auto bias of the two cases above
        Bias = tools.lens_bias(ellmin,
                               ellmax,
                               nlev_t,
                               beam_arcmin,
                               px_arcmin,
                               width_deg,
                               cmb1=cmb_wksz_g_band,
                               cmb2=cmb_wksz_band,
                               inkap=kap_band)

        Ls, Auto = Bias.auto(Lmin, Lmax, delta_L)

        # Store autospectra and their bias in a dictionary
        Auto_dict = {
            "L": Ls,
            "auto_wksz_g": Auto.stats['reckap1 x reckap1']['mean'],
            "auto_wksz_g_err": Auto.stats['reckap1 x reckap1']['err'],
            "auto_wksz": Auto.stats['reckap2 x reckap2']['mean'],
            "auto_wksz_err": Auto.stats['reckap2 x reckap2']['err'],
            "auto_bias": Auto.stats['bias']['mean'],
            "auto_bias_err": Auto.stats['bias']['err']
        }

        # Convert the data in DataFrame and save it in .csv files
        Auto_df = pd.DataFrame(Auto_dict)
        Auto_df.to_csv(data_path + 'auto_lmin=%s_lmax=%s_nlev_t=%s.csv' %
                       (ellmin, ellmax, nlev_t),
                       index=False)

        # Calculate the lensing reconstruction cross powerspectrum to check
        Ls, Cross = Bias.cross(Lmin, Lmax, delta_L)

        # Store crossspectra and their bias in a dictionary
        Cross_dict = {
            "L": Ls,
            "input_kappa": Cross.stats['inkap x inkap']['mean'],
            "cross_wksz_g": Cross.stats['inkap x reckap1']['mean'],
            "cross_wksz_g_err": Cross.stats['inkap x reckap1']['err'],
            "cross_wksz": Cross.stats['inkap x reckap2']['mean'],
            "cross_wksz_err": Cross.stats['inkap x reckap2']['err'],
            "cross_bias": Cross.stats['bias']['mean'],
            "cross_bias_err": Cross.stats['bias']['err']
        }

        # Convert the data in DataFrame and save it in .csv files
        Cross_df = pd.DataFrame(Cross_dict)
        Cross_df.to_csv(data_path + 'cross_lmin=%s_lmax=%s_nlev_t=%s.csv' %
                        (ellmin, ellmax, nlev_t),
                        index=False)
