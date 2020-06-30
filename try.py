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
import configparser
import json

# Simulate bias of lensing reconstruction from non-Gaussian kSZ

# Read in parameters from parameters.ini
config = configparser.ConfigParser()
config.read('parameters.ini')

# maps information
map_source = config['maps'].get('map_source')
ksz_type = config['maps'].get('ksz_type')
decmax = config['maps'].getint('decmax')
width_deg = config['maps'].getint('width_deg')

# experiments configurations
experiments = config['experiments'].get('experiments')
# pixel size in arcmin
px_arcmin = config['experiments'].getfloat('px_arcmin')

# CMB
ellmin = config['CMB'].getint('ellmin')
ellmaxs = json.loads(config['CMB'].get('ellmaxs'))

# Kappa
delta_L = config['Kappa'].getint('delta_L')



print('bin_width=%s' % (delta_L))
# Let's define a cut-sky cylindrical geometry with 1 arcminute pixel width
# and a maximum declination extent of +- 45 deg (see below for reason)
# band width in deg


# Use maps provided by websky
map_path = 'maps/' + map_source + '/'
# Path of output data
data_path = 'output/'
# shape and wcs  of the band
print(decmax)
print(ellmaxs)
band_shape, band_wcs = enmap.band_geometry(dec_cut=np.deg2rad(45),
                                            res=np.deg2rad(px_arcmin / 60.))
print(ellmaxs[0])
