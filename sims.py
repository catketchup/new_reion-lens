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
data_path = 'data/'
