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
import scipy
import pandas as pd

st = stats.Stats()

x1 = [1, 2, 3, 4, 5]
x2 = [2, 4, 6, 8, 10]
x3 = [3, 6, 9, 12, 15]
st.add_to_stats('x', x1)
st.add_to_stats('x', x2)
st.add_to_stats('x', x3)
st.get_stats()
print(st.stats['x']['mean'])
print(st.stats['x']['err'])
