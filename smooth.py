from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class bin_smooth():
    """ Bin a powerspectrum and smooth it by interpolation """

    def __init__(self, ells, ps, statistic='mean', bins=bins):
        self.ells = ells
        self.ps = ps
        self.statistic = statistic
        self.bins = bins

        self.binned_ps, self.bin_edges, self.binnumber = stats.binned_statistic(self.ells, self.ps, self.statistic, self.bins)
        self.bin_center = (self.bin_edges[1:]+bin_edges[:-1])/2

    def smooth(self, ellmin, ellmax, width):
        new_ells = np.arange(ellmin, ellmax, width)
