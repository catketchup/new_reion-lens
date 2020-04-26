from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class bin_smooth():
    """ Bin a powerspectrum and smooth it by interpolation """

    def __init__(self, bin_ells, ps, bin_width, statistic='mean'):
        self.bin_ells = bin_ells
        self.ps = ps
        # self.statistic = statistic
        self.bins = np.arange(np.min(self.bin_ells), np.max(self.bin_ells), bin_width)

        self.binned_ps, self.bin_edges, self.binnumber = stats.binned_statistic(self.bin_ells, self.ps, statistic='mean', bins=self.bins)
        self.bin_center = (self.bin_edges[1:]+self.bin_edges[:-1])/2

    def smooth(self, ellmin, ellmax, width):

        new_ells = np.arange(ellmin, ellmax+1, width)
        smooth_ps = interpolate.interp1d(self.bin_center, self.binned_ps)(new_ells)

        return new_ells, smooth_ps
