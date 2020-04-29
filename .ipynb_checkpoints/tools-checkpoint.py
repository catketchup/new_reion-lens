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

class bin_smooth():
    """ Bin a powerspectrum and smooth it by interpolation """

    def __init__(self, bin_ells, ps, bin_width, statistic='mean'):
        self.bin_ells = bin_ells
        self.ps = ps
        # self.statistic = statistic
        self.bins = np.arange(np.min(self.bin_ells), np.max(self.bin_ells)+1, bin_width)

        self.binned_ps, self.bin_edges, self.binnumber = scipy.stats.binned_statistic(self.bin_ells, self.ps, statistic='mean', bins=self.bins)
        self.bin_center = (self.bin_edges[1:]+self.bin_edges[:-1])/2

    def smooth(self, ellmin, ellmax, width):

        new_ells = np.arange(ellmin, ellmax, width)
        smooth_ps = scipy.interpolate.interp1d(self.bin_center, self.binned_ps)(new_ells)

        return new_ells, smooth_ps

    
class lens_bias():
    """ Get auto powerspectrum of two reconstructed kappa maps and compare the bias """
    def __init__(self, ellmin, ellmax, nlev_t, beam_arcmin, px_arcmin, width_deg,cmb1=None, cmb2=None, inkap=None):

        self.cmb1 = cmb1
        self.cmb2 = cmb2
        self.inkap = inkap
        self.ellmin = ellmin
        self.ellmax = ellmax
        self.nlev_t = nlev_t
        self.beam_arcmin = beam_arcmin
        self.px_arcmin = px_arcmin
        self.width_deg = width_deg

        self.shape = self.cmb1.shape
        self.wcs = self.cmb2.wcs

        # npix x npix cutouts
        self.npix = int(width_deg * 60 / self.px_arcmin)
        # total number of tiles
        self.ntiles = int(np.prod(self.shape) / self.npix**2)
        # number of tiles in a row
        self.num_x = int(360 / self.width_deg)

        # Get Cl_TT in theory for symlens, it is same for each cutout
        self.theory = cosmology.default_theory()

    def auto(self, Lmin, Lmax, delta_L):
        """
        Get cutout reconstructed kappa auto-power or cross-power with input cutout kappa
        """
        # for statistics
        st = stats.Stats()
        # Initialize upper-left pixel corner
        iy, ix = 0, 0



        for itile in range(self.ntiles):
            # Get bottom-right pixel corner
            ey = iy + self.npix
            ex = ix + self.npix

            # Slice both cmb maps
            cut_cmb1 = self.cmb1[iy:ey, ix:ex]
            cut_cmb2 = self.cmb2[iy:ey, ix:ex]

            # Get geometry of the cutouts, I assume cut_cmb1 and cut_cmb2 have same geometry
            cut_shape = cut_cmb1.shape
            cut_wcs = cut_cmb1.wcs
            cut_modlmap = enmap.modlmap(cut_shape, cut_wcs)
            ells = np.arange(0, cut_modlmap.max()+1, 1)
            ctt = self.theory.lCl('TT', ells)

            # Get taper for appodization
            taper, w2 = maps.get_taper_deg(cut_shape, cut_wcs)

            # Define feed_dict for symlens
            feed_dict = {}
            feed_dict['uC_T_T'] = utils.interp(ells,
                                               ctt)(cut_modlmap)
            feed_dict['tC_T_T'] = utils.interp(ells, ctt)(
                cut_modlmap) + (self.nlev_t * np.pi / 180. /
                                60.)**2. / utils.gauss_beam(
                                    cut_modlmap, self.beam_arcmin)**2

            # Get cmb mask
            cmask = utils.mask_kspace(cut_shape,cut_wcs,lmin=self.ellmin,lmax=self.ellmax)
            # Get mask for reconstruction
            kmask = utils.mask_kspace(cut_shape,cut_wcs,lmin=Lmin,lmax=Lmax)
            # Stride across the map, horizontally first and
            # increment vertically when at the end of a row
            if (itile + 1) % self.num_x != 0:
                ix = ix + self.npix
            else:
                ix = 0
                iy = iy + self.npix

            # Apodize cutout CMB maps
            cut_cmb1 = taper * cut_cmb1
            cut_cmb2 = taper * cut_cmb2

            # Get the Fourier maps
            cut_cmb1_k = enmap.fft(cut_cmb1, normalize='phys')
            cut_cmb2_k = enmap.fft(cut_cmb2, normalize='phys')

            # Reconstruct kappa fourier maps
            cut_reckap1, noise_2d = cutout_rec(cut_shape, cut_wcs, feed_dict, cmask, kmask, cut_cmb1_k, cut_cmb1_k)
            cut_reckap2, noise_2d = cutout_rec(cut_shape, cut_wcs, feed_dict, cmask, kmask, cut_cmb2_k, cut_cmb2_k)

            # Get auto powerspectra
            center_L, cut_reckap1_x_reckap1 = powspec(cut_reckap1, cut_reckap1, taper, 4, cut_modlmap, Lmin, Lmax, delta_L)
            center_L, cut_reckap2_x_reckap2 = powspec(cut_reckap2, cut_reckap2, taper, 4, cut_modlmap, Lmin, Lmax, delta_L)

            # Get bias
            bias = (cut_reckap2_x_reckap2 - cut_reckap1_x_reckap1)/cut_reckap1_x_reckap1

            # Add to stats
            st.add_to_stats('reckap1 x reckap1', cut_reckap1_x_reckap1)
            st.add_to_stats('reckap2 x reckap2', cut_reckap2_x_reckap2)
            st.add_to_stats('bias', bias)

        # Get spectra and bias statistics
        st.get_stats()

        return center_L, st

    def cross(self, Lmin, Lmax, delta_L):
        # for statistics
        st = stats.Stats()
        # Initialize upper-left pixel corner
        iy, ix = 0, 0

        for itile in range(self.ntiles):
            # Get bottom-right pixel corner
            ey = iy + self.npix
            ex = ix + self.npix

            # Slice both cmb maps
            cut_cmb1 = self.cmb1[iy:ey, ix:ex]
            cut_cmb2 = self.cmb2[iy:ey, ix:ex]
            cut_inkap = self.inkap[iy:ey, ix:ex]

            # Get geometry of the cutouts, I assume cut_cmb1 and cut_cmb2 have same geometry
            cut_shape = cut_cmb1.shape
            cut_wcs = cut_cmb1.wcs
            cut_modlmap = enmap.modlmap(cut_shape, cut_wcs)
            ells = np.arange(0, cut_modlmap.max()+1, 1)
            ctt = self.theory.lCl('TT', ells)

            # Get taper for appodization
            taper, w2 = maps.get_taper_deg(cut_shape, cut_wcs)

            # Define feed_dict for symlens
            feed_dict = {}
            feed_dict['uC_T_T'] = utils.interp(ells,
                                               ctt)(cut_modlmap)
            feed_dict['tC_T_T'] = utils.interp(ells, ctt)(
                cut_modlmap) + (self.nlev_t * np.pi / 180. /
                                60.)**2. / utils.gauss_beam(
                                    cut_modlmap, self.beam_arcmin)**2

            # Get cmb mask
            cmask = utils.mask_kspace(cut_shape,cut_wcs,lmin=self.ellmin,lmax=self.ellmax)
            # Get mask for reconstruction
            kmask = utils.mask_kspace(cut_shape,cut_wcs,lmin=Lmin,lmax=Lmax)
            # Stride across the map, horizontally first and
            # increment vertically when at the end of a row
            if (itile + 1) % self.num_x != 0:
                ix = ix + self.npix
            else:
                ix = 0
                iy = iy + self.npix

            # Apodize cutout CMB maps
            cut_cmb1 = taper * cut_cmb1
            cut_cmb2 = taper * cut_cmb2
            cut_inkap = taper * cut_inkap

            # Get the Fourier maps
            cut_cmb1_k = enmap.fft(cut_cmb1, normalize='phys')
            cut_cmb2_k = enmap.fft(cut_cmb2, normalize='phys')

            # Reconstruct kappa fourier maps
            cut_reckap1, noise_2d = cutout_rec(cut_shape, cut_wcs, feed_dict, cmask, kmask, cut_cmb1_k, cut_cmb1_k)
            cut_reckap2, noise_2d = cutout_rec(cut_shape, cut_wcs, feed_dict, cmask, kmask, cut_cmb2_k, cut_cmb2_k)

            # Get cross powerspectra
            center_L, cut_inkap_x_reckap1 = powspec(cut_inkap, cut_reckap1, taper, 4, cut_modlmap, Lmin, Lmax, delta_L)
            center_L, cut_inkap_x_reckap2 = powspec(cut_inkap, cut_reckap2, taper, 4, cut_modlmap, Lmin, Lmax, delta_L)
            center_L, cut_inkap_x_inkap = powspec(cut_inkap, cut_inkap, taper, 2, cut_modlmap, Lmin, Lmax, delta_L)
            # Get bias
            bias = (cut_inkap_x_reckap2 - cut_inkap_x_reckap1)/cut_inkap_x_reckap1

            # Add to stats
            st.add_to_stats('inkap x inkap', cut_inkap_x_inkap)
            st.add_to_stats('inkap x reckap1', cut_inkap_x_reckap1)
            st.add_to_stats('inkap x reckap2', cut_inkap_x_reckap2)
            st.add_to_stats('bias', bias)

        # Get spectra and bias statistics
        st.get_stats()

        return center_L, st





def cutout_rec(shape, wcs, feed_dict, cmask, kmask, map1_k, map2_k):
    """ cutout lensing reconstruction """
    feed_dict['X'] = map1_k
    feed_dict['Y'] = map2_k

    # unnormalized lensing map in fourier space
    ukappa_k = s.unnormalized_quadratic_estimator(shape,
                                                  wcs,
                                                  feed_dict,
                                                  "hu_ok",
                                                  "TT",
                                                  xmask=cmask,
                                                  ymask=cmask)

    # normaliztion
    norm_k = s.A_l(shape,
                   wcs,
                   feed_dict,
                   "hu_ok",
                   "TT",
                   xmask=cmask,
                   ymask=cmask,kmask=kmask)

    # noise
    noise_2d = s.N_l_from_A_l_optimal(shape, wcs, norm_k)
    # normalized Fourier space CMB lensing convergence map
    kappa_k = norm_k * ukappa_k

    # real space CMB lensing convergence map
    kappa = enmap.ifft(kappa_k, normalize='phys')

    return kappa, noise_2d


def powspec(map1, map2, taper, taper_order, modlmap, ellmin, ellmax,
            delta_ell):
    bin_edges = np.arange(ellmin, ellmax, delta_ell)
    binner = utils.bin2D(modlmap, bin_edges)

    kmap1 = enmap.fft(map1, normalize='phys')
    kmap2 = enmap.fft(map2, normalize='phys')

    # kmap1_ave = utils.bin2D(modlmap, bin_edges)
    # kmap2_ave = utils.bin2D(modlmap, bin_edges)

    # correct power spectra
    w = np.mean(taper**taper_order)

    p2d = (kmap1 * kmap2.conj()).real / w
    # p2d = ((kmap1-kmap1_ave) * (kmap2.conj()-kmap2_ave.conj())).real / w
    # p2d = abs((kmap1 * kmap2.conj()).real / w)
    centers, p1d = binner.bin(p2d)
    return centers, p1d

def binave(map, modlmap, ellmin, ellmax, delta_ell):
    bin_edges = np.arange(ellmin, ellmax, delta_ell)
    binner = utils.bin2D(modlmap, bin_edges)

    centers, p1d = binner.bin(map)
    return centers, p1d