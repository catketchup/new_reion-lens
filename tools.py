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

# def cutout_rec(shape, wcs, feed_dict, cmask, kmask, map1_k, map2_k):

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

def Rec(ellmin,
          ellmax,
          Lmin,
          Lmax,
          delta_L,
          nlev,
          beam_arcmin,
          enmap1=None,
          enmap2=None,
          noise=False,
        ksz_cls=None,
        deconvolve=True):
    """ Reconstruct a reckap map or its Fourier map """
    map_shape = enmap1.shape
    map_wcs = enmap1.wcs
    map_modlmap = enmap.modlmap(map_shape, map_wcs)
    ells = np.arange(0, map_modlmap.max() + 1, 1)
    theory = cosmology.default_theory()

    if ksz_cls is None:
        cltt = theory.lCl('TT', ells)
    else:
        cltt = theory.lCl('TT', ells) + ksz_cls[ells]
    kbeam = utils.gauss_beam(map_modlmap, beam_arcmin)
    # deconvolved 2d noise
    n2d = (nlev*np.pi/180./60.)**2./kbeam**2.


    taper, w2 = maps.get_taper_deg(map_shape, map_wcs)

    feed_dict = {}

    feed_dict['uC_T_T'] = utils.interp(ells, cltt)(map_modlmap)
    feed_dict['tC_T_T'] = feed_dict['uC_T_T'] + n2d


    cmask = utils.mask_kspace(map_shape, map_wcs, lmin=ellmin, lmax=ellmax)
    kmask = utils.mask_kspace(map_shape, map_wcs, lmin=Lmin, lmax=Lmax)

    enmap1 = taper * enmap1
    enmap2 = taper * enmap2

    enmap1_k = enmap.fft(enmap1, normalize='phys')
    enmap2_k = enmap.fft(enmap2, normalize='phys')

    if deconvolve == True:
        feed_dict['X'] = enmap1_k/kbeam
        feed_dict['Y'] = enmap2_k/kbeam
    else:
        feed_dict['X'] = enmap1_k
        feed_dict['Y'] = enmap2_k

    # unnormalized lensing map in fourier space
    ukappa_k = s.unnormalized_quadratic_estimator(map_shape,
                                                  map_wcs,
                                                  feed_dict,
                                                  "hu_ok",
                                                  'TT',
                                                  xmask=cmask,
                                                  ymask=cmask)
    # normalization Fourier map
    norm_k = s.A_l(map_shape,
                   map_wcs,
                   feed_dict,
                   "hu_ok",
                   'TT',
                   xmask=cmask,
                   ymask=cmask,
                   kmask=kmask)

    results = {}

    # normalized Fourier space CMB lensing convergence map(reckap)
    reckap_k = norm_k * ukappa_k
    reckap = enmap.ifft(reckap_k, normalize='phys')

    # reckap power spectrum
    Ls, reckap_x_reckap = powspec_k(reckap_k, taper=taper, taper_order=4, modlmap=map_modlmap, lmin=Lmin, lmax=Lmax, delta_l=delta_L)

    Ls, ureckap_x_ureckap = powspec_k(ukappa_k, taper=taper, taper_order=4, modlmap=map_modlmap, lmin=Lmin, lmax=Lmax, delta_l=delta_L)

    Ls, norm_x_norm = powspec_k(norm_k, taper=taper, taper_order=4, modlmap=map_modlmap, lmin=Lmin, lmax=Lmax, delta_l=delta_L)

    # deflection field k map1_k
    d_k = reckap_k/(1/2*map_modlmap)
    # deflection field power spectrum
    d_auto_cl = reckap_x_reckap/(1/4*Ls**2)
    # phi power spectrum
    phi_auto_cl = reckap_x_reckap/(1/4*Ls**4)
    # norm
    Ls, norm = binave(norm_k, map_modlmap, Lmin, Lmax, delta_L)
    # noise
    Ls, noise_cl = binave(
        s.N_l_from_A_l_optimal(map_shape, map_wcs, norm_k),
        map_modlmap,
        Lmin,
        Lmax,
        delta_L,
    )
    results = {'Ls':Ls, 'reckap':reckap, 'reckap_k':reckap_k, 'reckap_x_reckap':reckap_x_reckap, 'ureckap_x_ureckap':ureckap_x_ureckap, 'norm_x_norm':norm_x_norm, 'd_auto_cl':d_auto_cl, 'phi_auto_cl':phi_auto_cl, 'norm':norm, 'noise_cl':noise_cl}
    return results


def powspec_k(enmap1_k, enmap2_k=None, taper=None, taper_order=None, modlmap=None, lmin=None, lmax=None, delta_l=None):

    bin_edges = np.arange(lmin, lmax+delta_l+1, delta_l)
    binner = utils.bin2D(modlmap, bin_edges)

    # correct power spectra
    w = np.mean(taper**taper_order)

    if enmap2_k != None:
        p2d = (enmap1_k * enmap2_k.conj()).real / w
    else:
        p2d = (enmap1_k * enmap1_k.conj()).real / w

    centers, p1d = binner.bin(p2d)
    return centers, p1d

def powspec(enmap1, enmap2=None, taper_order=2, lmin=None, lmax=None, delta_l=None):

    shape = enmap1.shape
    wcs = enmap1.wcs
    modlmap = enmap.modlmap(shape, wcs)
    weights = (2*modlmap + 1)

    bin_edges = np.arange(lmin, lmax +delta_l+1, delta_l)
    binner = utils.bin2D(modlmap, bin_edges)
    taper, w2 = maps.get_taper_deg(shape, wcs)
    # w is for correction of powerspectrum
    w = np.mean(taper**taper_order)

    enmap1 = taper*enmap1
    enmap1_k = enmap.fft(enmap1, normalize='phys')

    if enmap2 is not None:
        enmap2 = taper*enmap2
        enmap2_k = enmap.fft(enmap2, normalize='phys')
        p2d = (enmap1_k * enmap2_k.conj()).real / w
    else:
        enmap2_k = None
        p2d = (enmap1_k * enmap1_k.conj()).real / w

    centers, p1d = binner.bin(p2d)
    return centers, p1d


def binave(map, modlmap, ellmin, ellmax, delta_l):
    bin_edges = np.arange(ellmin, ellmax+delta_l+1, delta_l)
    binner = utils.bin2D(modlmap, bin_edges)

    centers, p1d = binner.bin(map)
    return centers, p1d


# class rec():
#     def __init__(self,
#                  ellmin,
#                  ellmax,
#                  Lmin,
#                  Lmax,
#                  delta,
#                  nlev,
#                  beam_arcmin,
#                  enmap1=None,
#                  enmap2=None):

#         self.ellmin = ellmin
#         self.ellmax = ellmax
#         self.Lmin = Lmin
#         self.Lmax = Lmax
#         self.delta = delta
#         self.nlev = nlev
#         self.beam_arcmin = beam_arcmin
#         self.enmap1 = enmap1
#         self.enmap2 = enmap2

#     def reckap(self):
#         return reckap(self.ellmin, self.ellmax, self.Lmin, self.Lmax,
#                      self.delta_L, self.nlev, self.beam_arcmin, self.enmap1,
#                      self.enmap2)

#     def phi_auto_cl(self):
#         return 0
