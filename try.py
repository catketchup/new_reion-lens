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


class rec():
    def __init__(self,
                 ellmin,
                 ellmax,
                 Lmin,
                 Lmax,
                 delta,
                 nlev,
                 beam_arcmin,
                 enmap1=None,
                 enmap2=None):

        self.ellmin = ellmin
        self.ellmax = ellmax
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.delta = delta
        self.nlev = nlev
        self.beam_arcmin = beam_arcmin
        self.enmap1 = enmap1
        self.enmap2 = enmap2

    def kappa(self):
        return kappa(self.ellmin, self.ellmax, self.Lmin, self.Lmax,
                     self.delta_L, self.nlev, self.beam_arcmin, self.enmap1,
                     self.enmap2)

    def phi_cl(self):
        return 0


def kappa(ellmin,
          ellmax,
          Lmin,
          Lmax,
          delta_L,
          nlev,
          beam_arcmin,
          enmap1=None,
          enmap2=None,
          noise=False):
    """ Reconstruct a kappa map or its Fourier map """
    shape = enmap1.shape
    wcs = enmap1.wcs
    modlmap = enmap1.modlmap
    ells = np.arange(0, modlmap.max() + 1, 1)
    ctt = theory.lCl('TT', ells)
    taper, w2 = maps.get_taper_deg(shape, wcs)

    feed_dict = {}
    feed_dict['uC_T_T'] = utils.interp(ells, ctt)(modlmap)
    feed_dict['tC_T_T'] = utils.interp(
        ells,
        ctt)(modlmap) + (nlev * np.pi / 180. / 60.)**2. / utils.gauss_beam(
            modlmap, beam_arcmin)**2

    cmask = utils.mask_kspace(shape, wcs, lmin=ellmin, lmax=ellmax)
    kmask = utils.mask_kspace(shape, wcs, lmin=Lmin, lmax=Lmax)

    enmap1 = taper * enmap1
    if enmap2 == None:
        enmap2 = enmap1
    else:
        enmap2 = taper * enmap2

    enmap1_k = enmap.fft(enmap1, normalize='phys')
    if enmap2 == None:
        enmap2_k = enmap1_k
    else:
        enmap2_k = enmap.fft(enmap2, normalize='phys')

    feed_dict['X'] = enmap1_k
    feed_dict['Y'] = enmap2_k

    # unnormalized lensing map in fourier space
    ukappa_k = s.unnormalized_quadratic_estimator(shape,
                                                  wcs,
                                                  feed_dict,
                                                  "hu_ok",
                                                  'TT',
                                                  xmask=cmask,
                                                  ymask=cmask)
    # normalization Fourier map
    norm_k = s.A_l(shape,
                   wcs,
                   feed_dict,
                   "hu_ok",
                   'TT',
                   xmask=cmask,
                   ymask=cmask,
                   kmask=kmask)

    results = {}

    # normalized Fourier space CMB lensing convergence map(kappa)
    kappa_k = norm_k * ukappa_k
    kappa = enmap.ifft(kappa_k, normalize='phys')

    # kappa power spectrum
    Ls, kappa_cl = powspec(kappa, kappa, taper, 4, modlmap, Lmin, Lmax,
                           delta_L)

    # norm
    # noise
    Ls, noise_cl = binave(
        s.N_l_from_A_l_optimal(shape, wcs, norm_k),
        modlmap,
        Lmin,
        Lmax,
        delta_L,
    )
    results{'Ls':Ls, 'kappa':kappa, 'kappa_cl':kappa_cl, 'noise_cl':noise_cl}
    return results


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

    map = enmap.ifft(kmap, normalize='phys')
    centers, p1d = binner.bin(map)
    return centers, p1d