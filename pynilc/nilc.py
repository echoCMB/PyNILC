import os
from rich.progress import track
import numpy as np
import dask.array as da
import healpy as hp
from pynilc.needlets import NeedletTransform
from pynilc.sht import HealpixDUCC
from rich.console import Console
console = Console()



class NILC:

    def __init__(self, freqs, fwhms, common_fwhm, nthreads='auto', use_healpy=False):
        self.freqs = freqs
        self.fwhms = fwhms
        self.common_fwhm = np.radians(common_fwhm/60)
        match nthreads:
            case 'auto':
                self.nthreads = os.cpu_count()
            case _:
                self.nthreads = nthreads
        self.use_healpy = use_healpy
        console.print(f'Number of threads: {self.nthreads}', style='bold blue')
        console.print(f'Using healpy: {self.use_healpy}', style='bold red')


    def __call__(self, maps, needlet_bands):
        self.nside = hp.npix2nside(maps[0].shape[-1])
        maps = da.stack(maps, axis=0, allow_unknown_chunksizes=True)
        self.needlet_bands = needlet_bands

        alms = self.maps2alms(maps)
        alms = self.apply_common_fwhm(alms)
        return alms


    
    def maps2alms(self, maps):
        nside = hp.npix2nside(maps.shape[-1])
        lmax = 3 * nside - 1
        hpducc = HealpixDUCC(nside)
        alms = []
        for m in track(range(maps.shape[0]), description='Computing spherical harmonics',total=maps.shape[0]):
            if self.use_healpy:
                alms.append(hp.map2alm(m))
            else:
                alms.append(hpducc.map2alm(maps[m].compute(), lmax=lmax, nthreads=self.nthreads))
        alms = da.stack(alms, axis=0, allow_unknown_chunksizes=True)
        return alms
    
    def apply_common_fwhm(self, alms):
        alms_common_fwhm = []
        blc = hp.gauss_beam(self.common_fwhm, lmax=3*self.nside-1, pol=True).T
        for a in track(range(alms.shape[0]), description='Applying common FWHM', total=alms.shape[0]):
            blf = hp.gauss_beam(np.radians(self.fwhms[a]/60), lmax=3*self.nside-1, pol=True).T
            br = blf/blc
            alm = alms[a].compute()
            hp.almxfl(alm[0], br[0], inplace=True)
            hp.almxfl(alm[1], br[1], inplace=True)
            hp.almxfl(alm[2], br[2], inplace=True)
            alms_common_fwhm.append(alm)
        alms_common_fwhm = da.stack(alms_common_fwhm, axis=0, allow_unknown_chunksizes=True)
        return alms_common_fwhm
    
    def alms2needlets(self, alms):
        needlettrans = []
        for a in track(range(alms.shape[0]), description='Computing needlets', total=alms.shape[0]):
            needlettrans_a = []
            alm = alms[a].compute()
            nbands = self.needlet_bands.shape[0]
            for b in range(nbands):
                nlmax = np.max(np.where(self.needlet_bands[b, :] > 0)) 
                nside = int(2 ** np.ceil(np.log2(nlmax / 2)))
                hp.almxfl(alm[0], self.needlet_bands[b, :nlmax + 1], inplace=True)
                hp.almxfl(alm[1], self.needlet_bands[b, :nlmax + 1], inplace=True)
                hp.almxfl(alm[2], self.needlet_bands[b, :nlmax + 1], inplace=True)
                if self.use_healpy:
                    needlet = hp.alm2map(alm, nside)
                else:
                    hpducc = HealpixDUCC(nside)
                    needlet = hpducc.alm2map(alm, lmax=nlmax, nthreads=self.nthreads)
                needlettrans_a.append(needlet)
            needlettrans.append(np.array(needlettrans_a))
        return np.array(needlettrans)


        