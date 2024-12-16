from ducc0.sht.experimental import synthesis, adjoint_synthesis
import ducc0
import numpy as np



class HealpixDUCC:
    """
    A class to handle HEALPix geometry and perform spherical harmonic transforms.
    Attributes:
    ----------
    nside : int
        The HEALPix nside parameter.
    """
    def __init__(self, nside):
        base = ducc0.healpix.Healpix_Base(nside, "RING")
        geom = base.sht_info()
        area = (4 * np.pi) / (12 * nside ** 2)
        w = np.full((geom['theta'].size, ), area)
        phi0 = geom['phi0']
        nphi = geom['nphi']
        ringstart = geom['ringstart']
        theta = geom['theta']
        
        for arr in [phi0, nphi, ringstart, w]:
            assert arr.size == theta.size

        argsort = np.argsort(ringstart) # We sort here the rings by order in the maps
        self.theta = theta[argsort].astype(np.float64)
        self.weight = w[argsort].astype(np.float64)
        self.phi0 = phi0[argsort].astype(np.float64)
        self.nph = nphi[argsort].astype(np.uint64)
        self.ofs = ringstart[argsort].astype(np.uint64)
    
    def getsize(self, lmax, mmax):
        """
        Calculate the size of a data structure based on the given lmax and mmax values.

        Args:
            lmax (int): The maximum degree.
            mmax (int): The maximum order.

        Returns:
            int: The calculated size of the data structure.
        """
        return ((mmax+1) * (mmax+2)) // 2 + (mmax+1) * (lmax-mmax)


    def __tomap__(self, alm: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, map:np.ndarray=None, **kwargs):
        alm = np.atleast_2d(alm)
        return synthesis(alm=alm, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, spin=spin, phi0=self.phi0,
                         nthreads=nthreads, ringstart=self.ofs, map=map, **kwargs)

    def __toalm__(self, m: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, alm=None, apply_weights=True, **kwargs):
        m = np.atleast_2d(m)
        if apply_weights:
            for of, w, npi in zip(self.ofs, self.weight, self.nph):
                m[:, of:of + npi] *= w
        if alm is not None:
            assert alm.shape[-1] == self.getsize(lmax, mmax)
        return adjoint_synthesis(map=m, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, spin=spin, phi0=self.phi0,
                                 nthreads=nthreads, ringstart=self.ofs, alm=alm,  **kwargs)

    def __alm2map_spin2__(self, alm:np.ndarray, lmax:int, nthreads:int, **kwargs):
        return self.__tomap__(alm, 2, lmax, lmax, nthreads, **kwargs)

    def __map2alm_spin2__(self, m:np.ndarray, lmax:int, nthreads:int, **kwargs):
        return self.__toalm__(m.copy(), 2, lmax, lmax, nthreads, **kwargs)

    def __alm2map_spin0__(self, alm:np.ndarray, lmax:int,  nthreads:int, **kwargs):
        return self.__tomap__(alm, 0, lmax, lmax, nthreads, **kwargs).squeeze()

    def __map2alm_spin0__(self, m:np.ndarray, lmax:int, nthreads:int, **kwargs):
        return self.__toalm__(m.copy(), 0, lmax, lmax, nthreads, **kwargs).squeeze()
    
    def alm2map(self, alm:np.ndarray, lmax:int,nthreads:int, **kwargs):
        if len(alm) > 3:
            return self.__alm2map_spin0__(alm, lmax, nthreads, **kwargs)
        elif len(alm) == 2:
            return self.__alm2map_spin2__(alm, lmax, nthreads, **kwargs)
        elif len(alm) == 3:
            return np.concatenate([self.__alm2map_spin0__(alm[0], lmax, nthreads, **kwargs).reshape(1, -1),
                                   self.__alm2map_spin2__(alm[1:], lmax, nthreads, **kwargs)], axis=0)
        else:
            raise ValueError("Invalid alm shape")
    
    def map2alm(self, m:np.ndarray, lmax:int, nthreads:int, **kwargs):
        if len(m) > 3:
            return self.__map2alm_spin0__(m, lmax, nthreads, **kwargs)
        elif len(m) == 2:
            return self.__map2alm_spin2__(m, lmax, nthreads, **kwargs)
        elif len(m) == 3:
            return np.concatenate([self.__map2alm_spin0__(m[0], lmax, nthreads, **kwargs).reshape(1, -1),
                                   self.__map2alm_spin2__(m[1:], lmax, nthreads, **kwargs)], axis=0)
        else:
            raise ValueError("Invalid map shape")