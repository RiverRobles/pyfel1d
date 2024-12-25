import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numexpr as ne
from joblib import Parallel, delayed
import mpmath 

class ParticleLoader:
    """
    This is a class for handling loading of 1D particle distributions with appropriate shot noise for free-electron laser simulations.

    Attributes:
        Np (int): number of macroparticles in one slice 
        M (int): 2M is the number of macroparticles in one beamlet 
        bdes (float): desired rms bunching in one slice 
        enforce_bunching (bool): if True, enforce uniform bunching very close to bdes across all slices 
        rel_bdes_tol (float): relative difference allowed of the bunching in a slice from bdes 
        Nzeta (int): number of slices 
        chi (1D float array, length=Nzeta): represents the current profile of the beam to adjust bdes according to the local density. Peak should be 1
        eta_sample_function (function with one argument): a function which returns a set of sampled energies given input Np
    """
    def __init__(self, Np=2048, M=4, Nzeta=100, enforce_bunching=False, rel_bdes_tol=1e-1, slice_espread_function=lambda Np: np.zeros(Np)):
        """
        Constructor for the ParticleLoader class. Defines the attributes, if chi is not present it is set to 1 for all slices 
        """
        self.Np = Np
        self.M = M
        
        self.enforce_bunching = enforce_bunching 
        self.rel_bdes_tol = rel_bdes_tol

        self.slice_espread_function = slice_espread_function
        
        self.Nzeta = Nzeta
        
    def load_slice_thetas(self, bdes):
        '''
        Function to load theta coordinates for a single slice 

        Parameters:
            chi_val (float): value of chi at this slice 

        Returns:
            thetas (1D float array, length Np): array of theta coordinates for this slice
        '''
        
        nb = self.Np / (2*self.M)
        assert nb.is_integer(), 'Np must be divisible by 2M'
        nb = int(nb)
        
        theta0 = np.linspace(0, 2*np.pi, nb, endpoint=False)
        thetas = (theta0 + np.arange(2*self.M)[:,None]*np.pi/self.M)
        
        if bdes > 0:
            N = 1/bdes**2
            Nb = N / nb
        
            ms = np.arange(1, self.M+1)
            sigmas = np.sqrt(2 / Nb)/ms
            ab_vals = np.vstack([np.random.normal(0, sigma, 2*nb) for sigma in sigmas])
            a, b = ab_vals[:,:nb], ab_vals[:,nb:]
    
            thetas += np.sum(a[:,None,:]*np.cos(ms[:,None,None]*thetas) + b[:,None,:]*np.sin(ms[:,None,None]*thetas), axis=0)
            
        return (thetas % (2*np.pi)).flatten()

    def load_slice_etas(self):
        nb = self.Np / (2*self.M)
        assert nb.is_integer(), 'Np must be divisible by 2M'
        nb = int(nb)
        
        beamlet_energies = self.slice_espread_function(nb)
        return np.vstack([beamlet_energies for i in range(2*self.M)]).flatten()
    
    def load_all_thetas(self, bdes):
        """
        Function to load theta coordinates for all slices

        Returns:
            thetas (2D float array, size Nzeta x Np): theta coordinates for all slices 
        """
        if isinstance(bdes, (float, complex)):
            bdes_arr = np.ones(self.Nzeta)*bdes
        elif isinstance(bdes, (np.ndarray, list)):
            bdes_arr = bdes
        else:
            raise ValueError("Invalid input for bdes parameter")
        
        if not self.enforce_bunching:
            return np.vstack([self.load_slice_thetas(bdes_arr[i]) for i in range(self.Nzeta)])
        else:
            if isinstance(bdes, (float, complex)):
                thetas = self.load_slice_thetas(bdes)
                while np.abs(self.compute_bunching(1, thetas) - bdes) >= bdes*self.rel_bdes_tol:
                    thetas = self.load_slice_thetas(bdes)
                return np.vstack([thetas for i in range(self.Nzeta)])
            else:
                coords = []
                for i in range(self.Nzeta):
                    bdes = bdes_arr[i]
                    thetas = self.load_slice_thetas(bdes)
                    while (np.abs(self.compute_bunching(1, thetas) - bdes) >= bdes*self.rel_bdes_tol) & (bdes != 0.0):
                        thetas = self.load_slice_thetas(bdes)
                    coords.append(thetas)
                return np.vstack(coords)
                
    def load_all_etas(self, chirp=0.0):
        """ 
        Function to load eta coordinates for all slices 

        Returns:
            etas (2D float array, size Nzeta x Np): eta coordinates for all slices 
        """
        if isinstance(chirp, (float, int)):
            chirp_arr = np.ones(self.Nzeta)*chirp
        elif isinstance(chirp, (np.ndarray, list)):
            chirp_arr = chirp
        else:
            raise ValueError("Invalid input for chirp parameter")
        
        return np.vstack([chirp_arr[i] + self.load_slice_etas() for i in range(self.Nzeta)])

    def compute_bunching(self, n, thetas):
        """ 
        Function to calculate bunching in a single slice or across many slices

        Parameters:
            n (int): harmonic number to calculate bunching at. Calculates <e^(i*n*theta)>
            thetas (float array, size Np or Nzeta x Np): thetas for a single slice or for the whole bunch

        Returns:
            b (complex, single number of array length Nzeta): nth harmonic bunching for this slice or along all slices 
        """
        return np.mean(np.exp(1j*n*thetas), axis=-1)
