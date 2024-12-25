import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numexpr as ne
from joblib import Parallel, delayed
import mpmath 
from scipy.special import jv 
from particles import ParticleLoader
from scipy.interpolate import CubicSpline
from numba import njit

class OneDFELSolver:
    """
    A class to perform simulations of free-electron lasers (FEL) in one dimension in scaled coordinates. 
    
    Attributes
    ----------
    Lzhat : float
        Length in the zhat dimension.
    Nzhat : int
        Number of grid points in the zhat dimension.
    Lzeta : float
        Length in the zeta dimension.
    Nzeta : int
        Number of grid points in the zeta dimension.
    time_independent : bool
        Indicates if the simulation is time-independent.
    chi : nd.array
        Beam current profile, peak value should be 1.
    a0 : nd.array
        Seed field.
    theta0 : nd.array
        Initial theta values for particles.
    eta0 : nd.array
        Initial eta values for particles.
    harmonics : ndarray
        Harmonics tracked in the simulation.
    K0 : float
        Reference undulator K value for harmonic simulations.
    taper : ndarray
        Normalized taper strength along the undulator.
    """
    def __init__(self, Lzhat=30, Nzhat=1000, Lzeta=40, Nzeta=500, time_independent=False):
        """
        Initialize the OneDFELSolver class.
        
        Parameters
        ----------
        Lzhat : float, optional
            Length in the zhat dimension (default is 30).
        Nzhat : int, optional
            Number of grid points in the zhat dimension (default is 1000).
        Lzeta : float, optional
            Length in the zeta dimension (default is 40).
        Nzeta : int, optional
            Number of grid points in the zeta dimension (default is 500).
        time_independent : bool, optional
            Indicates if the simulation is time-independent (default is False).
        """
        self.Lzhat = Lzhat
        self.Nzhat = Nzhat
        self.Lzeta = Lzeta
        self.Nzeta = Nzeta
        self.time_independent = time_independent
        self.evaluate_grids()

        self.chi = None
        self.a0 = None 
        self.theta0 = None
        self.eta0 = None

        self.harmonics = None
        self.K0 = None
        
        self.taper = None

    def evaluate_grids(self):
        """
        Evaluate the grids for zhat and zeta dimensions and set grid spacing.
        """
        self.zhat_arr = np.linspace(0, self.Lzhat, self.Nzhat, endpoint=False)
        self.dzhat = self.zhat_arr[1]

        self.zeta_arr = np.linspace(0, self.Lzeta, self.Nzeta, endpoint=False)
        self.dzeta = self.zeta_arr[1]
        
        if self.time_independent:
            self.Nzeta = 1
        else:
            self.Nzeta = len(self.zeta_arr)

    def generate_beam(self, bdes, chirp_arr=0.0, Np=2048, M=4, enforce_bunching=False, rel_bdes_tol=1e-1, slice_espread_function=lambda Np: np.zeros(Np)):
        """
        Generate the initial beam parameters.
        
        Parameters
        ----------
        bdes : float or ndarray 
            If float, specifies desired rms bunching for all slices. If ndarray, sets different desired rms bunching for different slices.
        chirp_arr : float or ndarray, optional
            If float, offset for eta coordinates. If ndarray, is a zeta-dependent energy chirp (default is 0.0).
        Np : int, optional
            Number of macroparticles per slice (default is 2048).
        M : int, optional
            Half the number of macroparticles per beamlet (default is 4).
        enforce_bunching : bool, optional
            If True, bunching is ensured to be close to desired rms bunching (default is False).
        rel_bdes_tol : float, optional
            Allowed relative deviation of bunching from desired rms bunching, if enforce_bunching is True (default is 1e-1).
        slice_espread_function : callable, optional
            Function to generate slice energy spread (default returns zeros).
        """
        loader = ParticleLoader(Np=Np, M=M, 
                                Nzeta=self.Nzeta, 
                                enforce_bunching=enforce_bunching, rel_bdes_tol=rel_bdes_tol,
                                slice_espread_function = slice_espread_function)
        
        self.theta0 = loader.load_all_thetas(bdes=bdes)
        self.eta0 = loader.load_all_etas(chirp_arr)

    def calculate_moments(self, n, m):
        """
        Calculate moments for the input beam.
        
        Parameters
        ----------
        n : int
            Harmonic number.
        m : int
            Power of eta.
        
        Returns
        -------
        float
            Mean value of the moments.
        """
        return np.mean(self.eta0**m*np.exp(1j*n*self.theta0), axis=-1)
    
    def change_Lzhat_keep_step(self, new_Lzhat):
        """
        Change the length in the zhat dimension while keeping the grid step constant.
        
        Parameters
        ----------
        new_Lzhat : float
            New length in the zhat dimension.
        """
        self.Lzhat = new_Lzhat
        self.zhat_arr = np.arange(0, self.Lzhat, self.dzhat)
        self.Nzhat = len(self.zhat_arr)
        self.taper = None
    
    def convert_lists_to_arrays(self):
        """
        Convert any list inputs into arrays.
        """
        attributes = ['chi', 'a0', 'theta0', 'eta0', 'harmonics', 'taper']
        for attr in attributes:
            value = getattr(self, attr)
            if isinstance(value, list):
                setattr(self, attr, np.array(value))

    def check_inputs(self):
        """
        Check if inputs are present and valid, and assign default values as needed.
        """
        self.convert_lists_to_arrays()
        
        if self.chi is None:
            print(r'Beam profile $\chi$ is unspecified, setting it to 1 for all zeta')
            self.chi = np.ones(self.Nzeta)
        assert (self.chi.ndim == 1) & (self.chi.shape[0] == self.Nzeta), 'Input chi array is invalid, should be a 1D array of length Nzeta'
        
        if self.harmonics is None:
            print('Found no input harmonics array, assuming only first harmonic present')
            self.harmonics = np.array([1])
        assert self.harmonics[0] == 1, 'First value in harmonics must be 1'
        assert (self.harmonics.ndim == 1) & (self.harmonics.shape[0] > 0) & (self.harmonics.dtype == int), 'Harmonics input should be 1D array of integers'

        if self.taper is None:
            print('Found no input taper array, assuming it is zero everywhere')
            self.taper = np.zeros(self.Nzhat)
        assert (self.taper.ndim == 1) & (self.taper.shape[0] == self.Nzhat), 'Taper input should be 1D array of length Nz'
        
        if self.time_independent:
            if (self.a0 is None):
                print(r'Input seed field $a_0$ is unspecified, setting it to zero')
                self.a0 = np.array([[0.0j] for i in range(len(self.harmonics))])
            elif isinstance(self.a0, (float, int, complex)):
                if self.harmonics.shape[0] > 1: 
                    print("Scalar input for a0 is assumed to be for the first harmonic")
                    self.a0 = np.vstack([[self.a0], [[0.0j] for i in range(self.harmonics.shape[0]-1)]])
                else:
                    self.a0 = np.array([[self.a0]])
            elif (self.a0.ndim == 1):
                assert self.a0.shape[0] == self.harmonics.shape[0], 'a0 and harmonics must have the same shape'
                self.a0 = np.array([[a0val] for a0val in self.a0])
            elif self.a0.ndim == 2:
                assert self.a0.shape[0] == self.harmonics.shape[0], 'First axis of a0 must have same length as harmonics array'
                assert self.a0.shape[1] == self.Nzeta, 'Second axis of a0 must have length Nzeta'
            else:
                raise ValueError('a0 input is invalid, see documentation for valid inputs')
        else:
            if self.a0 is None:
                print(r'Input seed field $a_0$ is unspecified, setting it to zero')
                self.a0 = np.array([np.zeros(self.Nzeta) for i in range(self.harmonics.shape[0])])
            elif self.a0.ndim == 1:
                assert self.a0.shape[0] == self.Nzeta, 'One-dimensional vector input for a0 must length as Nzeta'
                if self.harmonics.shape[0] > 1: 
                    print("1D vector input for a0 is assumed to be for the first harmonic")
                    self.a0 = np.vstack([self.a0, [np.zeros(self.Nzeta) for i in range(self.harmonics.shape[0]-1)]])
                else:
                    self.a0 = np.array([self.a0])
            elif self.a0.ndim == 2:
                assert self.a0.shape[0] == self.harmonics.shape[0], 'First axis of a0 must have same length as harmonics array'
                assert self.a0.shape[1] == self.Nzeta, 'Second axis of a0 must have length Nzeta'
            else:
                raise ValueError('a0 input is invalid, see documentation for valid inputs')

        if (self.theta0 is None) or (self.eta0 is None):
            print('Initial particles coordinates were not fully specified, assuming quiet loading of 2048 particles per slice with zero energy spread')
            self.generate_beam(bdes=0)
        assert self.theta0.shape == self.eta0.shape, r'$\theta$ and $\eta$ must be the same size'
        if (self.time_independent) and (self.theta0.ndim == 1):
            self.theta0 = np.array([self.theta0])
            self.eta0 = np.array([self.eta0])
        assert self.theta0.shape[0] == self.Nzeta, r'First axis of $\theta$ and $\eta$ must have same size as $\zeta$ array'

        idx = self.harmonics % 2 != 0
        if len(self.harmonics[idx]) < len(self.harmonics): print('Removing even harmonics')
        self.harmonics = self.harmonics[idx]
        self.a0 = self.a0[idx]
    
        if len(self.harmonics) == 1:
            self.JJs = np.array([1])
            step_field_func = self.step_field
            step_particles_func = self.step_particles
        else:
            assert self.K0 is not None, r'Must specify $K_0$ in order to include harmonic fields'
            
            self.harmonic_couplings = np.array([self.JJ(h, self.K0)**2/self.JJ(1, self.K0)**2 for h in self.harmonics])
            step_field_func = self.step_field_harmonics
            step_particles_func = self.step_particles_harmonics

        return step_particles_func, step_field_func 
        
    def run_simulation(self, steps_between_store=10, verbose=False, store_particles=False):
        """
        Run an FEL simulation with the specified parameters.
        
        Parameters
        ----------
        steps_between_store : int
            Number of integration steps in zhat before storing the field and particles (default is 10).
        verbose : bool
            If True, print status of simulation in 10% intervals (default is False).
        store_particles : bool
            If True, store particles during simulation in addition to field (default is False).
        """
        step_particles_func, step_field_func = self.check_inputs()
        
        if verbose:
            print(f'Beginning simulation including harmonics: {self.harmonics}')
        
        self.zhat_store = self.zhat_arr[::steps_between_store]
        self.a_store = np.zeros((self.zhat_store.shape[0], self.harmonics.shape[0], self.Nzeta)).astype(complex)
        if store_particles:
            self.theta_store, self.eta_store = np.zeros((self.zhat_store.shape[0], self.Nzeta, self.theta0.shape[1])), np.zeros((self.zhat_store.shape[0], self.Nzeta, self.theta0.shape[1]))
    
        theta, eta = self.theta0, self.eta0
        a = self.a0

        self.a_store[0] = a
        if store_particles:
            self.theta_store[0] = theta
            self.eta_store[0] = eta
        
        theta, eta = step_particles_func(theta, eta, a, -self.dzhat/2, self.taper[0])
        
        for i in range(self.Nzhat):
            if (i % int(self.Nzhat/10) == 0) and verbose:
                print(f'{100*i / self.Nzhat:.0f}% done')
                
            if (i % steps_between_store == 0) & (i > 0):
                self.a_store[int(i / steps_between_store)] = a
                if store_particles:
                    self.theta_store[int(i / steps_between_store)] = theta
                    self.eta_store[int(i / steps_between_store)] = eta
                
            a = step_field_func(theta, eta, a, self.dzhat, self.chi)
            theta, eta = step_particles_func(theta, eta, a, self.dzhat, self.taper[i])

            if not self.time_independent:
                a = CubicSpline(self.zeta_arr, a, axis=-1)(self.zeta_arr - self.dzhat)
                a[:,0] *= 0.0
                
        self.a_store = np.squeeze(self.a_store)
        if store_particles:
            self.theta_store = np.squeeze(self.theta_store)
            self.eta_store = np.squeeze(self.eta_store)

        self.a_final = np.squeeze(a)
        self.theta_final = np.squeeze(theta)
        self.eta_final = np.squeeze(eta)
        if verbose:
            print(f'Done')
        
    def step_particles_harmonics(self, theta, eta, a, dz, taper):
        """
        Propagate particles through one integration step including harmonic field components.

        Parameters
        ----------
        theta : ndarray
            Ponderomotive phase coordinates.
        eta : ndarray
            Relative energy deviation coordinates.
        a : ndarray
            Field values.
        dz : float
            Integration step 
        taper : float
            Normalized taper strength at this position.
        """
        a_exp = a[:,:,None]
        h_exp = self.harmonics[:,None,None]
        theta_exp = theta[None,:,:]
        eta_change = np.sum(ne.evaluate("-2*dz*real(a_exp*exp(-1j*h_exp*theta_exp))"), axis=0)
        return ne.evaluate("theta + (eta - taper)*dz"), ne.evaluate("eta + eta_change")

    def step_particles(self, theta, eta, a, dz, taper):
        """
        Propagate particles through one integration step excluding harmonic field components. Runs faster than step_particles_harmonics.

        Parameters
        ----------
        theta : ndarray
            Ponderomotive phase coordinates.
        eta : ndarray
            Relative energy deviation coordinates.
        a : ndarray
            Field values.
        dz : float
            Integration step 
        taper : float
            Normalized taper strength at this position.
        """
        a_exp = a[0][:,None]
        return ne.evaluate("theta + (eta - taper)*dz"), ne.evaluate("eta - 2*dz*real(a_exp*exp(-1j*theta))")

    def step_field_harmonics(self, theta, eta, a, dz, chi):
        """
        Propagate field through one integration step including harmonic field components.

        Parameters
        ----------
        theta : ndarray
            Ponderomotive phase coordinates.
        eta : ndarray
            Relative energy deviation coordinates.
        a : ndarray
            Field values.
        dz : float
            Integration step 
        chi: ndarray
            Beam current profile with peak value 1. 
        """
        b = np.vstack([self.harmonic_couplings[h_ind]*np.mean(ne.evaluate("exp(1j*h*theta)"), axis=1) for h_ind, h in enumerate(self.harmonics)])
        return ne.evaluate("a + dz*chi*b")

    def step_field(self, theta, eta, a, dz, chi):
        """
        Propagate field through one integration step excluding harmonic field components. Runs faster than step_field_harmonics.

        Parameters
        ----------
        theta : ndarray
            Ponderomotive phase coordinates.
        eta : ndarray
            Relative energy deviation coordinates.
        a : ndarray
            Field values.
        dz : float
            Integration step 
        chi: ndarray
            Beam current profile with peak value 1. 
        """
        b = np.mean(ne.evaluate("exp(1j*theta)"), axis=1)
        return ne.evaluate("a + dz*chi*b")

    def JJ(self, h, K):
        """
        Calculate Bessel coupling factors for arbitrary harmonics. 

        Parameters
        ----------
        h : int
            Harmonic number to calculate JJ for. 
        K : float
            Undulator K value to calculate JJ for. 
        """
        if h % 2 == 0:
            return 0
        xi = K**2/(4+2*K**2)
        return jv(-(h+1)/2, h*xi) + jv(-(h-1)/2, h*xi)
    
class LinearTheory:
    """
    A class to compute linear theory solutions to the FEL problem.
    
    Attributes
    ----------
    solver : OneDFELSolver
        An instance of the OneDFELSolver class.
    w_arr : np.ndarray
        Array to store w(zeta, zeta') to speed up Green's function calculations.
    """
    def __init__(self, solver):
        """
        Initialize the LinearTheory class.
        
        Parameters
        ----------
        solver : OneDFELSolver
            An instance of the OneDFELSolver class.
        """
        self.solver = solver

        self.w_arr = None

    @staticmethod
    @njit
    def w(zeta, zetap, zeta_arr, chi):
        """
        Calculate w(zeta, zeta') assuming zeta and zeta'
        
        Parameters
        ----------
        zeta : float
            Zeta value.
        zetap : float
            Zeta prime value.
        zarray : array
            Array of zeta values.
        chi : float
            Chi parameter.
        
        Returns
        -------
        type
            Result of function 'w'.
        """
        if zeta > zetap:
            idx = (zeta_arr >= zetap) & (zeta_arr < zeta)
            return chi[idx].mean()
        else:
            return 0.0

    def calc_on_grid(self, function, *args):
        """
        Calculate values of function on the 2D zeta grid.
        
        Parameters
        ----------
        function : callable
            Function to be evaluated.
        *args : tuple
            Additional positional arguments to pass to the function.

        Returns
        -------
        nd.array
            Evaluated values on the grid.
        """
        values = np.zeros((self.solver.Nzeta, self.solver.Nzeta)).astype(complex)
        for i, zeta in enumerate(self.solver.zeta_arr):
            for j, zetap in enumerate(self.solver.zeta_arr):
                values[i,j] = function(zeta, zetap, *args)

        return values

    def hypergeometricpfq(self, x, p, q):
        """
        Evaluate HypergeometricPFQ({},{p,q},x) for complex input returning a single complex number.

        Parameters
        ----------
        x : complex
            Complex point at which to evaluate the function.
        p : float
            First part of second argument of HypergeometricPFQ.
        q : float
            Second part of second argument of HypergeometricPFQ.
        """
        val = mpmath.hyper([], [p, q], x)
        return float(val.real) + 1j*float(val.imag)

    @staticmethod
    def gf_field_exact(zhat, zeta, zetap, zeta_arr, w_arr):
        """
        Exact Green's function for an initial field.
        
        Parameters
        ----------
        z : float
            z dimension variable.
        zeta : float
            Zeta variable.
        zetap : float
            Zeta prime variable.
        zeta_arr : array
            Array of zeta values.
        chi : np.ndarray
            Chi parameter.
        
        Returns
        -------
        ndarray
            Green's function value.
        """
        tau = zeta - zetap
        
        dzeta = zeta_arr[1]
        i,j = [np.argwhere(zeta_arr==zeta_val)[0][0] for zeta_val in [zeta, zetap]]
        w = w_arr[i,j]
            
        arg = -1j*w*tau*(zhat-tau)**2/4
        if (tau > 0) & (tau < zhat) & (abs(tau - zhat) > dzeta):
            return -1j*w*tau*(zhat-tau)*self.hypergeometricpfq(arg, 3/2, 2)
        else:
            return 0.0j

    @staticmethod
    @njit
    def gf_field_asymptotic(zhat, zeta, zetap, zeta_arr, w_arr):
        """
        Asymptotic Green's function for an initial field.
        
        Parameters
        ----------
        z : float
            z dimension variable.
        zeta : float
            Zeta variable.
        zetap : float
            Zeta prime variable.
        zeta_arr : array
            Array of zeta values.
        chi : np.ndarray
            Chi parameter.
        
        Returns
        -------
        ndarray
            Green's function value.
        """
        tau = zeta - zetap
        
        dzeta = zeta_arr[1]
        i,j = [np.argwhere(zeta_arr==zeta_val)[0][0] for zeta_val in [zeta, zetap]]
        w = w_arr[i,j]
        
        arg = -1j*w*tau*(zhat-tau)**2/4
        if (tau > 0) & (tau < zhat) & (abs(tau - zhat) > dzeta):
            return -1j*w*tau*(zhat-tau)*np.exp(3*arg**(1/3))/(1e-20+4*np.sqrt(3*np.pi)*arg**(5/6))
        else:
            return 0.0j

    @staticmethod
    def gf_bunching_exact(zhat, zeta, zetap, zeta_arr, w_arr):
        """
        Exact Green's function for an initial bunching.
        
        Parameters
        ----------
        z : float
            z dimension variable.
        zeta : float
            Zeta variable.
        zetap : float
            Zeta prime variable.
        zeta_arr : array
            Array of zeta values.
        chi : np.ndarray
            Chi parameter.
        
        Returns
        -------
        ndarray
            Green's function value.
        """
        tau = zeta - zetap
        
        dzeta = zeta_arr[1]
        i,j = [np.argwhere(zeta_arr==zeta_val)[0][0] for zeta_val in [zeta, zetap]]
        w = w_arr[i,j]
            
        arg = -1j*w*tau*(zhat-tau)**2/4
        if (tau > 0) & (tau < zhat) & (abs(tau - zhat) > dzeta):
            return self.hypergeometricpfq(arg, 1/2, 1)
        else:
            return 0.0j

    @staticmethod
    @njit
    def gf_bunching_asymptotic(zhat, zeta, zetap, zeta_arr, w_arr):
        """
        Asymptotic Green's function for an initial bunching.
        
        Parameters
        ----------
        z : float
            z dimension variable.
        zeta : float
            Zeta variable.
        zetap : float
            Zeta prime variable.
        zeta_arr : array
            Array of zeta values.
        chi : np.ndarray
            Chi parameter.
        
        Returns
        -------
        ndarray
            Green's function value.
        """
        tau = zeta - zetap

        dzeta = zeta_arr[1]
        i,j = [np.argwhere(zeta_arr==zeta_val)[0][0] for zeta_val in [zeta, zetap]]
        w = w_arr[i,j]
            
        arg = -1j*w*tau*(zhat-tau)**2/4
        if (tau > 0) & (tau < zhat) & (abs(tau - zhat) > dzeta):
            return np.exp(3*arg**(1/3))/(1e-20+2*np.sqrt(3*np.pi)*arg**(1/6))
        else:
            return 0.0j

    @staticmethod
    def gf_modulation_exact(zhat, zeta, zetap, zeta_arr, w_arr):
        """
        Exact Green's function for an initial modulation.
        
        Parameters
        ----------
        z : float
            z dimension variable.
        zeta : float
            Zeta variable.
        zetap : float
            Zeta prime variable.
        zeta_arr : array
            Array of zeta values.
        chi : np.ndarray
            Chi parameter.
        
        Returns
        -------
        ndarray
            Green's function value.
        """
        tau = zeta - zetap
        
        dzeta = zeta_arr[1]
        i,j = [np.argwhere(zeta_arr==zeta_val)[0][0] for zeta_val in [zeta, zetap]]
        w = w_arr[i,j]
            
        arg = -1j*w*tau*(zhat-tau)**2/4
        if (tau > 0) & (tau < zhat) & (abs(tau - zhat) > dzeta):
            return 1j*(zhat-tau)*self.hypergeometricpfq(arg, 1, 3/2)
        else:
            return 0.0j

    @staticmethod
    @njit
    def gf_modulation_asymptotic(zhat, zeta, zetap, zeta_arr, w_arr):
        """
        Asymptotic Green's function for an initial modulation.
        
        Parameters
        ----------
        z : float
            z dimension variable.
        zeta : float
            Zeta variable.
        zetap : float
            Zeta prime variable.
        zeta_arr : array
            Array of zeta values.
        chi : np.ndarray
            Chi parameter.
        
        Returns
        -------
        ndarray
            Green's function value.
        """
        tau = zeta - zetap
        
        dzeta = zeta_arr[1]
        i,j = [np.argwhere(zeta_arr==zeta_val)[0][0] for zeta_val in [zeta, zetap]]
        w = w_arr[i,j]
            
        arg = -1j*w*tau*(zhat-tau)**2/4
        if (tau > 0) & (tau < zhat) & (abs(tau - zhat) > dzeta):
            return 1j*(zhat-tau)*np.exp(3*arg**(1/3))/(1e-20+4*np.sqrt(3*np.pi)*arg**(1/2))
        else:
            return 0.0j

    def calc_gf_on_grid(self, zhat, function):
        """
        Calculate Green's function on the grid.
        
        Parameters
        ----------
        z : float
            z dimension variable.
        function : callable
            Green's function to be evaluated.
        
        Returns
        -------
        np.ndarray
            Evaluated Green's function on the grid.
        """
        return self.calc_on_grid(lambda zeta, zetap: function(zhat, zeta, zetap, self.solver.zeta_arr, self.w_arr))

    def get_linear_solution(self, zhat, include_field=False, include_bunching=False, include_modulation=False, asymptotic=True):
        """
        Obtain the linear solution for the simulation.
        
        Parameters
        ----------
        z : float
            z dimension variable.
        include_field : bool, optional
            Whether to include the field in the solution (default is False).
        include_bunching : bool, optional
            Whether to include the bunching parameter in the solution (default is False).
        include_modulation : bool, optional
            Whether to include the modulation parameter in the solution (default is False).
        asymptotic : bool, optional
            Whether to use the asymptotic form of the Green's function (default is True).
        
        Returns
        -------
        np.ndarray
            The linear solution for the given z.
        """
        linear_soln = np.zeros(self.solver.Nzeta).astype(complex)

        if self.w_arr is None:
            print("First time evaluating Green's functions, must calculate w(zeta,zeta'), future runs will be slightly faster")
            self.w_arr = self.calc_on_grid(lambda zeta, zetap: self.w(zeta, zetap, self.solver.zeta_arr, self.solver.chi))

        zeta = self.solver.zeta_arr
        if include_field:
            a0 = self.solver.a0[0]
            if asymptotic:
                gf = self.calc_gf_on_grid(zhat, self.gf_field_asymptotic)
            else:
                gf = self.calc_gf_on_grid(zhat, self.gf_field_exact)
            linear_soln += CubicSpline(zeta, a0)(zeta - zhat) + np.trapz(gf*a0[None,:], zeta, axis=1)
        if include_bunching:
            b0chi = self.solver.chi * self.solver.calculate_moments(1,0)
            if asymptotic:
                gf = self.calc_gf_on_grid(zhat, self.gf_bunching_asymptotic)
            else:
                gf = self.calc_gf_on_grid(zhat, self.gf_bunching_exact)
            linear_soln += np.trapz(gf*b0chi[None,:], zeta, axis=1)
        if include_modulation:
            p0chi = self.solver.chi * self.solver.calculate_moments(1,1)
            if asymptotic:
                gf = self.calc_gf_on_grid(zhat, self.gf_modulation_asymptotic)
            else:
                gf = self.calc_gf_on_grid(zhat, self.gf_modulation_exact)
            linear_soln += np.trapz(gf*p0chi[None,:], zeta, axis=1)

        return linear_soln