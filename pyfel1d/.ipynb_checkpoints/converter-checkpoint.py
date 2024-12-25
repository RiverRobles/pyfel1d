import numpy as np
from scipy.special import jv, j0, j1
from scipy.constants import e, c, physical_constants, m_e, epsilon_0
Ia = e*c/physical_constants['classical electron radius'][0]
from pyfel1d.fel import OneDFELSolver

class FELScalingConverter:
    """
    A class to convert between scaled and unscaled coordinates for the 1D FEL.

    Attributes
    ----------
    Ipk : float
        Peak current in amperes
    K0 : float
        Reference undulator K 
    gamma0 : float 
        Reference beam Lorentz factor
    lamr0 : float
        Reference radiation wavelength in meters
    lamu : float
        Undulator period in meters
    sigx : float 
        Beam size in meters 
    Lz : float
        Undulator length in meters
    dz : float
        Undulator integration step size in meters
    Ls : float
        Beam window length in meters 
    ds : float
        Slice length in meters
    N_per_slice : int 
        Number of real electrons in one radiation wavelength long slice 
    energy_MeV : float
        Reference beam energy in MeV
    xi0 : float
        Reference xi parameter for JJ calculations
    JJ(h) : function
        Calculates JJ for harmonic h.
    rho1D : float
        1D Pierce parameter rho.
    rho : float
        Pierce parameter used for scaling, defaults to rho1D. 
    ku : float
        Undulator wavenumber in 1/m.
    kr : float
        Reference radiation wavenumber in 1/m.
    Lg1D : float
        1D gain length in meters.
    MX_etad : float
        Ming Xie's diffraction parameter.
    MX_etae(emit) : function
        Ming Xie's emittance parameter (function of geometric emittance in meters).
    MX_etag(sigdelta) : function
        Ming Xie's energy spread parameter (function of relative energy spread).
    """
    def __init__(self, Ipk=None, K0=None, gamma0=None, lamr0=None, lamu=None, sigx=None, Lz=None, Ls=None, dz=None, ds=None):
        """
        Initialize the FELScalingConverter class.
        
        Parameters
        ----------
        Ipk : float
            Peak current in amperes (default is None).
        K0 : float
            Reference undulator K (default is None). 
        gamma0 : float 
            Reference beam Lorentz factor (default is None).
        lamr0 : float
            Reference radiation wavelength in meters (default is None).
        lamu : float
            Undulator period in meters (default is None).
        sigx : float 
            Beam size in meters (default is None).
        Lz : float
            Undulator length in meters (default is None).
        dz : float
            Undulator integration step size in meters (default is None).
        Ls : float
            Beam window length in meters (default is None).
        ds : float
            Slice length in meters (default is None).
        """
        self.Ipk = Ipk
        self.K0 = K0
        self.gamma0 = gamma0
        self.lamr0 = lamr0
        self.lamu = lamu
        self.sigx = sigx

        self.Lz = Lz
        self.dz = dz
        self.Ls = Ls
        self.ds = ds

        assert self.Ipk is not None, 'Peak current must be specified'
        assert self.sigx is not None, 'Beam size must be specified'

        keys = ['K0', 'lamr0', 'gamma0', 'lamu']
        
        if all(getattr(self, key) is not None for key in keys):
            print(r"Including K0, lamu, lamr0, and gamma0 is overspecified, only include three")
        elif all(getattr(self, key) is not None for key in ['K0', 'lamr0', 'gamma0']):
            self.lamu = self.lamr0*(2*self.gamma0**2)/(1 + self.K0**2/2)
        elif all(getattr(self, key) is not None for key in ['K0', 'lamu', 'gamma0']):
            self.lamr0 = self.lamu/(2*self.gamma0**2)*(1 + self.K0**2/2)
        elif all(getattr(self, key) is not None for key in ['K0', 'lamr0', 'lamu']):
            self.gamma0 = np.sqrt(self.lamr0/(2*self.lamu)*(1 + self.K0**2/2))
        elif all(getattr(self, key) is not None for key in ['lamu', 'lamr0', 'gamma0']):
            self.K0 = np.sqrt(2*(2*self.gamma0**2*self.lamr0/self.lamu - 1))
        else:
            raise ValueError('3 of K, lamr, lamu, gamma must be specified')

        self.N_per_slice = int(self.Ipk/(e*c) * self.lamr0)
        self.energy_MeV = 1e-6*self.gamma0*m_e*c**2/e
        
        self.xi0 = self.K0**2 / (4 + 2*self.K0**2)
        self.JJ = lambda h: 0 if h%2==0 else jv(-(h+1)/2, h*self.xi0) + jv(-(h-1)/2, h*self.xi0)
        
        self.calculate_rho1D()
        self.rho = self.rho1D
        self.ku = 2*np.pi/self.lamu
        self.kr = 2*np.pi/self.lamr0
        
        self.Lg1D = self.lamu / (4*np.pi*np.sqrt(3)*self.rho1D)
        self.MX_etad = self.Lg1D / (2*self.kr*self.sigx**2)
        self.MX_etae = lambda emit: 2*self.Lg1D*self.kr/self.sigx**2*emit**2
        self.MX_etag = lambda sigdelta: 2*self.Lg1D*self.ku*sigdelta
        
    def calculate_rho1D(self):
        """
        Calculate the 1D Pierce parameter.
        """
        self.rho1D = (1/(8*np.pi)*self.Ipk/Ia*(self.K0*self.JJ(1)/(1+self.K0**2/2))**2*self.gamma0*self.lamr0**2/(2*np.pi*self.sigx**2))**(1/3)

    def z_to_zhat(self):
        """
        Numerical constant to convert z into zhat.
        """
        return 2*self.rho*self.ku

    def s_to_zeta(self):
        """
        Numerical constant to convert s into zeta.
        """
        return 2*self.rho*self.kr

    def Eh_to_ah(self, h=1):
        """
        Numerical constant to convert Eh into ah.
        """
        return e*self.K0*self.JJ(h)/(4*self.ku*self.rho**2*self.gamma0**2*m_e*c**2)

    def Esqr_to_P(self):
        """
        Numerical constant to convert |E|^2 into P.
        """
        return 2*epsilon_0*c*2*np.pi*self.sigx**2
    
    def MX_Delta(self, emit, sigdelta):
        """
        Calculate Ming Xie's Delta parameter as a function of geometric emittance and relative energy spread.

        Parameters
        ----------
        emit : float
            Geometric emittance in meters.
        sigdelta : float
            Relative energy spread.

        Returns 
        -------
        Delta : float
            Ming Xie's Delta parameter
        """
        etad = self.MX_etad
        etae = self.MX_etae(emit)
        etag = self.MX_etag(sigdelta)

        a = [0.45, 0.57, 0.55, 1.6, 3, 2, 0.35, 2.9, 2.4, 51, 0.95, 3, 5.4, 0.7, 1.9, 1140, 2.2, 2.9, 3.2]

        return a[0]*etad**a[1] + a[2]*etae**a[3] + a[4]*etag**a[5] + a[6]*etae**a[7]*etag**a[8] + a[9]*etad**a[10]*etag**a[11] + a[12]*etad**a[13]*etae**a[14] + a[15]*etad**a[16]*etae**a[17]*etag**a[18]

    def calculate_rho3D(self, emit, sigdelta):
        """
        Calculate effective 3D rho from Ming Xie formalism as a function of geometric emittance and relative energy spread.

        Parameters
        ----------
        emit : float
            Geometric emittance in meters.
        sigdelta : float
            Relative energy spread.

        Returns 
        -------
        rho3D : float
            Effective 3D rho parameter corrected for emittance, energy spread, and diffraction.
        """
        return self.rho1D / (1 + self.MX_Delta(emit, sigdelta))

    def convert_solver_results(self, solver):
        """
        Convert the results of a OneDFELSolver simulation into real units. 

        Parameters
        ----------
        solver : instance of OneDFELSolver
            An instance of the OneDFELSolver class which has had a simulation run. solver is modified in place.
        """
        solver.s_arr = solver.zeta_arr / self.s_to_zeta()
        solver.z_arr = solver.zhat_arr / self.z_to_zhat()
        solver.z_store = solver.zhat_store / self.z_to_zhat()

        if solver.a_final.ndim == 1:
            solver.E_final = solver.a_final / self.Eh_to_ah()
            solver.E_store = solver.a_store / self.Eh_to_ah()
        elif solver.a_final.ndim == 2:
            solver.E_final = solver.a_final / np.array([self.Eh_to_ah(h) for h in solver.harmonics])[:,None]
            solver.E_store = solver.a_store / np.array([self.Eh_to_ah(h) for h in solver.harmonics])[None,:,None]
        
        solver.P_final = np.abs(solver.E_final)**2 * self.Esqr_to_P()
        solver.P_store = np.abs(solver.E_store)**2 * self.Esqr_to_P()
        solver.U_final = np.trapz(solver.P_final, solver.s_arr/3e8)
        solver.U_store = np.trapz(solver.P_store, solver.s_arr/3e8, axis=-1)

    def get_solver(self):
        """
        Create an instance of the OneDFELSolver class based on these real units.

        Returns 
        -------
        solver : instance of OneDFELSolver
            An instance of the OneDFELSolver class with zhat and zeta arrays corresponding to the real parameters specified in this class. 
            Adds a function to solver generate_beam_simple(shift_MeV, espread_MeV, shotnoise) making it easier to generate a beam with proper shotnoise. 
        """
        assert self.Lz is not None, 'Must define undulator length Lz'
        assert self.dz is not None, 'Must define undulator step size dz'
        assert self.Ls is not None, 'Must define beam window length Ls'
        assert self.ds is not None, 'Must define beam window step size ds'

        Lzhat = self.z_to_zhat() * self.Lz
        dzhat = self.z_to_zhat() * self.dz 
        Lzeta = self.s_to_zeta() * self.Ls
        dzeta = self.s_to_zeta() * self.ds 

        solver = OneDFELSolver(Lzhat=Lzhat,
                               Nzhat=int(Lzhat/dzhat),
                               Lzeta=Lzeta,
                               Nzeta=int(Lzeta/dzeta))
    
        solver.s_arr = solver.zeta_arr / self.s_to_zeta()
        solver.z_arr = solver.zhat_arr / self.z_to_zhat()
        solver.K0 = self.K0
        
        solver.generate_beam_simple = lambda shift_MeV, espread_MeV, shotnoise: solver.generate_beam(bdes=shotnoise/np.sqrt(self.N_per_slice), 
                                                                                                     chirp_arr=shift_MeV/self.energy_MeV/self.rho,
                                                                                                     Np=2048,
                                                                                                     M=8, 
                                                                                                     enforce_bunching=False, 
                                                                                                     rel_bdes_tol=1e-1, 
                                                                                                     slice_espread_function=lambda Np: np.random.normal(0, espread_MeV/self.energy_MeV/self.rho, Np)) 

        return solver