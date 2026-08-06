import astropy.constants as apc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

h = apc.h.cgs.value
c_light = apc.c.cgs.value
k_B = apc.k_B.cgs.value

class opacity():
    """
    VERSION 1.0
    ----
    
    Main
    -----
    computes the Planck- and Rosseland-mean opacities. 
    Dust opacities are taken from the DSHARP opacity package 
    (Birnstiel et al. 2018), using the file
    'default_opacities_smooth.npz'.

    - cgs system is used for all units to match the units used in Dustpy
    - mean opacities are computed mass weighted and integrated over all wavelengths
    - some arrays are interpolated using interp1d from scipy.interpolate to eliminate shape mismatches
    - safes the opacities in mean_opacities.npz
    - 2d plot-function: def plot_kappa(self):
    - 3d plot-function: def plot_3d(self):

    Parameters
    ----------
    taken from opacity table:
    - Grain-size:                       a (array)  
    - Wavelength:                       lam (array)
    - scattering asymmetry factor:      g (array)
    - absorption opacity:               k_abs_a (array)
    - scattering opacity:               k_sca_a (array)

    taken from Dustpy:
    - gas temperature                   sim.gas.T
    - grain-size:                       sim.dust.a
    
    
    Notes
    ----
    

    References
    ----
    - Birnstiel, T., Dullemond, C. P., Zhu, Z., et al. 2018, ApJL, 869, L45 
      https://github.com/birnstiel/dsharp_opac/blob/master/dsharp_opac/data/default_opacities_smooth.npz.
    - S. M. Stammler and T. Birnstiel. DustPy: A Python Package for Dust Evolution in Protoplane-
      tary Disks. The Astrophysical Journal, 935(1):35, Aug. 2022. doi: 10.3847/1538-4357/ac7d58.
      URL https://stammler.github.io/dustpy/index.html.
    """

    def __init__(self, sim, fname):
        
        opac = np.load(fname)

        file_a = opac['a']
        self._a = sim.dust.a
        self._lam = opac['lam']
        self._k_abs_a = opac['k_abs']
        self._k_sca_a = opac['k_sca']
        self._g = opac['g']
        self._last_kappa = None
        
        #interpolation from (200,210) to (120,210) to match dustpy
        f_abs = interp1d(np.log10(file_a), opac['k_abs'], axis=0, bounds_error=False)
        f_sca = interp1d(np.log10(file_a), opac['k_sca'], axis=0, bounds_error=False)
        f_g   = interp1d(np.log10(file_a), opac['g'],     axis=0, bounds_error=False)
        
        self._k_abs_a = f_abs(np.log10(self._a)) 
        self._k_sca_a = f_sca(np.log10(self._a))  
        self._g       = f_g(np.log10(self._a)) 
        

    def mean_opacities(self, sim):
        lam_2d = self._lam[np.newaxis, :]
        T_2d = sim.gas.T[:, np.newaxis]

        #surface denisities for half time-steps
        Sigmad_old = sim.Sigma_dust_old
        Sigmag_old = sim.Sigma_gas_old
        #Sigmad_old = sim.dust._SigmaOld
        Sigmad_new = sim.dust.Sigma
        Sigmag_new = sim.gas.Sigma
        Sigmad = 0.5 * (Sigmad_new + Sigmad_old)
        Sigmag = 0.5 * (Sigmag_new + Sigmag_old)
        Sigmad_norm = Sigmad / np.sum(Sigmad, axis=1)[:, np.newaxis]
        
        #opacities
        k_tot_a = self._k_abs_a + (1.0 - self._g) * self._k_sca_a
        kappa_mean_abs = (self._k_abs_a * Sigmad_norm[:, :, np.newaxis]).sum(axis=1)
        #kappa_mean_sca = (self._k_sca_a * Sigmad_norm[:, :, np.newaxis]).sum(axis=1)
        kappa_mean_tot = (k_tot_a * Sigmad_norm[:, :, np.newaxis]).sum(axis=1)

        # defining B_lam and dB/ dT
        x = h * c_light / (lam_2d * k_B * T_2d)
        x = np.clip(x, -300, 300)
        coeff = np.exp(np.log(2.0 * h * c_light**2) - 5.0 * np.log(lam_2d))
        B_lam = coeff / np.expm1(x)
        dB_dT = (2 * h * c_light**2 * x / (lam_2d**5 * T_2d) * np.exp(x) / np.expm1(x)**2)
        
        # planck- and rosseland- mean opacities
        num_P = simpson(kappa_mean_abs * B_lam, x=self._lam, axis=-1) 
        den_P = simpson(B_lam, x=self._lam, axis=-1)           

        num_R = simpson(dB_dT, x=self._lam, axis=-1)            
        den_R = simpson((1.0 / kappa_mean_tot) * dB_dT, x=self._lam, axis=-1)

        kappa_P = num_P / den_P
        kappa_R = num_R / den_R

        #preparing the plot
        self._last_kappa = {"T": sim.gas.T.copy(),"r": sim.grid.r.copy(),"a": self._a.copy(),"weights": Sigmad_norm.copy(),"kappa_P": kappa_P.copy(),"kappa_R": kappa_R.copy(),}
        T_tab = self._last_kappa["T"]
        kapP_tab = self._last_kappa["kappa_P"]
        kapR_tab = self._last_kappa["kappa_R"]
        np.savez("mean_opacities.npz",T=T_tab,kappaP=kapP_tab,kappaR=kapR_tab,)
        
        return kappa_P, kappa_R

    def plot_kappa(self):
        """
        Plot the resulting mean opacities as logarithmic 2D scatter plots.
        - x-axis: Temperature T(r) [K]
        - y-axis: Planck and Rosseland mean opacities [cm^2 g^-1]
        - color: Mass-weighted mean grain size a_mean(r) [cm]
        """

        if self._last_kappa is None:
            raise RuntimeError('No opacities computed yet')

        a_cm = np.asarray(self._last_kappa['a'], dtype = float)
        T = np.asarray(self._last_kappa['T'], dtype = float)
        weights = np.asarray(self._last_kappa['weights'], dtype = float)
        kappa_P = np.asarray(self._last_kappa["kappa_P"], dtype = float)
        kappa_R = np.asarray(self._last_kappa["kappa_R"], dtype = float)

        Nr, Na = weights.shape

        # construct grain-size array
        if a_cm.ndim == 1:
            A = np.broadcast_to(a_cm[np.newaxis, :], weights.shape) # Same grain-size grid at every radial position
        elif a_cm.shape == weights.shape:   
            A = a_cm # Radius-dependent grain-size grid

        # ensure that the weights are normalized
        weight_sum = np.sum(weights, axis=1)
        weights_normalized = np.divide(weights, weight_sum[:, np.newaxis], out=np.zeros_like(weights), where=weight_sum[:, np.newaxis] > 0)

        # Mass-weighted mean grain size at every radius
        a_mean = np.sum(weights_normalized * A,axis=1)

        # Masks for valid logarithmic values
        valid_common = (np.isfinite(T) & (T > 0) & np.isfinite(a_mean) & (a_mean > 0))
        valid_P = (valid_common & np.isfinite(kappa_P) & (kappa_P > 0))
        valid_R = (valid_common & np.isfinite(kappa_R) & (kappa_R > 0))
        positive_a_mean = a_mean[np.isfinite(a_mean) & (a_mean > 0)]

        # Same color normalization for both plots
        norm = LogNorm(vmin = np.min(positive_a_mean), vmax = np.max(positive_a_mean))


        fig, (ax_P, ax_R) = plt.subplots(1,2, figsize=(16, 7), dpi=300, sharex=True)

        sc_P = ax_P.scatter(T[valid_P], kappa_P[valid_P], c=a_mean[valid_P], cmap="viridis", norm=norm, s=30, alpha=0.8)
        ax_P.set_title('Planck-mean opacity')
        ax_P.set_xlabel(r'$T$ [K]')
        ax_P.set_ylabel(r'$\kappa_{\mathrm{P}}$ [cm$^2$ g$^{-1}$]')
        ax_P.set_xscale('log')
        ax_P.set_yscale('log')
        ax_P.grid(True, which = 'both', alpha = 0.3)

        sc_R = ax_R.scatter(T[valid_R], kappa_R[valid_R], c=a_mean[valid_R], cmap="viridis", norm=norm, s=30, alpha=0.8)
        ax_R.set_title('Rosseland-mean opacity')
        ax_R.set_xlabel(r'$T$ [K]')
        ax_R.set_ylabel(r'$\kappa_{\mathrm{R}}$ [cm$^2$ g$^{-1}$]')
        ax_R.set_xscale('log')
        ax_R.set_yscale('log')
        ax_R.grid(True, which="both", alpha=0.3)

        cbar = fig.colorbar(sc_P, ax=[ax_P, ax_R], shrink=0.85, pad=0.03)
        cbar.set_label(r'Mass-weighted mean grain size ' r'$\bar{a}(r)$ [cm]')

        fig.suptitle('Mean opacities',fontsize=15)

        plt.show()

        return fig, (ax_P, ax_R)
