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
        function for plotting the resulting opacities in a logarithmic 2D-plot: 
        - x-axis: temperature T [K]
        - y-axis: opacities kappa_*(a(r),T) [cm^2 g^-1]
        """
        if self._last_kappa is None:
            raise RuntimeError('No opacities computed yet')

        T = self._last_kappa["T"]

        fig, ax = plt.subplots(dpi=300)

        ax.set_title('Planck- and Rosseland mean opacities')
        ax.loglog(T, self._last_kappa['kappa_P'], color = 'red', alpha = 0.5, label= r'$\kappa_P(a(r),T)$')
        ax.loglog(T, self._last_kappa['kappa_R'], color = 'blue', alpha = 0.5, label= r'$\kappa_R(a(r),T)$')

        ax.set_xlabel('T [K]')
        ax.set_ylabel(r'$\kappa$ [cm$^(2)$g$^{-1}$]')
        
        ax.grid(True, which = 'both')
        ax.legend(loc = 'best')
        plt.show()

    def plot_3d(self):
        """
        function for plotting the resulting opacities in a logarithmic 3D-plot with colorbar: 
        - x-axis: radius r [au]
        - y-axis: temperature T [K]
        - z-axis: opacities kappa_*(a(r),T) [cm^2 g^-1]
        - colorbar: mass-weighted mean grain-size
        """
        if self._last_kappa is None:
            raise RuntimeError('No opacities computed yet')

        required_keys = {"r", "a", "T", "weights", "kappa_P", "kappa_R"}
        missing_keys = required_keys.difference(self._last_kappa.keys())

        if missing_keys:
            raise RuntimeError(f"Missing data for 3D plot: {sorted(missing_keys)}")

        au_cgs = apc.au.cgs.value
        r_au = np.asarray(self._last_kappa["r"], dtype=float) / au_cgs
        T = np.asarray(self._last_kappa["T"], dtype=float)
        a_cm = np.asarray(self._last_kappa["a"], dtype=float)
        weights = np.asarray(self._last_kappa["weights"], dtype=float)
        kappa_P = np.asarray(self._last_kappa["kappa_P"], dtype=float)
        kappa_R = np.asarray(self._last_kappa["kappa_R"], dtype=float)
        Nr, Na = weights.shape

        if r_au.shape != (Nr,):
            raise ValueError(f"r has shape {r_au.shape}, expected {(Nr,)}")

        if T.shape != (Nr,):
            raise ValueError(f"T has shape {T.shape}, expected {(Nr,)}")

        if kappa_P.shape != (Nr,):
            raise ValueError(f"kappa_P has shape {kappa_P.shape}, expected {(Nr,)}")

        if kappa_R.shape != (Nr,):
            raise ValueError(f"kappa_R has shape {kappa_R.shape}, expected {(Nr,)}")

        # Build grain-size array with same shape as weights
        if a_cm.ndim == 1:
            if a_cm.shape != (Na,):
                raise ValueError(f"a has shape {a_cm.shape}, expected {(Na,)}")
            A = np.broadcast_to(a_cm[np.newaxis, :], weights.shape)
        elif a_cm.shape == weights.shape:
            A = a_cm
        else:
            raise ValueError(f"a has shape {a_cm.shape}. Expected {(Na,)} or {weights.shape}.")

        # Mass-weighted mean grain size
        a_mean = np.sum(weights * A, axis=1)

        # Valid points
        valid_P = (np.isfinite(r_au) & (r_au > 0) & np.isfinite(T) & (T > 0) & np.isfinite(kappa_P) & (kappa_P > 0) & np.isfinite(a_mean) & (a_mean > 0))

        valid_R = (np.isfinite(r_au) & (r_au > 0) & np.isfinite(T) & (T > 0) & np.isfinite(kappa_R) & (kappa_R > 0) & np.isfinite(a_mean) & (a_mean > 0))

        if not np.any(valid_P):
            raise ValueError("No valid Planck-opacity points found.")

        if not np.any(valid_R):
            raise ValueError("No valid Rosseland-opacity points found.")

        positive_a_mean = a_mean[np.isfinite(a_mean) & (a_mean > 0)]
        norm = LogNorm(vmin=np.min(positive_a_mean), vmax=np.max(positive_a_mean))

        fig = plt.figure(figsize=(16, 7), dpi=200)

        # Planck-mean opac
        ax_P = fig.add_subplot(121, projection="3d")
        sc_P = ax_P.scatter(np.log10(r_au[valid_P]), np.log10(T[valid_P]), np.log10(kappa_P[valid_P]), c=a_mean[valid_P], cmap="viridis", norm=norm, s=20)

        ax_P.set_title("Planck-mean opacity")
        ax_P.set_xlabel(r"$\log_{10}(r[\mathrm{au}])$")
        ax_P.set_ylabel(r"$\log_{10}(T[\mathrm{K}])$")
        ax_P.set_zlabel(r"$\log_{10}\left(\kappa_{\mathrm{P}}/"r"[\mathrm{cm^2\,g^{-1}}]\right)$")

        # Rosseland-mean opac
        ax_R = fig.add_subplot(122, projection="3d")
        sc_R = ax_R.scatter(np.log10(r_au[valid_R]), np.log10(T[valid_R]), np.log10(kappa_R[valid_R]), c=a_mean[valid_R], cmap="viridis", norm=norm, s=20)

        ax_R.set_title("Rosseland-mean opacity")
        ax_R.set_xlabel(r"$\log_{10}(r[\mathrm{au}])$")
        ax_R.set_ylabel(r"$\log_{10}(T[\mathrm{K}])$")
        ax_R.set_zlabel(r"$\log_{10}\left(\kappa_{\mathrm{R}}/"r"[\mathrm{cm^2\,g^{-1}}]\right)$")

        cbar = fig.colorbar(sc_R, ax=[ax_P, ax_R], shrink=0.7, pad=0.08)
        cbar.set_label(r"Mass-weighted mean grain size $\bar a(r)$ [cm]")

        plt.show()

        return fig, (ax_P, ax_R)