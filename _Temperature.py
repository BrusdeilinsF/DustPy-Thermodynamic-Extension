import astropy.constants as apc
import dustpy.constants as c
import numpy as np
import matplotlib.pyplot as plt

sigma_sb = apc.sigma_sb.cgs.value
kB = apc.k_B.cgs.value
mH = apc.m_p.cgs.value


class Temperature:
    """
    VERSION 1.0
    ----
    
    Main
    ----
    computes the radial temperature in a protoplanetary disk. 
    Every value starting with a sim.* is taken from the dustpy simulation
    and is used to compute the temperature.
    

    Parameters
    ----
    Taken directly from Dustpy:
    - star luminosity :         sim.star.L
    - star temperature:         sim.star.T
    - radial distance :         sim.grid.r
    - pressure scale height :   sim.gas.Hp
    - surface density:          sim.dust.Sigma / sim.gas.Sigma

    taken from opacity table computed in the class 'opacity':
    - Planck-opacity:           kappa_P
        -> capacity of the disk to absorp electromagnetic radiation emitted by the star (mainly UV/VIS/(NIR))
    - Rosseland-opacity:        kappa_R
        -> capacity of the disk to emit electromagnetic radiation (mainly IR/sub-mm)
    -> both are merged into a temperature depending array

    own Input (defined in: def __init__()):
    - floor temperature:        T_ext = 10.0
    - mean molecule mass:       mu = 2.35   
    - adiabatic index:          gamma = 7/5
    - albedo:                   epsilon = 0.5

    Notes
    ----
    - The values defined in the __init__ are subject to change if a different parameters are needed.
    - cgs system is used for all units to match the units used in Dustpy

    References
    ----
    - C. Dullemond, 2013, Theoretical Models of the Structure of Protoplanetary Disks, Les Houches
    - S. M. Stammler and T. Birnstiel. DustPy: A Python Package for Dust Evolution in Protoplane-
      tary Disks. The Astrophysical Journal, 935(1):35, Aug. 2022. doi: 10.3847/1538-4357/ac7d58.
      URL https://stammler.github.io/dustpy/index.html.
    - R. S. Klessen and S. C. O. Glover. Physical processes in the interstellar medium. Saas-
      Fee Advanced Course, 43:85-257, 2016. doi: 10.1007/978-3-662-47890-5 2. 
      URL https://ned.ipac.caltech.edu/level5/Sept19/Klessen/paper.pdf.
    - A. Ziampras, C. P. Dullemond, T. Birnstiel, M. Benisty, and R. P. Nelson. Spirals, rings,
      and vortices shaped by shadows in protoplanetary discs: from radiative hydrodynamical
      simulations to observable signatures. Monthly notices of the royal astronomical society, 540
      (1):1185-1201, June 2025. doi: 10.1093/mnras/staf785
    """
    def __init__(self, opac, T_min=10.0, epsilon= 0.5, gamma = 7/5, mu = 2.35, T_ext = 10.0):
        self.opac = opac
        #self.T_min = T_min
        self.eps = epsilon
        self.gamma = gamma
        self.mu = mu
        self.T_ext = T_ext
        self._last_Q = None
    

    def T_final(self, sim):
        """computation of the final dust temperature
        - uses the mean opacities to compute the optical depths tau_P and tau_R per grain size
        - uses Sigma_d(r,a)/Sigma_d(r) as mass distribution
        - uses dT/dt * dt to determine the change in temperature each time-step
        - clipped to damp oscillations
        """
        # setting dt to the half of a simulation timestep
        Sigmag_old = sim.Sigma_gas_old
        #Sigmag_old = sim.gas._SigmaOld
        Sigmag_new = sim.gas.Sigma
        Sigmag = 0.5 * (Sigmag_new + Sigmag_old)
        Sigmad_old = sim.Sigma_dust_old
        #Sigmag_old = sim.gas._SigmaOld
        Sigmad_new = sim.dust.Sigma
        Sigmad = 0.5 * (Sigmad_new + Sigmad_old)

        H = sim.gas.Hp
        T = sim.gas.T
        r = sim.grid.r
        T_lim = 1500
        # calling opacities 
        kappa_P, kappa_R = self.opac.mean_opacities(sim) 

        # setting 
        Sigma_dust_tot = Sigmad.sum(axis=1)
        #Sigma_dust_tot = np.maximum(Sigma_dust_tot, 1e-10)
        
        #optical depth t_eff
        kappa_P_gas = 1e-3
        kappa_R_gas = 1e-3

        tau_R = 0.25 * Sigma_dust_tot * (1- np.tanh((T - T_lim)/100)) * 0.5 * kappa_R + Sigmag * kappa_R_gas
        tau_P = 0.25 * Sigma_dust_tot * (1- np.tanh((T - T_lim)/100)) * 0.5 * kappa_P + Sigmag * kappa_P_gas
        tau_eff = (3 * tau_R / 8) + np.sqrt(3) / 4 + 1 / (4 * tau_P)

        #incidence angle
        h = H / r
        Theta0 = 2 * 4 * h / 7
        R_rim = 1
        Theta = Theta0 + 0.5 * (1 - Theta0) * (1- np.tanh((r - R_rim)/0.1))
        T_lim = 1500
        

        delta_t = 0.5 * sim.t.prevstepsize

        #since different definitions of nu can be found, feel free to try:

        nu = sim.gas.alpha * np.sqrt(self.gamma) * sim.gas.cs * H
        #nu = sim.gas.alpha * sim.gas.cs**2 / sim.grid.OmegaK

        #computation of the heating terms
        Q_visc = (9 / 4) * Sigmag * nu * sim.grid.OmegaK**2
        Q_irr = sim.star.L / (2 * np.pi * r**2) * (1 - self.eps) * Theta / tau_eff
        Q_cool = -2 * sigma_sb * T**4 / tau_eff
        Q_ext = 2 * sigma_sb * self.T_ext**4 / tau_eff
        Q_tot = Q_visc + Q_irr + Q_cool + Q_ext
        
        #saving data for plotting
        self._last_Q = {"r": sim.grid.r.copy(),"Q_visc": Q_visc.copy(),"Q_irr": Q_irr.copy(),"Q_cool": Q_cool.copy(),"Q_ext": Q_ext.copy(),}

        
        #div_v = np.gradient(r * vr, r) / r
        # Compressional heating/cooling is currently neglected. Therefore, the velocity divergence is set to zero.
        div_v = 0
        P = Sigmag * sim.gas.cs**2

        e = P / (self.gamma - 1)
        Cv = Sigmag * kB /(self.mu * mH * (self.gamma - 1))
        dTdt = (-self.gamma * e * div_v + Q_tot )/ Cv
        dT = dTdt * delta_t

        dT_max = 0.05 * T
        dT = np.clip(dT,-dT_max, dT_max)

        T_new = T + dT
        T_final = T_new
        return T_final

    def plot_heating(self):
        """used for plotting the computed heating values"""
        
        r_au = self._last_Q["r"] / c.au

        fig, ax = plt.subplots(dpi=300)

        ax.loglog(r_au, self._last_Q['Q_visc'], color = 'red', alpha = 0.5, label= 'Q_visc')
        ax.loglog(r_au, self._last_Q['Q_irr'], color = 'green', alpha = 0.5, label= 'Q_irr')
        ax.loglog(r_au, np.abs(self._last_Q['Q_cool']),'--', color = 'blue', alpha = 0.5, label= '|Q_cool|')
        ax.loglog(r_au, self._last_Q['Q_ext'], '--',color = 'grey', alpha = 0.5, label= 'Q_ext')

        ax.set_title("Heating")
        ax.set_xlabel('r [au]')
        ax.set_ylabel(r'$Q$ [erg cm$^{-2}$ s$^{-1}$]')
        ax.set_xlim(1, 400)
        ax.set_ylim(1e-12,400)
        ax.grid(True, which = 'both')
        ax.legend(loc = 'best')
        plt.show()

    def T_dustpy(self, sim):
        """used as the updater for sim.gas.T.updater.updater"""
        return self.T_final(sim)