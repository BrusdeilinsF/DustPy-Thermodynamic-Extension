# DustPy-Thermodynamic-Extension
This repository contains the code developed for the bachelor's thesis:<br>
<br>
*Extending DustPy: Integrating a Thermodynamic Model into Protoplanetary
Disk Simulations*<br>
<br>
A special thank goes T. Birnstiel for all the support and advise working on this project.<br>
<br>
Uses a simplified version of the thermodynamic model described in:<br>
A. Ziampras, C. P. Dullemond, T. Birnstiel, M. Benisty, and R. P. Nelson. Spirals, rings,
and vortices shaped by shadows in protoplanetary discs: from radiative hydrodynamical
simulations to observable signatures. Monthly notices of the royal astronomical society, 540
(1):1185–1201, June 2025. doi: 10.1093/mnras/staf785.

Uses the opacity table "default_opacities_smooth.npz" by:<br>
Birnstiel, T., Dullemond, C. P., Zhu, Z., et al. 2018, ApJL, 869, L45 
https://github.com/birnstiel/dsharp_opac/blob/master/dsharp_opac/data/default_opacities_smooth.npz.<br>
<br>
Integrated into the DustPy framework by:<br>
S. M. Stammler and T. Birnstiel. DustPy: A Python Package for Dust Evolution in Protoplane-
tary Disks. The Astrophysical Journal, 935(1):35, Aug. 2022. doi: 10.3847/1538-4357/ac7d58.
URL https://stammler.github.io/dustpy/index.html.<br>
<br>
<br>
Basic Usage:<br>
To run the simulation you need to run "MAIN_V_1.0.ipynb", but please note, that "_Temperature.py" and "_opacity.py" are required for the extension to work.<br>
<br>
If you want to compare your resulting temperatures for different alphas and other parameters you can use "compare_temperatures.ipynb". Note, that you need to establish a certain folder-structure with the data.hdf5 files you want to compare: "runs/a*/data00**.hdf5"<br>
Feel free to change the code if you want to do other comparisons.<br>
<br>
If you want to plot the changes in certain results like surface densities, fluxes, etc. you can use "plot_changes.ipynb". Please note, that this also requires a certain folder-structure. Just create a folder named "data_nat" inside your "data" folder and save the snapshots you want to use as a baseline in here. This file now compares the snapshots of your latest run, saved in the "data"-folder to the reference saved in "data_nat" and creates the respective plots.
