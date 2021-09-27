# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:36:31 2020

dftproc.py contains functions used to process data from DFT calculations of
interfacial tension with Dr. Huikuan Chao's DFT code.

@author: andyl
"""

# standard libraries
import numpy as np


    
def get_props(filepath, m_co2=22, m_poly=2700/123, N_A=6.022E23, 
              sigma_co2=2.79E-8, sort=False):
    """
    Computes important physical properties from the dft.input file, such as
    density of CO2 in the CO2-rich phase, solubility of CO2 in the polyol-rich
    phase, and specific volume of the polyol-rich phase. 
    The dft.input file is structured as:
        p \t gsrho1b \t gsrho1a \t 10^-gsrho2b \t gsrho2a.
        
    PARAMETERS
    ----------
    filepath : string
        Filepath to file containing densities and pressures (usually dft.input)
    m_co2 : float
        mass of one bead of CO2 in PC-SAFT model [amu/bead] (= Mw / N)
    m_poly : float
        mass of one bead of polyol in PC-SAFT model [amu/bead] (= Mw / N)
    N_A : float
        Avogadro's number (molecules per mol)
    sigma_co2 : float
        sigma parameter for co2 [cm]
    sort : bool
        If True, sorts solubility data in terms of increasing pressure
        
    RETURNS
    -------
    p : list of floats
        pressures corresponding to the solubilities [MPa]
    props : tuple of lists of floats
        Tuple of physical properties calculated (lists of floats):
            rho_co2 : density of CO2 in CO2-rich phase [g/mL]
            solub : solubility of CO2 in polyol-rich phase [w/w]
            spec_vol : specific volume of polyol-rich phase [mL/g]
    """
    # loads data
    data = np.genfromtxt(filepath, delimiter='\t')
    # extracts pressure [MPa] from first column
    p = data[:,0]
    # extracts the density of CO2 in the co2-rich phase [beads/sigma^3]
    rho_co2_v = data[:,1]
    # extracts the density of CO2 in the polyol-rich phase [beads/sigma^3]
    rho_co2_l = data[:,2]
    # extracts the density of polyol in the polyol-rich phase [beads/sigma^3]
    rho_poly_l = data[:,4]
    # conversions from beads/sigma^3 to g/mL
    conv_co2 = m_co2/N_A/sigma_co2**3
    conv_poly = m_poly/N_A/sigma_co2**3
    
    # computes density of CO2 in the CO2-rich phase [g/mL]
    rho_co2 = rho_co2_v*conv_co2
    # computes solubility of CO2 in the polyol-rich phase [w/w]
    solub = rho_co2_l*conv_co2 / (rho_co2_l*conv_co2 + rho_poly_l*conv_poly)
    # computes specific volume of the polyol-rich phase [mL/g]
    spec_vol = 1 / (rho_co2_l*conv_co2 + rho_poly_l*conv_poly)
    
    # sorts data if requested
    if sort:
        inds_sort = np.argsort(p)
        p = p[inds_sort]
        rho_co2 = rho_co2[inds_sort]
        solub = solub[inds_sort]
        spec_vol = spec_vol[inds_sort]
    
    props = (rho_co2, solub, spec_vol)
    
    return p, props