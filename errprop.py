# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:07:34 2019

@author: Andy
"""

import numpy as np


def norm(elements):
    result = 0
    for i in range(len(elements)):
        result += (elements[i])**2
    result = np.sqrt(result)
    
    return result


def error_solubility(solubility, v_samp, w_buoy, w_gas_act, v_drop, s_v_drop, w_poly,
                   s_frac_rho_co2, w_samp_ref, s_w_samp_ref, rho_samp_ref, 
                   s_rho_samp_ref, v_samp_ref, v_drop_ref, s_v_drop_ref, v_ref, 
                   s_v_ref, s_mp1, s_zero, s_w_gas_ref):
    """
    Estimates error in solubility measurement using error propagation.
    """
    n = len(solubility)
    # error in the sample volume
    s_v_samp = error_s_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                   s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref, s_v_drop_ref)
    # error in balance reading
    s_br = norm( (s_mp1, s_zero) )
    # extract first entry that is not a nan as error in balance reading at p=0
    s_br_0 = s_br[np.logical_not(np.isnan(s_br))][0]*np.ones([n])
    
    # uncertainty in measurement of mass of gas
    s_w_gas_act = error_w_gas_act(s_br, s_br_0, w_buoy, s_frac_rho_co2, \
                                            v_samp, s_v_samp, v_ref, s_v_ref)
    
    # uncertainty in estimate of dry mass of polyol [g]
    s_w_poly = norm( (s_w_samp_ref, s_w_gas_ref) )
    
    s_solubility = solubility*norm( (s_w_gas_act/w_gas_act, s_w_gas_act/ \
                                     norm((w_gas_act, w_poly)), s_w_poly/ \
                                     norm((w_gas_act, w_poly))) )
    
    return s_solubility
    
    
def error_spec_vol(spec_vol, v_samp, w_buoy, w_gas_act, v_drop, s_v_drop, w_poly,
                   s_frac_rho_co2, w_samp_ref, s_w_samp_ref, rho_samp_ref, 
                   s_rho_samp_ref, v_samp_ref, v_drop_ref, s_v_drop_ref, v_ref, 
                   s_v_ref, s_mp1, s_zero, s_w_gas_ref):
    """
    Estimates error in specific volume measurement using error propagation.
    """
    n = len(spec_vol)
    # uncertainty in volume of sample [mL]
    s_v_samp = error_s_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref, 
                              rho_samp_ref, s_rho_samp_ref, v_drop, s_v_drop, 
                              v_drop_ref, s_v_drop_ref)
    # error in balance reading
    s_br = norm( (s_mp1, s_zero) )
    # extract first entry that is not a nan as error in balance reading at p=0
    s_br_0 = s_br[np.logical_not(np.isnan(s_br))][0]*np.ones([n])

    # uncertainty in measurement of mass of gas
    s_w_gas_act, s_w_buoy = error_w_gas_act(s_br, s_br_0, w_buoy, s_frac_rho_co2, \
                                            v_samp, s_v_samp, v_ref, s_v_ref, \
                                            return_s_w_buoy=True)
    # uncertainty in estimate of dry mass of polyol [g]
    s_w_poly = norm( (s_w_samp_ref, s_w_gas_ref) )
    # uncertainty in specific volume [mL/g]
    s_spec_vol = spec_vol*norm( (s_v_samp/v_samp, s_w_gas_act/ \
                                 norm((w_gas_act, w_poly)), s_w_poly/ \
                                 norm((w_gas_act, w_poly))) )
    
    return s_spec_vol


def error_s_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                   s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref, s_v_drop_ref):
    """
    """
    # uncertainty in volume of sample at reference point
    s_v_samp_ref = v_samp_ref*norm( (s_w_samp_ref/w_samp_ref, s_rho_samp_ref/ \
                                     rho_samp_ref) )
    # uncertainty in volume of sample [mL]
    s_v_samp = v_samp*norm( (s_v_drop/v_drop, s_v_samp_ref/v_samp_ref, \
                              s_v_drop_ref/v_drop_ref) )
    
    return s_v_samp


def error_w_gas_act(s_br, s_br_0, w_buoy, s_frac_rho_co2, \
                                            v_samp, s_v_samp, v_ref, s_v_ref, \
                                            return_s_w_buoy=False):
    """
    Estimates the error in the measurement of the actual gas weight.
    """
    # uncertainty in buoyancy correction of mass [g]
    s_w_buoy = w_buoy * norm( (s_frac_rho_co2, s_v_samp/norm((v_samp, v_ref)), \
                               s_v_ref/norm((v_samp, v_ref))) )
    s_w_gas_act = norm( (s_br, s_br_0, s_w_buoy) )
    
    if return_s_w_buoy:
        return s_w_gas_act, s_w_buoy
    else:
        return s_w_gas_act